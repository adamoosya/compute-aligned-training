"""Shared CAT-weighted policy-gradient updates for MATH and protein experiments.

Rollout collection is separate. An update receives a complete logical batch of
prompts, estimates detached weights once, then backpropagates in small chunks.
"""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Any
import torch
from torch import Tensor
from torch.nn import functional as F
from cat.objectives import (passn_rl_weight, passn_sft_weight, majority_rl_weight,
                            majority_sft_weight, best_of_n_rl_weight)
from cat.objectives.weighting import normalize_sample_weights
from cat.training.sft import _check_labels, _model_logits, _precision_context


def _positive(n: Any, name: str, minimum: int = 1):
    if type(n) is not int or n < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(x: Tensor, name: str, ndim: int | None = None):
    if not isinstance(x, Tensor) or not x.is_floating_point() or not x.numel():
        raise ValueError(f"{name} must be a nonempty floating tensor")
    if ndim is not None and x.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} contains nonfinite values")


@dataclass(frozen=True)
class CATWeightConfig:
    strategy: str = "standard"
    n: int = 1
    form: str = "raw"
    probability: str = "empirical"
    normalize: bool = False
    clip: float | None = None
    quantile: str = "within_prompt"

    def __post_init__(self):
        _positive(self.n, "n")
        if self.strategy not in {"standard", "passn", "majority", "bon"}:
            raise ValueError("Unknown CAT strategy")
        if self.form not in {"raw", "log"} or self.probability not in {"sequence", "empirical"}:
            raise ValueError("Invalid weight form or probability estimator")
        if self.strategy == "standard" and self.n != 1:
            raise ValueError("Standard baseline requires n=1")
        if self.strategy == "majority" and self.n < 2:
            raise ValueError("Dynamic Majority Vote requires n>=2")
        if self.strategy == "bon" and self.form != "raw":
            raise ValueError("BoN uses the paper's raw rank weight")
        if type(self.normalize) is not bool:
            raise ValueError("normalize must be bool")
        if self.quantile not in {"within_prompt", "history"}:
            raise ValueError("quantile must be within_prompt or history")
        if self.clip is not None and (isinstance(self.clip, bool) or not math.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("clip must be positive or null")


@dataclass(frozen=True)
class RLUpdateConfig:
    surrogate: str = "reinforce"
    clip_ratio: float = .2
    beta: float = 0.0
    advantage_epsilon: float = 1e-4
    advantage_clip: float | None = None
    all_wrong_penalty: float = 0.0
    forward_batch_size: int = 1
    max_grad_norm: float = 1.0
    precision: str = "fp32"

    def __post_init__(self):
        if self.surrogate not in {"reinforce", "clipped"}:
            raise ValueError("surrogate must be reinforce or clipped")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Invalid precision")
        _positive(self.forward_batch_size, "forward_batch_size")
        for name in ("clip_ratio", "beta", "advantage_epsilon", "all_wrong_penalty", "max_grad_norm"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"Invalid {name}")
        if not 0 < self.clip_ratio < 1 or self.advantage_epsilon == 0 or self.max_grad_norm == 0:
            raise ValueError("Invalid clipping or epsilon")
        if self.advantage_clip is not None and (not math.isfinite(self.advantage_clip) or self.advantage_clip <= 0):
            raise ValueError("advantage_clip must be positive or null")


def group_advantages(rewards: Tensor, *, epsilon: float = 1e-4,
                     all_wrong_penalty: float = 0.0, clip: float | None = None) -> Tensor:
    _finite(rewards, "rewards", 2)
    if rewards.shape[1] < 2:
        raise ValueError("Group-relative advantages require at least two rollouts per prompt")
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(all_wrong_penalty) or all_wrong_penalty < 0:
        raise ValueError("Invalid advantage settings")
    if all_wrong_penalty and not bool(((rewards == 0) | (rewards == 1)).all()):
        raise ValueError("all_wrong_penalty is defined only for binary rewards")
    r = rewards.detach()
    a = (r - r.mean(1, keepdim=True)) / (r.std(1, correction=1, keepdim=True) + epsilon)
    a = a - (r.eq(0).all(1, keepdim=True) * all_wrong_penalty)
    if clip is not None:
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("advantage clip must be positive")
        a = a.clamp(-clip, clip)
    return a


def adaptive_majority_k(keys: list[str | None], correct: list[bool], n: int) -> int:
    _positive(n, "n", 2)
    if len(keys) != len(correct) or not keys or any(type(c) is not bool for c in correct):
        raise ValueError("Aligned nonempty answer keys and correctness labels required")
    if any(k is not None and not isinstance(k, str) for k in keys):
        raise ValueError("Answer keys must be strings or None")
    counts = Counter(k for k, c in zip(keys, correct) if not c)
    strongest = max(counts.values(), default=0)
    # Missing parses form one wrong-answer category, matching the source's empty key.
    return max(2, min(n // 2 + 1, (n * strongest) // len(keys) + 1))


class RewardHistory:
    """Unconditional rewards only; query before append avoids batch-order leakage."""
    def __init__(self, capacity: int = 4000):
        _positive(capacity, "capacity")
        self.values: deque[float] = deque(maxlen=capacity)

    def quantiles(self, rewards: Tensor) -> Tensor:
        _finite(rewards, "rewards", 2)
        history = torch.tensor(list(self.values), dtype=rewards.dtype, device=rewards.device)
        counts = (history[None, None, :] < rewards[:, :, None]).sum(-1)
        return ((counts.to(rewards.dtype) + 1) / (len(history) + 2)).clamp(.001, .999)

    def append(self, rewards: Tensor):
        _finite(rewards, "rewards")
        self.values.extend(rewards.detach().cpu().reshape(-1).tolist())


def within_prompt_quantiles(rewards: Tensor) -> Tensor:
    _finite(rewards, "rewards", 2)
    # Strict-less rank: equal rewards have equal weights. Never rank different prompts together.
    rank = (rewards[:, None, :] < rewards[:, :, None]).sum(-1)
    return (rank.to(rewards.dtype) + 1) / (rewards.shape[1] + 1)


def cat_weights(rewards: Tensor, old_logp: Tensor, config: CATWeightConfig, *,
                answer_keys: list[list[str | None]] | None = None,
                history: RewardHistory | None = None) -> tuple[Tensor, dict]:
    _finite(rewards, "rewards", 2)
    _finite(old_logp, "old_logp", 2)
    if rewards.shape != old_logp.shape or rewards.device != old_logp.device or bool((old_logp > 0).any()):
        raise ValueError("rewards and nonpositive old_logp must align")
    b, m = rewards.shape
    thresholds = None
    with torch.no_grad():
        if config.strategy == "standard":
            weights = torch.ones_like(rewards)
        elif config.strategy == "bon":
            if config.quantile == "history":
                if history is None:
                    raise ValueError("Unconditional BoN needs a reward history")
                q = history.quantiles(rewards)
            else:
                q = within_prompt_quantiles(rewards)
            weights = best_of_n_rl_weight(q, config.n)
        else:
            if not bool(((rewards == 0) | (rewards == 1)).all()):
                raise ValueError("MATH weights require binary rewards")
            if config.probability == "sequence":
                # Explicit Appendix J floor, evaluated in float64 to preserve 1-p near 1.
                p = old_logp.double().clamp(-50, 0).exp().clamp(1e-10, 1 - 1e-10)
            else:
                p = ((rewards.double().sum(1, keepdim=True) + 1) / (m + 2)).expand(b, m)
            if config.strategy == "passn":
                fn = passn_rl_weight if config.form == "raw" else passn_sft_weight
                weights = fn(p, config.n)
            else:
                if answer_keys is None or len(answer_keys) != b:
                    raise ValueError("Majority weights need answer keys for every prompt")
                thresholds = [adaptive_majority_k(keys, r.bool().tolist(), config.n)
                              for keys, r in zip(answer_keys, rewards)]
                fn = majority_rl_weight if config.form == "raw" else majority_sft_weight
                weights = torch.stack([fn(p[i], config.n, k) for i, k in enumerate(thresholds)])
            weights = weights.to(rewards.dtype)
        if config.clip is not None:
            weights = weights.clamp(max=config.clip)
        raw = weights.clone()
        # No division by one prompt's weight. At a short one-prompt final batch keep it raw.
        can_normalize = config.strategy == "bon" or b >= 2 or config.probability == "sequence"
        normalized = config.normalize and config.strategy != "standard" and can_normalize and bool((weights > 0).any())
        if normalized:
            weights = normalize_sample_weights(weights)
    return weights.detach(), {"raw_mean": float(raw.mean()), "mean": float(weights.mean()),
                               "min": float(weights.min()), "max": float(weights.max()),
                               "normalized": normalized, "thresholds": thresholds}


@dataclass
class RolloutBatch:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    rewards: Tensor  # [prompts, rollouts]; token rows are prompt-major.
    answer_keys: list[list[str | None]] | None = None

    def validate(self):
        _finite(self.rewards, "rewards", 2)
        b, m = self.rewards.shape
        if m < 2 or self.input_ids.ndim != 2 or self.input_ids.dtype != torch.long:
            raise ValueError("Expected int64 token rows and at least two rollouts per prompt")
        if self.input_ids.shape != self.labels.shape or self.input_ids.shape[0] != b * m:
            raise ValueError("Token rows must match all prompt/rollout pairs")
        if self.attention_mask.shape != self.input_ids.shape or bool((self.input_ids < 0).any()):
            raise ValueError("Invalid token IDs or mask")
        _check_labels(self.labels, self.attention_mask)
        if bool(((self.labels != -100) & (self.labels != self.input_ids)).any()):
            raise ValueError("Completion labels must be the actual generated tokens")


def token_log_probs(model, batch: dict[str, Tensor], precision: str) -> Tensor:
    _check_labels(batch["labels"], batch["attention_mask"])
    with _precision_context(batch["input_ids"].device, precision):
        logits = _model_logits(model, batch)
    logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    labels = batch["labels"][:, 1:]
    nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels.reshape(-1),
                          ignore_index=-100, reduction="none").reshape(labels.shape)
    result = -nll
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite rollout log probabilities")
    return result


def policy_loss_terms(current: Tensor, old: Tensor, reference: Tensor | None, mask: Tensor,
                      advantages: Tensor, weights: Tensor, config: RLUpdateConfig) -> Tensor:
    """One scalar per completion; policy sum and mean-token KL are explicit reductions."""
    for name, v in (("current", current), ("old", old), ("advantages", advantages), ("weights", weights)):
        _finite(v, name)
    if current.ndim != 2 or current.shape != old.shape or mask.shape != current.shape:
        raise ValueError("Token score shapes do not match")
    if advantages.shape != current.shape[:1] or weights.shape != advantages.shape:
        raise ValueError("One advantage and weight per completion required")
    if not bool(((mask == 0) | (mask == 1)).all()) or bool((mask.sum(1) == 0).any()) or bool((weights < 0).any()):
        raise ValueError("Invalid completion mask or weights")
    if current.device != old.device or current.device != advantages.device or current.device != weights.device:
        raise ValueError("Policy inputs must share a device")
    a = advantages.detach() * weights.detach()
    seq = (current * mask).sum(1)
    if config.surrogate == "reinforce":
        terms = -seq * a
    else:
        # Section C's sequence-level ratio, not an unannounced token-averaged variant.
        ratio = ((current - old.detach()) * mask).sum(1).exp()
        terms = -torch.minimum(ratio * a, ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * a)
    if config.beta:
        if reference is None or reference.shape != current.shape:
            raise ValueError("Positive beta requires fixed-reference token log probabilities")
        _finite(reference, "reference")
        delta = reference.detach() - current
        kl = (torch.expm1(delta) - delta) * mask
        terms = terms + config.beta * kl.sum(1) / mask.sum(1)
    if not bool(torch.isfinite(terms).all()):
        raise ValueError("Nonfinite RL loss; no optimizer update is allowed")
    return terms


def disable_dropout(model):
    """Deterministic old/current likelihoods while retaining train-mode checkpointing."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    # Mistral uses this configuration field for functional attention dropout.
    if hasattr(model, "config") and hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = 0.0


def update_rl(model, optimizer, rollouts: RolloutBatch, weight_config: CATWeightConfig,
              update_config: RLUpdateConfig, *, reference_model=None, history=None,
              scaler=None, scheduler=None) -> dict:
    """Exactly one logical optimizer update; chunk size never changes loss reduction."""
    rollouts.validate()
    device = next(model.parameters()).device
    _precision_context(device, update_config.precision)
    if ((update_config.precision == "fp16") != (scaler is not None)
            or (scaler is not None and not scaler.is_enabled())):
        raise ValueError("fp16 requires a persistent scaler; other precisions do not")
    if update_config.beta and reference_model is None:
        raise ValueError("Positive beta needs a reference model")
    if reference_model is model:
        raise ValueError("Reference must be a distinct frozen model")
    if reference_model is not None and any(p.requires_grad for p in reference_model.parameters()):
        raise ValueError("Reference model must be frozen")
    trainable = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    if not trainable:
        raise ValueError("Optimizer has no trainable parameters")
    disable_dropout(model)
    model.eval()
    if reference_model is not None:
        reference_model.eval()
        if hasattr(reference_model, "config"):
            reference_model.config.use_cache = False
    size = rollouts.input_ids.shape[0]
    chunk = update_config.forward_batch_size
    def get_batch(start):
        return {k: getattr(rollouts, k)[start:start + chunk].to(device)
                for k in ("input_ids", "attention_mask", "labels")}
    old, ref = [], []
    with torch.no_grad():
        for start in range(0, size, chunk):
            batch = get_batch(start)
            old.append(token_log_probs(model, batch, update_config.precision).cpu())
            if update_config.beta:
                ref.append(token_log_probs(reference_model, batch, update_config.precision).cpu())
    old = torch.cat(old)
    reference = torch.cat(ref) if ref else None
    rewards = rollouts.rewards.detach().cpu()
    weights, weight_stats = cat_weights(rewards, old.sum(1).reshape(rewards.shape), weight_config,
                                        answer_keys=rollouts.answer_keys, history=history)
    advantages = group_advantages(rewards, epsilon=update_config.advantage_epsilon,
                                  all_wrong_penalty=update_config.all_wrong_penalty,
                                  clip=update_config.advantage_clip)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    disable_dropout(model)
    total_loss = 0.0
    try:
        for start in range(0, size, chunk):
            end = min(size, start + chunk)
            batch = get_batch(start)
            current = token_log_probs(model, batch, update_config.precision)
            terms = policy_loss_terms(current, old[start:end].to(device),
                      reference[start:end].to(device) if reference is not None else None,
                      (batch["labels"][:, 1:] != -100), advantages.reshape(-1)[start:end].to(device),
                      weights.reshape(-1)[start:end].to(device), update_config)
            loss = terms.sum() / size
            total_loss += float(loss.detach())
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(trainable, update_config.max_grad_norm,
                                              error_if_nonfinite=scaler is None)
        if scaler is None:
            optimizer.step()
            skipped = False
        else:
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < before
        if scheduler is not None and not skipped:
            scheduler.step()
        if history is not None:
            history.append(rewards)
    finally:
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss, "reward_mean": float(rewards.mean()), "prompts": rewards.shape[0],
            "rollouts": size, "gradient_norm": float(norm) if torch.isfinite(norm) else None,
            "skipped": skipped, "weights": weight_stats}
