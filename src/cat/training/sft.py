"""Completion-only SFT loss and a single-process training loop with optional CUDA AMP.

The caller supplies a model and optimizer. This module never loads a model,
contacts the Hub, changes the device of model weights, or chooses a dataset.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
import math
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cat.objectives import majority_sft_loss, passn_sft_loss


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive Python integer")
    return value


@dataclass(frozen=True)
class SFTObjective:
    strategy: str = "ce"
    n: int = 1
    k: int | None = None
    reduction: str = "sequence_mean"

    def __post_init__(self) -> None:
        if self.strategy not in {"ce", "passn", "majority"}:
            raise ValueError("strategy must be 'ce', 'passn', or 'majority'")
        _positive_int(self.n, "n")
        if self.strategy == "majority":
            _positive_int(self.k, "k")
            if self.k > self.n:
                raise ValueError("k must not exceed n")
        elif self.k is not None:
            raise ValueError("k is only used by the majority strategy")
        if self.strategy == "ce" and self.n != 1:
            raise ValueError("Use n=1 for the CE baseline")
        if self.reduction not in {"sequence_mean", "token_mean"}:
            raise ValueError("reduction must be 'sequence_mean' or 'token_mean'")


@dataclass(frozen=True)
class SequenceScores:
    log_prob: Tensor
    target_tokens: Tensor


@dataclass(frozen=True)
class SFTStats:
    loss: float
    sequences: int
    target_tokens: int
    optimizer_steps: int
    skipped_steps: int = 0


def _check_labels(labels: Tensor, attention_mask: Tensor | None) -> Tensor:
    if not isinstance(labels, Tensor) or labels.ndim != 2 or labels.dtype != torch.long:
        raise ValueError("labels must be a [batch, length] int64 tensor")
    if labels.shape[0] == 0 or labels.shape[1] < 2:
        raise ValueError("labels need a nonempty batch and at least two positions")
    if bool((labels[:, 0] != -100).any()):
        raise ValueError("The first position has no causal predictor; its label must be -100")
    if attention_mask is not None:
        if attention_mask.shape != labels.shape or attention_mask.device != labels.device:
            raise ValueError("attention_mask must match labels in shape and device")
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError("attention_mask must be binary")
        if bool(((labels != -100) & (attention_mask == 0)).any()):
            raise ValueError("Padding positions must have label -100")
        if bool(((labels[:, 1:] != -100) & (attention_mask[:, :-1] == 0)).any()):
            raise ValueError("A target token cannot be predicted from a padding position")
    counts = (labels[:, 1:] != -100).sum(dim=-1)
    if bool((counts == 0).any()):
        raise ValueError("Every sequence must contain at least one supervised target token")
    return counts


def completion_log_probs(logits: Tensor, labels: Tensor,
                         attention_mask: Tensor | None = None) -> SequenceScores:
    """Sum target token log-probabilities with exactly one causal shift.

    Prompt and padding labels must be -100. The sum, never token-mean NLL,
    supplies p to the CAT objective. Float16/bfloat16 logits are promoted.
    """
    counts = _check_labels(labels, attention_mask)
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [batch, length, vocabulary]")
    if logits.shape[:2] != labels.shape or logits.shape[-1] < 2:
        raise ValueError("logits and labels must have matching batch/length dimensions")
    if logits.device != labels.device:
        raise ValueError("logits and labels must be on the same device")
    if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("logits must have a floating-point dtype")
    invalid = (labels != -100) & ((labels < 0) | (labels >= logits.shape[-1]))
    if bool(invalid.any()):
        raise ValueError("Target label is outside the model vocabulary")
    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    token_nll = F.cross_entropy(
        work[:, :-1, :].reshape(-1, work.shape[-1]), labels[:, 1:].reshape(-1),
        ignore_index=-100, reduction="none",
    ).reshape(labels.shape[0], labels.shape[1] - 1)
    log_prob = -token_nll.sum(dim=-1)
    if not bool(torch.isfinite(log_prob).all()):
        raise ValueError("Nonfinite completion log-probability")
    return SequenceScores(log_prob, counts)


def _per_sequence_loss(scores: SequenceScores, objective: SFTObjective) -> Tensor:
    if objective.strategy == "ce":
        return -scores.log_prob
    if objective.strategy == "passn":
        return passn_sft_loss(scores.log_prob, objective.n, reduction="none")
    return majority_sft_loss(scores.log_prob, objective.n, objective.k, reduction="none")


def sft_loss(logits: Tensor, labels: Tensor, objective: SFTObjective,
             attention_mask: Tensor | None = None) -> Tensor:
    """Differentiate the scalar CAT loss directly, not w(p) times a CE loss."""
    scores = completion_log_probs(logits, labels, attention_mask)
    denominator = scores.log_prob.numel() if objective.reduction == "sequence_mean" else scores.target_tokens.sum()
    return _per_sequence_loss(scores, objective).sum() / denominator


def _batch_on_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    result = {}
    for key in ("input_ids", "attention_mask", "labels"):
        value = batch.get(key)
        if not isinstance(value, Tensor):
            raise ValueError(f"Batch is missing tensor {key!r}")
        result[key] = value.to(device)
    if result["input_ids"].dtype != torch.long or result["input_ids"].shape != result["labels"].shape:
        raise ValueError("input_ids must match labels in shape and have int64 dtype")
    return result


def _model_logits(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    if isinstance(outputs, Tensor):
        return outputs
    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    else:
        logits = getattr(outputs, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("Model must return logits as a Tensor, mapping, or .logits attribute")
    return logits


def _device(model: nn.Module, requested: str | torch.device | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("Model has no parameters; supply a trainable causal model") from error


def _precision_context(device: torch.device, precision: str):
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed precision in this training loop requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_sft_epoch(
    model: nn.Module, batches: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer, objective: SFTObjective, *,
    accumulation_steps: int = 1, max_grad_norm: float | None = None,
    device: str | torch.device | None = None, scheduler: Any = None,
    precision: str = "fp32", scaler: Any = None,
) -> SFTStats:
    """One epoch; normalize over the actual effective batch, including its tail.

    For token_mean, each accumulation window divides summed sequence losses by
    the total target token count in that window. This is not an average of
    microbatch averages. CUDA fp16 requires a caller-owned GradScaler; bf16
    uses autocast without scaling. Distributed training is not supported.
    """
    _positive_int(accumulation_steps, "accumulation_steps")
    if max_grad_norm is not None:
        if isinstance(max_grad_norm, bool) or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
    target_device = _device(model, device)
    _precision_context(target_device, precision)  # Validate before changing model state.
    if precision == "fp16" and (scaler is None or not scaler.is_enabled()):
        raise ValueError("fp16 training requires an enabled CUDA GradScaler")
    if precision != "fp16" and scaler is not None:
        raise ValueError("A GradScaler is only accepted with fp16 training")
    model.train()
    iterator = iter(batches)
    trainable = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    if not trainable:
        raise ValueError("Optimizer has no trainable parameters")
    total_numerator = 0.0
    sequences = tokens = steps = skipped_steps = 0
    try:
        while True:
            # Prefetch CPU batches, not computation graphs or extra GPU copies.
            window = list(islice(iterator, accumulation_steps))
            if not window:
                break
            counts = [_check_labels(b["labels"], b["attention_mask"]) for b in window]
            window_sequences = sum(c.numel() for c in counts)
            window_tokens = sum(int(c.sum().item()) for c in counts)
            denominator = window_sequences if objective.reduction == "sequence_mean" else window_tokens
            optimizer.zero_grad(set_to_none=True)
            for cpu_batch in window:
                batch = _batch_on_device(cpu_batch, target_device)
                with _precision_context(target_device, precision):
                    logits = _model_logits(model, batch)
                scores = completion_log_probs(logits, batch["labels"], batch["attention_mask"])
                numerator = _per_sequence_loss(scores, objective).sum()
                if not bool(torch.isfinite(numerator)):
                    raise ValueError("Nonfinite SFT loss; optimizer step was not taken")
                scaled_loss = numerator / denominator
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                total_numerator += float(numerator.detach().item())
            if scaler is not None:
                scaler.unscale_(optimizer)
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=scaler is None)
            elif scaler is None and any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in trainable):
                raise ValueError("Nonfinite gradient; optimizer step was not taken")
            took_step = True
            if scaler is None:
                optimizer.step()
            else:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                took_step = scaler.get_scale() >= previous_scale
            if took_step:
                if scheduler is not None:
                    scheduler.step()
                steps += 1
            else:
                skipped_steps += 1
            sequences += window_sequences
            tokens += window_tokens
    finally:
        optimizer.zero_grad(set_to_none=True)
    if sequences == 0:
        raise ValueError("Training loader is empty")
    denominator = sequences if objective.reduction == "sequence_mean" else tokens
    return SFTStats(total_numerator / denominator, sequences, tokens, steps, skipped_steps)


def evaluate_sft_loss(
    model: nn.Module, batches: Iterable[Mapping[str, Tensor]], objective: SFTObjective,
    *, device: str | torch.device | None = None, precision: str = "fp32",
) -> SFTStats:
    """Teacher-forced loss only; this is not Pass@k or Majority Vote accuracy."""
    target_device = _device(model, device)
    _precision_context(target_device, precision)
    prior_mode = model.training
    total_numerator = 0.0
    sequences = tokens = 0
    try:
        model.eval()
        with torch.no_grad():
            for cpu_batch in batches:
                batch = _batch_on_device(cpu_batch, target_device)
                with _precision_context(target_device, precision):
                    logits = _model_logits(model, batch)
                scores = completion_log_probs(logits, batch["labels"], batch["attention_mask"])
                total_numerator += float(_per_sequence_loss(scores, objective).sum().item())
                sequences += scores.log_prob.numel()
                tokens += int(scores.target_tokens.sum().item())
    finally:
        model.train(prior_mode)
    if sequences == 0:
        raise ValueError("Evaluation loader is empty")
    denominator = sequences if objective.reduction == "sequence_mean" else tokens
    return SFTStats(total_numerator / denominator, sequences, tokens, 0)
