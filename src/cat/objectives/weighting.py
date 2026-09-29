"""Explicit normalization and a detached weighted policy-gradient loss term."""

from __future__ import annotations

import torch
from torch import Tensor

from ._numerics import log_probability, nonnegative, reduce_loss, work_tensor


def _mean_normalize(weights: Tensor) -> Tensor:
    if not bool((weights > 0).any()):
        raise ValueError("Cannot mean-normalize all-zero weights; skip or use them unnormalized")
    # Rescaling first prevents overflow in a mean and preserves tiny weights.
    scaled = weights.detach() / weights.detach().max()
    return scaled / scaled.mean()


def normalize_prompt_weights(weights: Tensor) -> Tensor:
    """Mean-normalize one weight per distinct prompt, before rollout expansion.

    Input must be [num_prompts] with at least two prompts. For a single prompt,
    keep the raw weight rather than dividing it by itself. With unequal rollout
    counts, an explicit per-sample normalization defines a different reduction.
    """
    weights = nonnegative(weights)
    if weights.ndim != 1 or weights.numel() < 2:
        raise ValueError("Prompt normalization needs a 1-D tensor with at least two distinct prompts")
    return _mean_normalize(weights)


def normalize_sample_weights(weights: Tensor) -> Tensor:
    """Mean-normalize genuinely sample-specific weights, e.g. BoN ranks.

    Do not pass repeated copies of one prompt-level weight to this function.
    Use normalize_prompt_weights before expanding prompt weights to rollouts.
    """
    weights = nonnegative(weights)
    if weights.numel() < 2:
        raise ValueError("Sample normalization needs at least two samples")
    return _mean_normalize(weights)


def weighted_policy_loss(
    log_probs: Tensor, advantages: Tensor, weights: Tensor, *, reduction: str = "mean"
) -> Tensor:
    """Return -mean(stopgrad(weights*advantages) * log_probs).

    All three inputs have the same shape. log_probs contains completion
    log-probabilities (already summed/masked by the caller). This is the
    REINFORCE loss term, not a complete PPO/GRPO objective: it adds no clipping,
    importance ratio, KL penalty, advantage estimation, or normalization.
    """
    log_probs = log_probability(log_probs)
    advantages = work_tensor(advantages, "advantages")
    weights = nonnegative(weights)
    if log_probs.shape != advantages.shape or log_probs.shape != weights.shape:
        raise ValueError("log_probs, advantages, and weights must have the same shape")
    if log_probs.device != advantages.device or log_probs.device != weights.device:
        raise ValueError("log_probs, advantages, and weights must be on the same device")
    return reduce_loss(-log_probs * advantages.detach() * weights.detach(), reduction)
