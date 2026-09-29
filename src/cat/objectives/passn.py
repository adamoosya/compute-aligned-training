"""Pass@N probabilities, SFT losses, and the paper's SFT/RL weights.

Probability APIs take p in [0, 1]. Log-domain APIs take finite natural log(p)
<= 0, which need not be exponentiable in the working dtype.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ._numerics import log1mexp, log_probability, positive_int, probability, reduce_loss


def passn_probability(p: Tensor, n: int) -> Tensor:
    """Return 1-(1-p)**n without subtracting nearly equal floating-point values."""
    p = probability(p)
    n = positive_int(n, "n")
    if n == 1:
        return p
    certain = p == 1
    safe = torch.where(certain, torch.full_like(p, 0.5), p)
    result = -torch.expm1(n * torch.log1p(-safe))
    return torch.where(certain, torch.ones_like(result), result)


def passn_log_probability(log_p: Tensor, n: int) -> Tensor:
    """Return log(1-(1-p)**n), retaining finite losses for very small p.

    When n*p is below machine epsilon, log(n)+log(p) has error below
    floating-point precision. This branch also avoids exponentiating extremely
    negative sequence log-probabilities. No probability floor is applied.
    """
    log_p = log_probability(log_p)
    n = positive_int(n, "n")
    if n == 1:
        return log_p
    tiny = log_p < math.log(torch.finfo(log_p.dtype).eps) - math.log(n)
    certain = log_p == 0
    safe = torch.where(tiny | certain, torch.full_like(log_p, -math.log(2.0)), log_p)
    regular = log1mexp(n * log1mexp(safe))
    result = torch.where(tiny, log_p + math.log(n), regular)
    return torch.where(certain, torch.zeros_like(result), result)


def passn_sft_loss(log_p: Tensor, n: int, *, reduction: str = "mean") -> Tensor:
    """Negative log Pass@N success, from sequence log-probabilities.

    Reduction is over the provided samples, not over their token lengths.
    Token masking and any explicit length normalization belong to the trainer.
    """
    return reduce_loss(-passn_log_probability(log_p, n), reduction)


def passn_sft_weight_from_logp(log_p: Tensor, n: int) -> Tensor:
    """Return n*p*(1-p)**(n-1)/(1-(1-p)**n) in log space."""
    log_p = log_probability(log_p)
    n = positive_int(n, "n")
    if n == 1:
        return torch.ones_like(log_p)
    tiny = log_p < math.log(torch.finfo(log_p.dtype).eps) - math.log(n)
    certain = log_p == 0
    safe = torch.where(tiny | certain, torch.full_like(log_p, -math.log(2.0)), log_p)
    log_w = math.log(n) + safe + (n - 1) * log1mexp(safe) - passn_log_probability(safe, n)
    result = torch.where(tiny, torch.ones_like(log_p), torch.exp(log_w))
    return torch.where(certain, torch.zeros_like(result), result)


def passn_sft_weight(p: Tensor, n: int) -> Tensor:
    """SFT weight for p in [0,1]; use its continuous extension w(0)=1."""
    p = probability(p)
    n = positive_int(n, "n")
    zero = p == 0
    safe = torch.where(zero, torch.full_like(p, 0.5), p)
    result = passn_sft_weight_from_logp(torch.log(safe), n)
    return torch.where(zero, torch.ones_like(result), result)


def passn_rl_weight(p: Tensor, n: int) -> Tensor:
    """Raw RL weight n*(1-p)**(n-1); no normalization or clipping."""
    p = probability(p)
    n = positive_int(n, "n")
    return n * (1 - p).pow(n - 1)
