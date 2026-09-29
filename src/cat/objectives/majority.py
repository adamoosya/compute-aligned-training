"""The fixed-threshold Majority Vote surrogate from Table 1 and Appendix A.3.

These functions evaluate Pr[Binomial(n,p) >= k], not a full plurality
probability over the rival-answer distribution. k is a fixed integer per call.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ._numerics import (
    budget_and_threshold, log1mexp, log_combinations, log_probability,
    probability, reduce_loss,
)
from .passn import (
    passn_log_probability, passn_probability, passn_rl_weight,
    passn_sft_weight_from_logp,
)


def _upper_remainder(log_p: Tensor, n: int, k: int) -> Tensor:
    """Log upper-tail sum after factoring out p**k."""
    counts = torch.arange(k, n + 1, device=log_p.device, dtype=log_p.dtype)
    coeffs = log_p.new_tensor(log_combinations(n)[k:])
    terms = (coeffs + (counts - k) * log_p.unsqueeze(-1)
             + (n - counts) * log1mexp(log_p).unsqueeze(-1))
    return torch.logsumexp(terms, dim=-1)


def _log_failure(log_p: Tensor, n: int, k: int) -> Tensor:
    """Log Pr[Binomial(n,p) < k], factoring out (1-p)**(n-k+1)."""
    counts = torch.arange(k, device=log_p.device, dtype=log_p.dtype)
    coeffs = log_p.new_tensor(log_combinations(n)[:k])
    log_q = log1mexp(log_p)
    terms = (coeffs + counts * log_p.unsqueeze(-1)
             + (k - 1 - counts) * log_q.unsqueeze(-1))
    return (n - k + 1) * log_q + torch.logsumexp(terms, dim=-1)


def majority_log_probability(log_p: Tensor, n: int, k: int) -> Tensor:
    """Log binomial upper tail, with stable lower- and upper-tail branches."""
    log_p = log_probability(log_p)
    n, k = budget_and_threshold(n, k)
    if k == 1:
        return passn_log_probability(log_p, n)
    if k == n:
        return n * log_p

    # Use a small tail directly; compute a near-one success from its complement.
    pivot = math.log(k / (n + 1))
    certain = log_p == 0
    safe = torch.where(certain, torch.full_like(log_p, pivot), log_p)
    use_upper = safe < pivot
    upper_arg = torch.where(use_upper, safe, torch.full_like(safe, pivot))
    lower_arg = torch.where(use_upper, torch.full_like(safe, pivot), safe)
    upper = k * upper_arg + _upper_remainder(upper_arg, n, k)
    complement = log1mexp(_log_failure(lower_arg, n, k))
    result = torch.where(use_upper, upper, complement)
    return torch.where(certain, torch.zeros_like(result), result)


def majority_probability(p: Tensor, n: int, k: int) -> Tensor:
    """Pr[Binomial(n,p) >= k] for p in [0,1]."""
    p = probability(p)
    n, k = budget_and_threshold(n, k)
    if k == 1:
        return passn_probability(p, n)
    if k == n:
        return p.pow(n)
    zero = p == 0
    safe = torch.where(zero, torch.full_like(p, 0.5), p)
    result = torch.exp(majority_log_probability(torch.log(safe), n, k))
    return torch.where(zero, torch.zeros_like(result), result)


def majority_sft_loss(log_p: Tensor, n: int, k: int, *, reduction: str = "mean") -> Tensor:
    """Negative log fixed-threshold success, from finite sequence log(p)."""
    return reduce_loss(-majority_log_probability(log_p, n, k), reduction)


def majority_sft_weight_from_logp(log_p: Tensor, n: int, k: int) -> Tensor:
    """SFT factor k*C(n,k)*p**k*(1-p)**(n-k) divided by the upper tail."""
    log_p = log_probability(log_p)
    n, k = budget_and_threshold(n, k)
    if k == 1:
        return passn_sft_weight_from_logp(log_p, n)
    if k == n:
        return torch.full_like(log_p, float(n))
    certain = log_p == 0
    safe = torch.where(certain, torch.full_like(log_p, -math.log(2.0)), log_p)
    # Cancel p**k analytically, rather than subtracting two enormous log values.
    log_w = (math.log(k) + log_combinations(n)[k]
             + (n - k) * log1mexp(safe) - _upper_remainder(safe, n, k))
    result = torch.exp(log_w)
    return torch.where(certain, torch.zeros_like(result), result)


def majority_sft_weight(p: Tensor, n: int, k: int) -> Tensor:
    """SFT factor for p in [0,1], with its continuous extension w(0)=k."""
    p = probability(p)
    n, k = budget_and_threshold(n, k)
    zero = p == 0
    safe = torch.where(zero, torch.full_like(p, 0.5), p)
    result = majority_sft_weight_from_logp(torch.log(safe), n, k)
    return torch.where(zero, torch.full_like(result, float(k)), result)


def majority_rl_weight(p: Tensor, n: int, k: int) -> Tensor:
    """Raw factor n*C(n-1,k-1)*p**(k-1)*(1-p)**(n-k)."""
    p = probability(p)
    n, k = budget_and_threshold(n, k)
    if k == 1:
        return passn_rl_weight(p, n)
    if k == n:
        return n * p.pow(n - 1)
    boundary = (p == 0) | (p == 1)
    safe = torch.where(boundary, torch.full_like(p, 0.5), p)
    log_w = (math.log(k) + log_combinations(n)[k]
             + (k - 1) * torch.log(safe) + (n - k) * torch.log1p(-safe))
    return torch.where(boundary, torch.zeros_like(p), torch.exp(log_w))
