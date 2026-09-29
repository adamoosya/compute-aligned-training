"""Internal validation and log-domain arithmetic."""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Integral

import torch
from torch import Tensor


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def budget_and_threshold(n: int, k: int) -> tuple[int, int]:
    n = positive_int(n, "n")
    k = positive_int(k, "k")
    if k > n:
        raise ValueError("k must not exceed n")
    return n, k


def work_tensor(value: Tensor, name: str) -> Tensor:
    """Preserve device/shape; promote half-precision arithmetic to float32."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{name} must be a dense tensor")
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError(f"{name} must have a real floating-point dtype")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def probability(value: Tensor, name: str = "p") -> Tensor:
    value = work_tensor(value, name)
    if not bool(((value >= 0) & (value <= 1)).all()):
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def log_probability(value: Tensor) -> Tensor:
    value = work_tensor(value, "log_p")
    if bool((value > 0).any()):
        raise ValueError("log_p must be <= 0 (natural log-probabilities, not NLLs)")
    return value


def nonnegative(value: Tensor, name: str = "weights") -> Tensor:
    value = work_tensor(value, name)
    if bool((value < 0).any()):
        raise ValueError(f"{name} must be nonnegative")
    return value


def reduce_loss(value: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.mean()
    if reduction == "sum":
        return value.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def log1mexp(value: Tensor) -> Tensor:
    """Compute log(1-exp(value)) for value <= 0, avoiding cancellation.

    Safe dummy arguments keep unselected torch.where branches from producing
    NaN derivatives. Public objectives handle their probability-one boundary.
    """
    split = -math.log(2.0)
    far = value < split
    placeholder = torch.full_like(value, split)
    far_value = torch.where(far, value, placeholder)
    near_value = torch.where(far, placeholder, value)
    return torch.where(
        far,
        torch.log1p(-torch.exp(far_value)),
        torch.log(-torch.expm1(near_value)),
    )


@lru_cache(maxsize=128)
def log_combinations(n: int) -> tuple[float, ...]:
    """CPU scalar coefficients; no cached device tensors or autograd graphs."""
    values = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
              for i in range(n + 1)]
    values[0] = values[-1] = 0.0
    return tuple(values)
