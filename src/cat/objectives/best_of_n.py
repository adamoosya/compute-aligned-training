"""Best-of-N rank weight used in the paper (Table 1 and Appendix B.2.3)."""

from __future__ import annotations

from torch import Tensor

from ._numerics import positive_int, probability


def best_of_n_rl_weight(quantile: Tensor, n: int) -> Tensor:
    """Return n*quantile**(n-1), where quantile is P[R(Y') < R(y)].

    The caller supplies the lower-tail probability. This function does not
    estimate ranks, resolve reward ties, normalize weights, or clip them.
    It implements the paper's rank weight, not a derivative of the full
    expected-maximum reward with respect to arbitrary model parameters.
    """
    quantile = probability(quantile, "quantile")
    n = positive_int(n, "n")
    return n * quantile.pow(n - 1)
