"""Forced-vote sensitivities on explicit finite-support distributions.

These are conditional probability diagnostics, not parameter-gradient checks.
"""
from __future__ import annotations
import math
import numpy as np


def forced_sensitivities(probabilities, n, *, target=0, strategy="plurality", rewards=None,
                         trials=5000, seed=42):
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or (p < 0).any()
            or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("Supply a finite categorical probability vector summing to one")
    if type(n) is not int or n < 1 or type(trials) is not int or trials < 1:
        raise ValueError("n and trials must be positive integers")
    if type(seed) is not int or seed < 0 or type(target) is not int or not 0 <= target < len(p):
        raise ValueError("Invalid seed or target")
    if not 0 < p[target] < 1:
        raise ValueError("The target probability must be strictly between zero and one")
    if strategy not in {"plurality", "best_of_n"}:
        raise ValueError("Unknown strategy")
    r = None
    if strategy == "best_of_n":
        r = np.asarray(rewards, dtype=float)
        if r.shape != p.shape or not np.isfinite(r).all() or len(np.unique(r)) != len(r):
            raise ValueError("Best-of-N diagnostic requires a distinct finite reward per category")
    # One shared set of competitors preserves the event-containment inequality.
    m = np.random.default_rng(seed).multinomial(n-1, p, size=trials)
    sensitivities=[]
    other = np.arange(len(p)) != target
    for j in range(len(p)):
        counts=m.copy(); counts[:, j] += 1
        if strategy == "plurality":
            wins = counts[:, target] > counts[:, other].max(axis=1)
        else:
            better = r > r[target]
            wins = (counts[:, target] > 0) & (counts[:, better].sum(axis=1) == 0)
        sensitivities.append(float(n * wins.mean()))
    a = sensitivities[target]
    correction = float(np.dot(p[other], np.asarray(sensitivities)[other]) / (1-p[target]))
    total = a - correction
    return {"kind":"finite_support_diagnostic","strategy":strategy,"n":n,"trials":trials,"seed":seed,
            "target":target,"probabilities":p.tolist(),"sensitivities":sensitivities,
            "a":a,"c_rival":correction,"total_derivative":total,
            "rho":a/total if total > 0 else None,
            "undefined_reason":None if total > 0 else "Estimated total derivative is not positive",
            "tie_policy":"strict_plurality" if strategy == "plurality" else "distinct_reward_order"}


def pivotal_variances(p, n):
    """Exact binary-logit example for the appendix's *single-index* RB comparison.

    Not a comparison with the fully summed leave-one-out estimator or GRPO.
    """
    if isinstance(p, bool) or not isinstance(p, (float,int)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must lie in [0,1]")
    if type(n) is not int or n < 2:
        raise ValueError("n must be an integer at least two")
    pivotal_probability = p*(1-p)**(n-1)
    z_amplitude = n*(1-p)
    cat_amplitude = n*(1-p)**n
    return {"p":p,"n":n,"mean":n*p*(1-p)**n,
            "sampled_index_variance":z_amplitude**2*pivotal_probability*(1-pivotal_probability),
            "conditioned_variance":cat_amplitude**2*p*(1-p)}
