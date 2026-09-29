"""Expected maximum of a saved empirical reward pool (with replacement)."""
from __future__ import annotations
import math
from typing import Sequence


def expected_max_reward(rewards: Sequence[float], n: int) -> float:
    """Exact order-statistic expectation; works with negative rewards and ties.

    This removes Monte Carlo resampling noise from the legacy protein evaluator.
    It does not recover the population reward distribution or its standard error.
    """
    if type(n) is not int or n < 1 or not rewards:
        raise ValueError("Positive budget and nonempty reward pool required")
    values = sorted(float(r) for r in rewards)
    if any(not math.isfinite(r) for r in values):
        raise ValueError("Nonfinite reward")
    m = len(values)
    # E[max] = top value minus gaps times the CDF below each gap.
    return values[-1] - math.fsum((values[i + 1] - values[i]) * ((i + 1) / m) ** n
                                  for i in range(m - 1))


def summarize_protein_rewards(pools: Sequence[Sequence[float]], budgets: Sequence[int]) -> dict:
    if not pools or not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("Nonempty reward pools and unique budgets required")
    result = {}
    for n in budgets:
        per_prompt = [expected_max_reward(pool, n) for pool in pools]
        mean = math.fsum(per_prompt) / len(per_prompt)
        se = (math.sqrt(math.fsum((v - mean) ** 2 for v in per_prompt) /
                       (len(per_prompt) * (len(per_prompt) - 1))) if len(per_prompt) > 1 else None)
        result[str(n)] = {"mean": mean, "prompt_standard_error": se, "per_prompt": per_prompt}
    return {"metric": "empirical_expected_max_with_replacement", "prompts": len(pools), "budgets": result}
