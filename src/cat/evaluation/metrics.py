"""Metrics on an observed candidate pool, independent of model/training code."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import random
import statistics


def require_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be a Python integer >= {minimum}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that a size-k subset of n candidates contains a success.

    Equals 1 - C(n-c,k)/C(n,k). For independent generated candidates, this is
    the paper's unbiased Pass@k estimator, not 1 - (1-c/n)**k.
    """
    require_integer(n, "n")
    require_integer(c, "c", 0)
    require_integer(k, "k")
    if c > n or k > n:
        raise ValueError("c and k cannot exceed the pool size n")
    if k == 1:
        return c / n
    if c == 0:
        return 0.0
    if k > n - c:
        return 1.0
    # Sum log failure factors and use expm1 to preserve small success values.
    log_failure = math.fsum(math.log1p(-c / (n - i)) for i in range(k))
    return -math.expm1(log_failure)


def _validate_answers(answers: Sequence[str | None]) -> tuple[str | None, ...]:
    if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence) or not answers:
        raise ValueError("answers must be a nonempty sequence of strings or None")
    if any(a is not None and (not isinstance(a, str) or not a) for a in answers):
        raise ValueError("use None for an invalid answer, not an empty/non-string value")
    return tuple(answers)


def _winner(answers: Sequence[str | None], tie_break: str) -> str | None:
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None
    # Counter retains first occurrence; this matches most_common(1) in the sources.
    highest = max(counts.values())
    tied = [answer for answer, count in counts.items() if count == highest]
    return None if tie_break == "failure" and len(tied) > 1 else tied[0]


def majority_vote(answers: Sequence[str | None], *, tie_break: str = "first") -> str | None:
    """Select the most common valid answer; correctness never affects selection.

    `first` resolves ties by first occurrence in the supplied draw order;
    `failure` rejects a tied highest count. Invalid draws remain in the sampling
    pool but cannot win. An all-invalid group fails.
    """
    answers = _validate_answers(answers)
    if tie_break not in {"first", "failure"}:
        raise ValueError("tie_break must be 'first' or 'failure'")
    return _winner(answers, tie_break)


@dataclass(frozen=True)
class MajorityEstimate:
    value: float
    method: str
    trials: int
    monte_carlo_se: float


def majority_at_k(
    answers: Sequence[str | None], k: int, *, is_correct: Callable[[str], bool],
    trials: int = 500, seed: int = 42, tie_break: str = "first",
) -> MajorityEstimate:
    """Average plurality success from independently drawn, ordered subsets.

    Each draw samples k distinct candidate indices. Different trials may reuse
    indices. First-occurrence tie breaking therefore acts on the randomized draw
    order, not the original file order. At k=1 the pool mean is calculated exactly.
    """
    answers = _validate_answers(answers)
    require_integer(k, "k")
    require_integer(trials, "trials", 2)
    require_integer(seed, "seed", 0)
    if k > len(answers):
        raise ValueError("k exceeds the candidate pool size")
    if tie_break not in {"first", "failure"}:
        raise ValueError("tie_break must be 'first' or 'failure'")
    if not callable(is_correct):
        raise TypeError("is_correct must be callable")
    # This lookup only scores a selected winner. It does not merge answer classes.
    correct = {a: bool(is_correct(a)) for a in dict.fromkeys(answers) if a is not None}
    if k == 1:
        value = sum(correct.get(a, False) for a in answers) / len(answers)
        return MajorityEstimate(value, "exact_single_draw", 0, 0.0)
    rng = random.Random(seed)
    wins = 0
    indices = range(len(answers))
    for _ in range(trials):
        chosen = rng.sample(indices, k)
        winner = _winner([answers[i] for i in chosen], tie_break)
        wins += correct.get(winner, False)
    value = wins / trials
    # Sample variance of Bernoulli trials / number of trials. Conditional on pool.
    mc_se = math.sqrt(value * (1.0 - value) / (trials - 1))
    return MajorityEstimate(value, "mc_without_replacement", trials, mc_se)


def summarize_problem_scores(values: Sequence[float]) -> dict[str, float | int | None]:
    """Equal weight per problem; no binomial or synthetic correlation assumption."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError("values must be a nonempty sequence")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 <= v <= 1 for v in values):
        raise ValueError("problem scores must be finite values in [0,1]")
    count = len(values)
    return {
        "mean": math.fsum(values) / count,
        "standard_error": statistics.stdev(values) / math.sqrt(count) if count > 1 else None,
        "num_problems": count,
    }
