from fractions import Fraction
from itertools import permutations
import math
import random

import pytest

from cat.evaluation.metrics import majority_at_k, majority_vote, pass_at_k, summarize_problem_scores


@pytest.mark.parametrize("n", range(1, 10))
def test_pass_matches_independent_combinatorics(n):
    for c in range(n + 1):
        for k in range(1, n + 1):
            expected = 1 - Fraction(math.comb(n - c, k), math.comb(n, k)) if k <= n - c else 1
            assert pass_at_k(n, c, k) == pytest.approx(float(expected), abs=1e-15)


def test_pass_tiny_probability_is_not_rounded_to_zero():
    assert pass_at_k(10**12, 1, 2) == pytest.approx(2e-12, rel=1e-10, abs=0)
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5, rel=0, abs=1e-15)
    assert pass_at_k(4, 1, 2) != 1 - (1 - .25) ** 2


@pytest.mark.parametrize("n,c,k", [
    (0, 0, 1), (-1, 0, 1), (4, -1, 1), (4, 5, 1), (4, 1, 0), (4, 1, 5),
    (True, 0, 1), (4, True, 1), (4, 1, True), (4.0, 1, 2), (4, .5, 2), (4, 1, 2.0),
])
def test_pass_validation(n, c, k):
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


@pytest.mark.parametrize("answers,expected", [
    (["2", "2", "3"], "2"), ([None, "2", "3"], "2"),
    ([None, None], None), (["wrong", "right"], "wrong"), (["a", "b", "b"], "b"),
])
def test_majority_first_occurrence(answers, expected):
    assert majority_vote(answers) == expected


def test_strict_tie_and_invalid_draws():
    assert majority_vote(["right", "wrong"], tie_break="failure") is None
    assert majority_vote(["right", "right", "wrong"], tie_break="failure") == "right"
    result = majority_at_k(["yes", None, None, None], 1, is_correct=lambda a: a == "yes")
    assert result.value == 0.25
    assert result.method == "exact_single_draw" and result.trials == 0
    assert result.monte_carlo_se == 0


@pytest.mark.parametrize("answers", [[], "abc", [""], [2], [False]])
def test_vote_invalid_input(answers):
    with pytest.raises(ValueError):
        majority_vote(answers)


@pytest.mark.parametrize("kwargs", [
    {"k": 0}, {"k": 5}, {"k": True}, {"trials": 1}, {"trials": True},
    {"seed": -1}, {"seed": True}, {"tie_break": "correct"},
])
def test_mc_validation(kwargs):
    options = {"k": 2, "is_correct": lambda a: a == "x", **kwargs}
    with pytest.raises(ValueError):
        majority_at_k(["x", "y", "x", None], **options)


@pytest.mark.parametrize("tie", ["first", "failure"])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_mc_matches_enumerated_ordered_subsets(k, tie):
    pool = ["right", "right", "wrong", None]
    samples = list(permutations(range(len(pool)), k))
    expected = sum(majority_vote([pool[i] for i in subset], tie_break=tie) == "right"
                   for subset in samples) / len(samples)
    result = majority_at_k(pool, k, is_correct=lambda a: a == "right", trials=10000, seed=7, tie_break=tie)
    assert result.value == pytest.approx(expected, abs=0.025)
    assert result.method == "mc_without_replacement"


def test_full_pool_first_tie_uses_sample_order_not_file_order():
    estimate = majority_at_k(["wrong", "right"], 2, is_correct=lambda a: a == "right", trials=4000)
    assert estimate.value == pytest.approx(0.5, abs=0.03)
    strict = majority_at_k(["wrong", "right"], 2, is_correct=lambda a: a == "right", tie_break="failure")
    assert strict.value == 0


def test_mc_reproducibility_and_global_random_state():
    before = random.getstate()
    args = (["x", "x", "y", None], 3)
    a = majority_at_k(*args, is_correct=lambda x: x == "x", seed=70)
    b = majority_at_k(*args, is_correct=lambda x: x == "x", seed=70)
    assert a == b
    assert random.getstate() == before
    assert a.monte_carlo_se == pytest.approx(math.sqrt(a.value * (1 - a.value) / (a.trials - 1)))


def test_correctness_does_not_merge_votes():
    # Two different correct strings lose against three copies of the same wrong
    # string. Pooling all correct strings using the reference answer would cheat.
    answers = ["1/2", "1/2", ".5", ".5", "wrong", "wrong", "wrong"]
    result = majority_at_k(answers, 7, is_correct=lambda a: a != "wrong")
    assert result.value == 0
    assert pass_at_k(7, 4, 7) == 1


def test_summary_weights_problems_and_has_sample_se():
    result = summarize_problem_scores([0.0, 0.5, 1.0])
    assert result["mean"] == 0.5
    assert result["standard_error"] == pytest.approx(0.5 / math.sqrt(3))
    assert result["num_problems"] == 3
    assert summarize_problem_scores([0.4])["standard_error"] is None


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-.1], [1.1], [True], ["1"], "1"])
def test_summary_invalid(values):
    with pytest.raises(ValueError):
        summarize_problem_scores(values)
