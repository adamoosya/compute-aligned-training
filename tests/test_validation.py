import pytest
import torch

from cat.objectives import (
    best_of_n_rl_weight, majority_log_probability, majority_probability,
    majority_rl_weight, majority_sft_loss, majority_sft_weight,
    passn_log_probability, passn_probability, passn_rl_weight,
    passn_sft_loss, passn_sft_weight,
)

PROB_FUNCTIONS = [
    lambda p: passn_probability(p, 8), lambda p: passn_sft_weight(p, 8),
    lambda p: passn_rl_weight(p, 8), lambda p: majority_probability(p, 8, 5),
    lambda p: majority_sft_weight(p, 8, 5), lambda p: majority_rl_weight(p, 8, 5),
    lambda p: best_of_n_rl_weight(p, 8),
]


@pytest.mark.parametrize("fn", PROB_FUNCTIONS)
def test_invalid_probabilities_rejected(fn):
    for p in [-0.1, 1.1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            fn(torch.tensor(p))
    with pytest.raises(TypeError):
        fn(0.5)
    with pytest.raises(TypeError):
        fn(torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        fn(torch.tensor([]))


@pytest.mark.parametrize("fn", [passn_log_probability, passn_sft_loss, lambda x, n: majority_log_probability(x, n, 2), lambda x, n: majority_sft_loss(x, n, 2)])
def test_log_api_rejects_nll_and_nonfinite_values(fn):
    for x in [0.1, float("inf"), float("-inf"), float("nan")]:
        with pytest.raises(ValueError):
            fn(torch.tensor(x), 8)


@pytest.mark.parametrize("n", [0, -1, True, 2.5, "8"])
def test_invalid_budgets(n):
    for fn in [passn_probability, passn_rl_weight, best_of_n_rl_weight]:
        with pytest.raises((ValueError, TypeError)):
            fn(torch.tensor(0.4), n)


@pytest.mark.parametrize("k", [0, -1, 9, True, 2.5])
def test_invalid_thresholds(k):
    with pytest.raises((ValueError, TypeError)):
        majority_sft_weight(torch.tensor(0.4), 8, k)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtypes_and_no_input_mutation(dtype):
    p = torch.tensor([[0.1, 0.3], [0.7, 0.9]], dtype=dtype)
    original = p.clone()
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    for fn in PROB_FUNCTIONS:
        result = fn(p)
        assert result.dtype == expected_dtype
        assert result.shape == p.shape
        assert result.device == p.device
        assert torch.isfinite(result).all()
        torch.testing.assert_close(p, original)


def test_invalid_reduction():
    with pytest.raises(ValueError, match="reduction"):
        passn_sft_loss(torch.tensor(-1.0), 8, reduction="tokens")
    with pytest.raises(ValueError, match="reduction"):
        majority_sft_loss(torch.tensor(-1.0), 8, 5, reduction="tokens")
