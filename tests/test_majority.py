import math
from decimal import Decimal, localcontext

import pytest
import torch

from cat.objectives import (
    majority_log_probability, majority_probability, majority_rl_weight,
    majority_sft_loss, majority_sft_weight, majority_sft_weight_from_logp,
    passn_log_probability, passn_rl_weight, passn_sft_loss,
)


def reference(p_value, n, k):
    with localcontext() as ctx:
        ctx.prec = 150
        p = Decimal(str(p_value))
        q = 1 - p
        tail = sum(Decimal(math.comb(n, i)) * p ** i * q ** (n - i) for i in range(k, n + 1))
        raw = Decimal(n * math.comb(n - 1, k - 1)) * p ** (k - 1) * q ** (n - k)
        return float(tail), float(raw), float(p * raw / tail), float(-tail.ln())


@pytest.mark.parametrize("n,k", [(1, 1), (4, 1), (4, 2), (8, 5), (16, 4), (64, 26), (128, 65), (16, 16)])
@pytest.mark.parametrize("p", [1e-12, 0.01, 0.2, 0.5, 0.8, 1 - 1e-6])
def test_values_against_decimal(n, k, p):
    x = torch.tensor(p, dtype=torch.float64)
    success, raw, sft, nll = reference(p, n, k)
    assert majority_probability(x, n, k).item() == pytest.approx(success, rel=3e-10, abs=1e-300)
    assert majority_rl_weight(x, n, k).item() == pytest.approx(raw, rel=1e-7, abs=1e-300)
    assert majority_sft_weight(x, n, k).item() == pytest.approx(sft, rel=1e-7, abs=1e-300)
    assert majority_sft_loss(x.log(), n, k).item() == pytest.approx(nll, rel=2e-8, abs=1e-140)


@pytest.mark.parametrize("n,k", [(1, 1), (4, 1), (8, 2), (8, 5), (16, 8), (64, 26), (128, 65), (8, 8)])
def test_loss_gradient_matches_sft_weight(n, k):
    lp = torch.tensor([-80.0, -10.0, -3.0, -1.0, -0.5, -0.1], dtype=torch.float64, requires_grad=True)
    grad, = torch.autograd.grad(majority_sft_loss(lp, n, k, reduction="sum"), lp)
    expected = -majority_sft_weight_from_logp(lp.detach(), n, k)
    torch.testing.assert_close(grad, expected, rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("n,k", [(4, 2), (8, 5), (16, 4), (64, 26)])
def test_log_loss_gradcheck(n, k):
    lp = torch.tensor([-2.0, -0.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: majority_sft_loss(x, n, k), (lp,))


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_raw_weight_matches_independently_differentiated_polynomial(n):
    for k in range(1, n + 1):
        p = torch.tensor([0.03, 0.2, 0.6, 0.97], dtype=torch.float64, requires_grad=True)
        tail = sum(math.comb(n, i) * p.pow(i) * (1 - p).pow(n - i) for i in range(k, n + 1))
        derivative, = torch.autograd.grad(tail.sum(), p)
        torch.testing.assert_close(majority_rl_weight(p.detach(), n, k), derivative, rtol=1e-9, atol=1e-13)


def test_rl_coefficient_uses_n_minus_one_choose_k_minus_one():
    # Regression for n*C(n,k), which is larger by n/k and is not Table 1.
    assert majority_rl_weight(torch.tensor(0.5, dtype=torch.float64), 8, 3).item() == pytest.approx(1.3125)


@pytest.mark.parametrize("n,k", [(1, 1), (8, 1), (8, 5), (8, 8)])
def test_endpoints(n, k):
    p = torch.tensor([0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(majority_probability(p, n, k), p)
    torch.testing.assert_close(majority_sft_weight(p, n, k), p.new_tensor([k, n if k == n else 0]))
    torch.testing.assert_close(majority_rl_weight(p, n, k), p.new_tensor([n if k == 1 else 0, n if k == n else 0]))
    lp = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    loss = majority_sft_loss(lp, n, k)
    loss.backward()
    assert loss.item() == 0
    assert lp.grad.item() == (-n if k == n else 0)


@pytest.mark.parametrize("n,k", [(8, 1), (8, 5), (64, 26), (128, 65), (16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_very_small_p_does_not_get_clipped(n, k, dtype):
    lp = torch.tensor([-1000.0, -10000.0], dtype=dtype, requires_grad=True)
    loss = majority_sft_loss(lp, n, k, reduction="none")
    log_coeff = math.log(math.comb(n, k))
    torch.testing.assert_close(loss, -k * lp - log_coeff)
    torch.testing.assert_close(majority_sft_weight_from_logp(lp, n, k), torch.full_like(lp, float(k)), rtol=2e-5, atol=2e-5)
    loss.sum().backward()
    torch.testing.assert_close(lp.grad, torch.full_like(lp, -float(k)))


def test_k_one_reduces_to_passn():
    lp = torch.tensor([-1000.0, -30.0, -1.0, -0.01, 0.0], dtype=torch.float64)
    for n in [1, 4, 64]:
        torch.testing.assert_close(majority_log_probability(lp, n, 1), passn_log_probability(lp, n))
        torch.testing.assert_close(majority_sft_loss(lp, n, 1), passn_sft_loss(lp, n))
        torch.testing.assert_close(majority_rl_weight(lp.exp(), n, 1), passn_rl_weight(lp.exp(), n))


def test_large_budget_log_tail_is_finite():
    lp = torch.tensor([-1000.0, -3.0, -0.7, -0.001], dtype=torch.float64, requires_grad=True)
    loss = majority_sft_loss(lp, 1024, 513, reduction="sum")
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(lp.grad).all()


def test_shape_reductions_and_probability_range():
    p = torch.tensor([[0.05, 0.3], [0.7, 0.999]], dtype=torch.float64)
    logp = majority_log_probability(p.log(), 16, 5)
    assert logp.shape == p.shape
    assert (logp <= 0).all()
    loss = majority_sft_loss(p.log(), 16, 5, reduction="none")
    torch.testing.assert_close(majority_sft_loss(p.log(), 16, 5), loss.mean())
    torch.testing.assert_close(majority_sft_loss(p.log(), 16, 5, reduction="sum"), loss.sum())


def test_tail_branch_agreement_near_pivot():
    for n, k in [(8, 5), (64, 26), (128, 65)]:
        pivot = k / (n + 1)
        values = torch.tensor([pivot - 1e-10, pivot, pivot + 1e-10], dtype=torch.float64)
        actual = majority_probability(values, n, k)
        target = values.new_tensor([reference(float(p), n, k)[0] for p in values])
        torch.testing.assert_close(actual, target, atol=1e-12, rtol=1e-11)
