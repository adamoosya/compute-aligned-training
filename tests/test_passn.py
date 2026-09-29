import math
from decimal import Decimal, localcontext

import pytest
import torch

from cat.objectives import (
    passn_log_probability, passn_probability, passn_rl_weight,
    passn_sft_loss, passn_sft_weight, passn_sft_weight_from_logp,
)


def reference(p, n):
    with localcontext() as ctx:
        ctx.prec = 100
        p = Decimal(str(p))
        q = 1 - p
        success = 1 - q ** n
        raw = n * q ** (n - 1)
        return float(success), float(raw), float(p * raw / success)


@pytest.mark.parametrize("n", [1, 2, 4, 16, 64, 128])
@pytest.mark.parametrize("p", [1e-20, 1e-8, 0.01, 0.3, 0.9, 1 - 1e-8])
def test_values_against_decimal(n, p):
    x = torch.tensor(p, dtype=torch.float64)
    expected_prob, expected_raw, expected_sft = reference(p, n)
    assert passn_probability(x, n).item() == pytest.approx(expected_prob, rel=2e-12, abs=1e-300)
    assert passn_rl_weight(x, n).item() == pytest.approx(expected_raw, rel=1e-6, abs=1e-300)
    assert passn_sft_weight(x, n).item() == pytest.approx(expected_sft, rel=1e-6, abs=1e-300)


@pytest.mark.parametrize("n", [1, 2, 8, 64, 128])
def test_loss_gradient_is_negative_sft_weight(n):
    lp = torch.tensor([-60.0, -12.0, -2.0, -0.5, -0.01], dtype=torch.float64, requires_grad=True)
    loss = passn_sft_loss(lp, n, reduction="sum")
    grad, = torch.autograd.grad(loss, lp)
    expected = -passn_sft_weight_from_logp(lp.detach(), n)
    torch.testing.assert_close(grad, expected, rtol=2e-10, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 8, 64])
def test_gradcheck(n):
    lp = torch.tensor([-4.0, -0.5], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: passn_sft_loss(x, n), (lp,))


@pytest.mark.parametrize("n", [1, 2, 64, 128])
def test_endpoints(n):
    p = torch.tensor([0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(passn_probability(p, n), p)
    expected_sft = p.new_tensor([1.0, 1.0 if n == 1 else 0.0])
    expected_raw = p.new_tensor([float(n), 1.0 if n == 1 else 0.0])
    torch.testing.assert_close(passn_sft_weight(p, n), expected_sft)
    torch.testing.assert_close(passn_rl_weight(p, n), expected_raw)
    lp = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    loss = passn_sft_loss(lp, n)
    loss.backward()
    assert loss.item() == 0.0
    assert lp.grad.item() == (-1.0 if n == 1 else 0.0)


@pytest.mark.parametrize("n", [1, 8, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extreme_log_probabilities_retain_loss_and_gradient(n, dtype):
    lp = torch.tensor([-1000.0, -10000.0], dtype=dtype, requires_grad=True)
    assert (lp.exp() == 0).all()
    loss = passn_sft_loss(lp, n, reduction="none")
    torch.testing.assert_close(loss, -lp - math.log(n))
    torch.testing.assert_close(passn_sft_weight_from_logp(lp, n), torch.ones_like(lp))
    loss.sum().backward()
    torch.testing.assert_close(lp.grad, -torch.ones_like(lp))


def test_reductions_and_shape():
    lp = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    losses = passn_sft_loss(lp, 8, reduction="none")
    assert losses.shape == lp.shape
    torch.testing.assert_close(passn_sft_loss(lp, 8), losses.mean())
    torch.testing.assert_close(passn_sft_loss(lp, 8, reduction="sum"), losses.sum())
    torch.testing.assert_close(passn_log_probability(lp, 1), lp)


def test_increasing_budget_increases_success_and_decreases_sft_weight():
    p = torch.linspace(0.001, 0.999, 101, dtype=torch.float64)
    for n in [1, 4, 16]:
        assert (passn_probability(p, n + 1) >= passn_probability(p, n)).all()
        assert (passn_sft_weight(p, n + 1) <= passn_sft_weight(p, n) + 1e-14).all()
