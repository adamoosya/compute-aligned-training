import pytest
import torch

from cat.objectives import (
    best_of_n_rl_weight, normalize_prompt_weights, normalize_sample_weights,
    passn_rl_weight, weighted_policy_loss,
)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 64])
def test_best_of_n_rank_weight(n):
    q = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
    torch.testing.assert_close(best_of_n_rl_weight(q, n), n * q ** (n - 1))
    torch.testing.assert_close(best_of_n_rl_weight(1 - q, n), passn_rl_weight(q, n))


def test_best_of_n_interior_gradients():
    q = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: best_of_n_rl_weight(x, 8), (q,))


def test_prompt_normalization_preserves_ratios_and_is_detached():
    w = torch.tensor([0.1, 0.4, 2.5], dtype=torch.float64, requires_grad=True)
    normalized = normalize_prompt_weights(w)
    torch.testing.assert_close(normalized.mean(), w.new_tensor(1.0))
    torch.testing.assert_close(normalized / normalized[0], w.detach() / w.detach()[0])
    assert not normalized.requires_grad


def test_one_prompt_is_not_normalized_to_one():
    with pytest.raises(ValueError, match="at least two distinct prompts"):
        normalize_prompt_weights(torch.tensor([3.2]))
    with pytest.raises(ValueError, match="1-D"):
        normalize_prompt_weights(torch.tensor([[3.2, 3.2, 3.2]]))


def test_expand_after_normalization_preserves_between_prompt_difference():
    weights = normalize_prompt_weights(torch.tensor([0.25, 1.75]))
    expanded = weights[:, None].expand(2, 4)
    torch.testing.assert_close(expanded[1] / expanded[0], torch.full((4,), 7.0))
    assert not torch.allclose(expanded, torch.ones_like(expanded))


def test_sample_normalization_and_zeros():
    weights = torch.tensor([[0.0, 2.0], [4.0, 10.0]], dtype=torch.float64)
    torch.testing.assert_close(normalize_sample_weights(weights), weights / weights.mean())
    for fn in [normalize_prompt_weights, normalize_sample_weights]:
        with pytest.raises(ValueError, match="all-zero"):
            fn(torch.zeros(4))


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_normalization_avoids_overflow_and_epsilon_bias(scale):
    w = torch.tensor([scale, scale * 2, scale * 4], dtype=torch.float64)
    torch.testing.assert_close(normalize_prompt_weights(w), w.new_tensor([3/7, 6/7, 12/7]))


def test_weighted_policy_loss_detaches_weights_and_advantages():
    lp = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64, requires_grad=True)
    weights = lp.exp() * 5  # Deliberately connected to the same policy graph.
    weights.retain_grad()
    loss = weighted_policy_loss(lp, advantages, weights)
    loss.backward()
    torch.testing.assert_close(lp.grad, -advantages.detach() * weights.detach() / 3)
    assert advantages.grad is None
    assert weights.grad is None


def test_policy_reductions():
    lp = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    a = torch.tensor([[1.0, -1.0], [0.5, 0.0]])
    w = torch.tensor([[0.5, 0.5], [1.5, 1.5]])
    loss = weighted_policy_loss(lp, a, w, reduction="none")
    torch.testing.assert_close(loss, -lp * a * w)
    torch.testing.assert_close(weighted_policy_loss(lp, a, w), loss.mean())
    torch.testing.assert_close(weighted_policy_loss(lp, a, w, reduction="sum"), loss.sum())


def test_policy_loss_rejects_accidental_broadcasting():
    with pytest.raises(ValueError, match="same shape"):
        weighted_policy_loss(-torch.ones(2, 4), torch.ones(2, 4), torch.ones(2))


def test_raw_prompt_weight_changes_update_without_normalization():
    lp = torch.tensor([-1.0, -2.0], requires_grad=True)
    advantage = torch.tensor([1.0, -1.0])
    loss = weighted_policy_loss(lp, advantage, torch.full((2,), 3.0))
    loss.backward()
    torch.testing.assert_close(lp.grad, torch.tensor([-1.5, 1.5]))
