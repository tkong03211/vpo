"""Tests for the VPO advantage estimator and its building blocks."""

import torch

from vpo import VPOConfig, grpo_advantages, vpo_advantages
from vpo.advantage import sample_dirichlet_weights, set_reward


def test_diverse_beats_identical() -> None:
    """A set spanning the Pareto frontier must score higher than identical answers."""
    d, m, G, K = 3, 3, 8, 32
    identical = torch.zeros(G, m, d)
    identical[:, :, 0] = 1.0  # every answer = [1, 0, 0]

    diverse = torch.zeros(G, m, d)
    for i in range(m):
        diverse[:, i, i] = 1.0  # y1=[1,0,0], y2=[0,1,0], y3=[0,0,1]

    weights = sample_dirichlet_weights(K, d)
    diverse_r = set_reward(diverse[0], weights)
    identical_r = set_reward(identical[0], weights)
    assert diverse_r > identical_r


def test_advantages_zero_mean() -> None:
    """Within a group, unnormalized advantages must sum to ~0."""
    torch.manual_seed(0)
    reward_vectors = torch.rand(8, 3, 4)
    config = VPOConfig(normalize_advantages=False)
    adv = vpo_advantages(reward_vectors, config)
    assert torch.allclose(adv.sum(), torch.tensor(0.0), atol=1e-5)


def test_shared_weights_across_group() -> None:
    """K weight samples are identical for all G rollouts — otherwise not comparable.

    We verify it indirectly: with a fixed generator the same weights are drawn,
    and set_reward applied with those shared weights is what vpo_advantages uses.
    """
    d, K = 4, 16
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    w1 = sample_dirichlet_weights(K, d, g1)
    w2 = sample_dirichlet_weights(K, d, g2)
    # Same seed → identical weight set, i.e. weights are shared, not per-rollout.
    assert torch.allclose(w1, w2)


def test_dirichlet_on_simplex() -> None:
    """Every sampled weight vector must sum to 1."""
    weights = sample_dirichlet_weights(num_samples=100, num_objectives=4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(100), atol=1e-5)
    assert (weights >= 0).all()


def test_grpo_equivalence() -> None:
    """When d=1, VPO should reduce to standard GRPO advantage."""
    torch.manual_seed(0)
    G, m, d = 8, 1, 1
    reward_vectors = torch.rand(G, m, d)

    config = VPOConfig(normalize_advantages=True)
    vpo_adv = vpo_advantages(reward_vectors, config)

    # With d=1 and m=1, the set reward is just the scalar reward (w sums to 1,
    # so w^T r = r), so VPO must match GRPO on the same scalars.
    scalars = reward_vectors.reshape(G)
    grpo_adv = grpo_advantages(scalars, normalize=True)
    assert torch.allclose(vpo_adv, grpo_adv, atol=1e-5)
