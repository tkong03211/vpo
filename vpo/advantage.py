"""Core VPO math — Dirichlet sampling, set reward, and the advantage estimator.

All of the VPO mathematics lives in this one file, top to bottom:
  1. sample_dirichlet_weights — draw weight vectors on the simplex
  2. set_reward               — Equation 1, the set-level reward R(S)
  3. vpo_advantages          — group-relative advantages from set rewards
  4. grpo_advantages         — the scalar GRPO baseline, for comparison

Paper: Vector Policy Optimization (arXiv:2605.22817), Section 3.
"""

import torch

from vpo.config import VPOConfig


def sample_dirichlet_weights(
    num_samples: int, num_objectives: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw weight vectors uniformly from the simplex via the Gamma trick.

    Dir(1) is uniform over the simplex; we need it to randomly weight rewards.

    num_samples: K — how many weight vectors to draw
    num_objectives: d — reward dimensionality
    returns: (K, d), each row non-negative and summing to 1
    """
    # Dir(1) sample = normalized vector of Gamma(1, 1) draws.
    # g_i ~ Gamma(1, 1); shape param 1, rate 1 (Gamma(1,1) == Exponential(1)).
    gammas = torch._standard_gamma(
        torch.ones(num_samples, num_objectives), generator=generator
    )
    # w_i = g_i / sum_j g_j  → lands on the simplex.
    return gammas / gammas.sum(dim=-1, keepdim=True)


def set_reward(
    reward_vectors: torch.Tensor, weight_samples: torch.Tensor
) -> torch.Tensor:
    """Monte Carlo estimate of E_{w~Dir(1)}[max_{y in S} w^T r(x, y)].

    Equation 1 of the VPO paper: a set scores high when some answer wins under
    many weightings — i.e. when the set covers the Pareto frontier.

    reward_vectors: (m, d) — the m answers' reward vectors for one rollout
    weight_samples: (K, d) — shared across all rollouts in the group
    returns: scalar set reward R_hat(S)
    """
    # For each of K weightings, score every answer: w^T r(x, y). Shape (K, m).
    scalarized = weight_samples @ reward_vectors.T

    # max_{y in S} w^T r(x, y) — best answer under each weighting. Shape (K,).
    best_per_weight = scalarized.max(dim=-1).values

    # R_hat(S) = (1/K) * sum_k max_{y} w_k^T r(x, y).
    return best_per_weight.mean()


def vpo_advantages(
    reward_vectors: torch.Tensor,
    config: VPOConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Group-relative advantages from VPO set rewards.

    Same structure as GRPO, but each rollout's scalar reward is the set reward
    R_hat(S) instead of a single scalar — so diversity is rewarded.

    reward_vectors: (G, m, d) — G rollouts, m answers each, d reward dims
    returns: (G,) advantages, applied uniformly to every token in each rollout
    """
    if config is None:
        config = VPOConfig()

    G, m, d = reward_vectors.shape

    # K Dirichlet weights, SHARED across the group so rollouts are comparable
    # (Section 3.2). Same w^(1..K) scores every rollout's set.
    weights = sample_dirichlet_weights(config.num_weight_samples, d, generator)

    # Per-rollout set reward R_hat(S^(g)). Shape (G,).
    set_rewards = torch.stack(
        [set_reward(reward_vectors[g], weights) for g in range(G)]
    )

    # A^(g) = R_hat(S^(g)) - mean_g R_hat(S^(g)).
    advantages = set_rewards - set_rewards.mean()

    # Optional GRPO-style normalization by the group std.
    if config.normalize_advantages:
        advantages = advantages / (set_rewards.std() + config.eps)

    return advantages


def grpo_advantages(
    rewards: torch.Tensor, normalize: bool = True, eps: float = 1e-8
) -> torch.Tensor:
    """Standard GRPO scalar advantage — the baseline VPO replaces.

    Group-relative: subtract the group mean, optionally divide by group std.

    rewards: (G,) — one scalar reward per rollout
    returns: (G,) advantages
    """
    # A^(g) = r^(g) - mean_g r^(g).
    advantages = rewards - rewards.mean()
    if normalize:
        advantages = advantages / (rewards.std() + eps)
    return advantages
