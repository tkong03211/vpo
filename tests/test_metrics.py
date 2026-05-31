"""Tests for reward diversity and best@k metrics."""

import torch

from vpo.metrics import best_at_k, reward_diversity


def test_diversity_identical_is_zero() -> None:
    """Identical answers have zero pairwise distance."""
    reward_vectors = torch.zeros(3, 4)
    reward_vectors[:, 0] = 1.0
    assert reward_diversity(reward_vectors) == 0.0


def test_diversity_orthogonal_positive() -> None:
    """Orthogonal one-hot answers are maximally spread."""
    reward_vectors = torch.eye(3)  # [1,0,0], [0,1,0], [0,0,1]
    # Each pair differs by L1 distance 2; mean over the 3 pairs is 2.
    assert abs(reward_diversity(reward_vectors) - 2.0) < 1e-6


def test_diversity_single_answer() -> None:
    """Fewer than two answers means no pairs, diversity 0."""
    assert reward_diversity(torch.ones(1, 4)) == 0.0


def test_best_at_k_picks_max() -> None:
    """best@k returns the best scalarized reward among the first k answers."""
    reward_vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    weights = torch.tensor([1.0, 0.0])  # only the first objective matters
    # First answer scores 1.0 under this weighting; nothing in first 2 beats it.
    assert best_at_k(reward_vectors, weights, k=2) == 1.0


def test_best_at_k_grows_with_k() -> None:
    """Considering more answers can only help (monotone in k)."""
    reward_vectors = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    weights = torch.tensor([1.0, 0.0])
    b1 = best_at_k(reward_vectors, weights, k=1)
    b2 = best_at_k(reward_vectors, weights, k=2)
    assert b2 >= b1
