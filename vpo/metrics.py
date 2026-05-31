"""Evaluation metrics: reward diversity and best@k.

These are the diversity and search-quality numbers reported in the paper's
tables (arXiv:2605.22817). Both operate on vector rewards.
"""

import torch


def reward_diversity(reward_vectors: torch.Tensor) -> float:
    """Average pairwise L1 distance between answer reward vectors.

    The diversity metric from all the paper's tables: high means the answers
    spread across reward space rather than collapsing onto one point.

    reward_vectors: (m, d) — m answers' reward vectors
    returns: mean L1 distance over all answer pairs (0.0 if fewer than 2)
    """
    m = reward_vectors.shape[0]
    if m < 2:
        return 0.0
    # Pairwise L1 distances; cdist with p=1 gives the full (m, m) matrix.
    dists = torch.cdist(reward_vectors, reward_vectors, p=1)
    # Average over the upper triangle (each unordered pair counted once).
    iu = torch.triu_indices(m, m, offset=1)
    return dists[iu[0], iu[1]].mean().item()


def best_at_k(
    reward_vectors: torch.Tensor, scalar_weights: torch.Tensor, k: int
) -> float:
    """Best scalarized reward among the first k answers: max_{y<k} w^T r(x, y).

    The central eval metric from the paper — how good is the best candidate a
    user would find by drawing k samples and scoring them under preference w.

    reward_vectors: (m, d) — m answers' reward vectors
    scalar_weights: (d,) — the preference weighting w
    k: how many answers to consider (clamped to m)
    returns: the best scalarized reward over the first k answers
    """
    k = min(k, reward_vectors.shape[0])
    # w^T r(x, y) for the first k answers, then take the max.
    scalarized = reward_vectors[:k] @ scalar_weights
    return scalarized.max().item()
