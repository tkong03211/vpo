"""VPOConfig: the handful of knobs the VPO advantage estimator needs."""

from dataclasses import dataclass


@dataclass
class VPOConfig:
    """Configuration for the VPO advantage estimator.

    Holds the few hyperparameters from the paper; nothing else lives here.
    """

    num_weight_samples: int = 32  # K — Dirichlet weight samples (Section 3.2)
    normalize_advantages: bool = True  # divide advantages by group std
    eps: float = 1e-8  # numerical floor for the normalization divide
