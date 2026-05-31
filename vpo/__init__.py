"""VPO — Vector Policy Optimization.

Drop-in replacement for GRPO's advantage estimator that rewards diverse
answer sets via stochastic Dirichlet scalarization (arXiv:2605.22817).
"""

from vpo.advantage import grpo_advantages, vpo_advantages
from vpo.config import VPOConfig

__all__ = ["VPOConfig", "vpo_advantages", "grpo_advantages"]
