"""Minimal VPO example — random vector rewards, no GPU needed.

Run: python examples/minimal.py

Shows that a diverse answer set earns a higher VPO advantage than a collapsed
one, even though both have the same per-answer reward magnitude.
"""

import torch

from vpo import VPOConfig, vpo_advantages
from vpo.metrics import reward_diversity

torch.manual_seed(0)

G, m, d = 4, 3, 3  # 4 rollouts, 3 answers each, 3 reward objectives

# Rollout 0: a diverse set spanning the three objectives (one-hot per answer).
# Rollouts 1-3: collapsed sets where every answer chases the same objective.
reward_vectors = torch.zeros(G, m, d)
for i in range(m):
    reward_vectors[0, i, i] = 1.0  # diverse
reward_vectors[1:, :, 0] = 1.0  # collapsed onto objective 0

adv = vpo_advantages(reward_vectors, VPOConfig())

for g in range(G):
    div = reward_diversity(reward_vectors[g])
    print(f"rollout {g}: diversity={div:.2f}  advantage={adv[g]:+.3f}")

print(f"\nadvantages sum to ~0: {adv.sum().item():.2e}")
print("the diverse rollout (0) gets the highest advantage.")
