# VPO — Vector Policy Optimization

A minimal, readable implementation of **Vector Policy Optimization**
([arXiv:2605.22817](https://arxiv.org/abs/2605.22817)), a drop-in replacement
for GRPO's advantage estimator.

GRPO collapses reward into a single scalar, so the policy narrows onto one
response style: ten samples of the same prompt give ten near-identical answers.
That kills inference-time search (pass@k, best@k, AlphaEvolve), which needs
genuinely diverse candidates. VPO instead generates a *set* of answers and
scores the set by how well it covers the Pareto frontier under randomly sampled
reward weightings — rewarding diversity directly.

## Install

```bash
pip install vpo            # once published
# or, from a clone:
git clone https://github.com/tkong03211/vpo && cd vpo && pip install -e .
```

## Minimal example

```python
import torch
from vpo import VPOConfig, vpo_advantages

# (G, m, d): G rollouts, m answers per chain, d reward objectives
reward_vectors = torch.rand(8, 3, 4)
advantages = vpo_advantages(reward_vectors, VPOConfig(num_weight_samples=32))
# advantages: (G,), applied uniformly to every token in each rollout
```

A diverse answer set earns a higher advantage than a collapsed one — see
[`examples/minimal.py`](../examples/minimal.py).

## ART integration

Swapping VPO into an OpenPipe/ART GRPO loop is a one-line change to the
advantage computation (plus scoring each answer with a *vector* reward):

```python
# before — scalar GRPO
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)   # (G,)

# after — VPO
from vpo import VPOConfig, vpo_advantages
advantages = vpo_advantages(reward_vectors, VPOConfig())            # (G,)
```

Full before/after diff: [`examples/art_integration.py`](../examples/art_integration.py).

## The math

The set-level reward (Equation 1) is

$$R(S) = \mathbb{E}_{w \sim \mathrm{Dir}(\mathbf{1})}\left[\max_{y \in S} w^\top r(x, y)\right]$$

In plain English: sample a random weighting `w` over the `d` reward objectives.
For that weighting, pick the best answer in the set `S`. Average over many
weightings. A diverse set wins under *many* weightings, so it scores high; a
collapsed set only wins under a narrow region of weightings, so it scores low.

We estimate the expectation with `K` Monte Carlo weight samples drawn from
`Dir(1)` (uniform over the simplex) and **shared across the group** so all
rollouts are comparable:

$$\hat{R}(S^{(g)}) = \frac{1}{K}\sum_{k=1}^{K} \max_{y \in S^{(g)}} {w^{(k)}}^\top r(x, y)$$

Advantages are then group-relative, exactly as in GRPO:

$$A^{(g)} = \hat{R}(S^{(g)}) - \mathrm{mean}_g\, \hat{R}(S^{(g)})$$

optionally divided by the group standard deviation.

## Citation

```bibtex
@misc{bahlousboldi2026vpo,
  title         = {Vector Policy Optimization},
  author        = {Bahlous-Boldi and others},
  year          = {2026},
  eprint        = {2605.22817},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

Paper: <https://arxiv.org/abs/2605.22817>
