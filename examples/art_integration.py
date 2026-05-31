"""Plugging VPO into an OpenPipe/ART GRPO training loop.

This is NOT runnable ART code — it's the minimal before/after diff showing
where VPO swaps in. ART computes scalar advantages from a scalar reward per
rollout; VPO computes them from a *vector* reward per *answer set* instead.

ART repo: https://github.com/OpenPipe/ART


BEFORE — standard GRPO inside ART
---------------------------------

    # Each rollout produced one completion with one scalar reward.
    rewards = torch.tensor([traj.reward for traj in group])   # (G,)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    for traj, adv in zip(group, advantages):
        traj.advantage = adv.item()


AFTER — VPO
-----------

    from vpo import VPOConfig, vpo_advantages
    from vpo.prompt import format_multi_answer_prompt, split_answers

    # 1. At rollout time, ask for m answers in one chain:
    #    prompt = format_multi_answer_prompt(prompt, num_answers=3)
    #    then split_answers(completion) to recover the m answers.

    # 2. Score EACH answer with a VECTOR reward (d objectives), giving a
    #    (G, m, d) tensor instead of a (G,) scalar tensor:
    reward_vectors = torch.stack([
        torch.stack([reward_fn(ans) for ans in traj.answers])  # (m, d)
        for traj in group
    ])                                                          # (G, m, d)

    # 3. Swap the advantage line for VPO. Same uniform-per-token application.
    advantages = vpo_advantages(reward_vectors, VPOConfig(num_weight_samples=32))
    for traj, adv in zip(group, advantages):
        traj.advantage = adv.item()

Everything downstream (the PPO/GRPO loss, token masking, optimizer) is
unchanged — VPO only changes how the scalar advantage per rollout is computed.
"""
