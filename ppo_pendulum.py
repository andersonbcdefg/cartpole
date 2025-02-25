import time
import signal
import numpy as np
import sys
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Normal
from jaxtyping import Float32
from util import save_graph
from tqdm.auto import trange

class PolicyNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_hiddens = 2,
        hidden_dim = 128
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.hiddens = nn.Sequential(*[
            layer for _ in range(num_hiddens)
            for layer in (
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
        ])
        self.classification_head = nn.Linear(hidden_dim, out_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Float32[Tensor, "batch in_dim"]):
        h1: Float32[Tensor, "batch hidden_dim"] = self.linear1(x)
        h2: Float32[Tensor, "batch hidden_dim"] = self.hiddens(F.relu(h1))
        logits: Float32[Tensor, "batch out_dim"] = self.classification_head(h2)
        value: Float32[Tensor, "batch 1"] = self.value_head(h2)
        return value, logits

def pad_tensor_list(
    tensors: list[Tensor], # len(tensors) = batch_size
):
    # pads along dim 1
    max_len = max([t.shape[0] for t in tensors])
    new_shape = (len(tensors), max_len, *tensors[0].shape[1:])
    padded = torch.zeros(new_shape, dtype=tensors[0].dtype)
    for idx, t in enumerate(tensors):
        padded[idx, :t.shape[0]] = t

    return padded

@torch.no_grad()
def collect_episodes(
    env,
    policy: PolicyNetwork,
    num_envs = 8,
    num_episodes = 2048
) -> tuple:
    assert num_episodes % num_envs == 0, "num_episodes not divisible by num_envs"
    all_states: list[Tensor] = []
    all_actions: list[Tensor] = []
    all_action_logprobs: list[Tensor] = []
    all_rewards: list[Tensor] = []
    all_trajectory_lengths = []

    for loop in range(num_episodes // num_envs): # , desc="Collecting episodes..."):
        # States come as (num_envs, 4) array
        observation, _ = env.reset()
        done = False
        states: list[Tensor] = [torch.from_numpy(observation).float()]
        actions: list[Tensor] = []
        action_logprobs: list[Tensor] = []
        rewards: list[Tensor] = []
        termination_idxs: list[int | None] = [None] * num_envs

        step = 0
        while True:
            _, logits = policy(
                torch.from_numpy(observation).float()
            )
            # for pendulum, sample from normal distribution
            # dist_mean = logits[:, 0]
            dist = convert_logits_to_dist(logits)
            action = dist.sample()
            clamped_action = action.clamp(min=-2.0, max=2.0)
            logprobs: Tensor = dist.log_prob(action)

            observation, reward, terminated, truncated, info = env.step(clamped_action.numpy().reshape(-1, 1))
            action_logprobs.append(logprobs) # already a tensor
            actions.append(action) # already a tensor
            rewards.append(torch.from_numpy(reward).float())
            states.append(torch.from_numpy(observation).float())

            for i in range(num_envs):
                if (terminated[i] or truncated[i]) and termination_idxs[i] is None:
                    termination_idxs[i] = step

            step += 1
            if all([x is not None for x in termination_idxs]):
                # accumulate to the batched thingies
                for i, idx in enumerate(termination_idxs):
                    assert idx is not None, "this shouldn't happen"
                    s_single = torch.stack(states)[:idx + 1, i, :]
                    a_single = torch.stack(actions)[:idx + 1, i]
                    lp_single = torch.stack(action_logprobs)[:idx + 1, i]
                    r_single = torch.stack(rewards)[:idx + 1, i]

                    all_states.append(s_single)
                    all_actions.append(a_single)
                    all_action_logprobs.append(lp_single)
                    all_rewards.append(r_single)
                    all_trajectory_lengths.append(idx + 1)
                break
    # handle masking
    padded_states = pad_tensor_list(all_states)
    padded_actions = pad_tensor_list(all_actions)
    padded_logprobs = pad_tensor_list(all_action_logprobs)
    padded_rewards = pad_tensor_list(all_rewards)

    return (
        padded_states, # (jagged) num_samples * traj_len * 4
        padded_actions, # (jagged) num_samples * traj_len
        padded_logprobs, # (jagged) num_samples * traj_len * 2
        padded_rewards, # (jagged) num_samples * traj_len
        all_trajectory_lengths
    )

def compute_returns(
    padded_rewards,
    gamma=0.99
):
    padded_returns = padded_rewards.clone()
    for idx in range(padded_rewards.shape[1] - 2, -1, -1):
        padded_returns[:, idx] += gamma * padded_returns[:, idx + 1]
    return padded_returns

@torch.no_grad()
def compute_advantage(
    padded_rewards,
    padded_values, # should be 0 everywhere that's not real
    gamma =  0.99,
    gae_lambda = 0.97
):
    # NEW: estimate advantages with GAE
    bsz = padded_values.shape[0]
    next_values = torch.cat([
        padded_values[:, 1:], torch.zeros((bsz, 1))
    ], dim=1)
    td_errors = padded_rewards + next_values * gamma - padded_values
    advantages = td_errors
    for j in range(advantages.shape[1] - 2, -1, -1):
        advantages[:, j] += gae_lambda * gamma * advantages[:, j + 1]

    return advantages

def convert_logits_to_dist(
    logits
):
    loc = 2.0 * torch.tanh(logits[..., 0])
    return Normal(loc, 0.5)

def apply_policy(
    policy,
    padded_states,
    trajectory_lengths
):
    mask = torch.arange(padded_states.shape[1]).view(1, -1) >= torch.tensor(trajectory_lengths).view(-1, 1)
    values, logits = policy(padded_states)
    dist = convert_logits_to_dist(logits)
    action = dist.sample()
    logprobs: Tensor = dist.log_prob(action)
    values = values.squeeze(-1).masked_fill(mask, 0)
    return logprobs, values

def get_action_logprobs_and_values(
    policy,
    padded_actions,
    padded_states,
    trajectory_lengths
):
    # like apply_policy, but the actions are fixed
    mask = torch.arange(padded_states.shape[1]).view(1, -1) >= torch.tensor(trajectory_lengths).view(-1, 1)
    values, logits = policy(padded_states)
    dist = convert_logits_to_dist(logits)
    logprobs: Tensor = dist.log_prob(padded_actions)
    values = values.squeeze(-1)
    values.masked_fill_(mask, 0)

    return logprobs, values, dist.scale

if __name__ == "__main__":
    NUM_ENVS = 32
    NUM_SAMPLES = 256
    NUM_OUTER_STEPS = 10_000
    NUM_INNER_STEPS = 5
    CLIP_EPSILON = 0.2
    VALUE_LOSS_COEFF = 0.1
    ENTROPY_LOSS_COEFF = 1.0
    MINIBATCH_SIZE = 32
    env = gym.make_vec('Pendulum-v1', num_envs=NUM_ENVS)
    policy = PolicyNetwork(in_dim=3, out_dim=1)  # For Pendulum
    optimizer = torch.optim.AdamW(
        policy.parameters()
    )
    all_returns = []
    all_value_losses = []
    all_means = []
    all_stds = []

    signal.signal(signal.SIGINT, lambda sig, frame: save_graph(all_returns, all_value_losses, all_means, all_stds) or exit(0))

    for i in range(NUM_OUTER_STEPS):
        (
            padded_states, # (jagged) num_samples * traj_len * 4
            padded_actions, # (jagged) num_samples * traj_len
            padded_logprobs, # (jagged) num_samples * traj_len * 2
            padded_rewards, # (jagged) num_samples * traj_len
            trajectory_lengths
        ) = collect_episodes(env, policy, NUM_ENVS, NUM_SAMPLES)
        # After collect_episodes, add:
        with torch.no_grad():
            initial_states = padded_states[:, 0]  # Look at first state of each episode
            _, logits = policy(initial_states)
            dist = convert_logits_to_dist(logits)
            all_means.append(dist.loc.mean().item())
            all_stds.append(dist.scale.mean().item())
            # print("initial mean", dist.loc.mean().item())
            # print("initial std", dist.scale.mean().item())

        padded_returns = compute_returns(padded_rewards)

        episode_returns = padded_returns[:, 0]
        avg_return = episode_returns.mean().item()
        mask = torch.arange(padded_returns.shape[1]).view(1, -1) < torch.tensor(trajectory_lengths).view(-1, 1)

        # compute advantages once from the episodes
        with torch.no_grad():
            _, padded_values = apply_policy(policy, padded_states, trajectory_lengths)
            fixed_advantages = compute_advantage(padded_rewards, padded_values)
            # print("fixed mean", fixed_advantages.mean())
            # print("fixed std", fixed_advantages.std())
            # sys.exit(0)
            # NEW: normalize advantages
            fixed_advantages = (fixed_advantages - fixed_advantages[mask].mean()) / (fixed_advantages[mask].std() + 1e-8)
        # avg_return = np.mean(trajectory_lengths)
        all_returns.append(avg_return)

        # do the PPO updates
        total_value_loss = 0
        for j in trange(NUM_INNER_STEPS, desc="PPO updates..."):
            # -- minibatched ppo updates
            perm = torch.randperm(NUM_SAMPLES)
            for batch_idxs in perm.split(MINIBATCH_SIZE, dim=0):
                states_chunk = padded_states[batch_idxs]
                lengths_chunk = [trajectory_lengths[idx] for idx in batch_idxs]
                actions_chunk = padded_actions[batch_idxs]
                old_logprobs_chunk = padded_logprobs[batch_idxs]
                advantages_chunk = fixed_advantages[batch_idxs]
                returns_chunk = padded_returns[batch_idxs]
                new_logprobs, new_values, dist_std = get_action_logprobs_and_values(
                    policy,
                    actions_chunk,
                    states_chunk,
                    lengths_chunk
                )

                mini_mask = torch.arange(states_chunk.shape[1]).view(1, -1) < \
                           torch.tensor(lengths_chunk).view(-1, 1)


                # print("about to compute ratio", new_logprobs.shape, "-",  old_logprobs_chunk.shape)

                ratio: Float32[Tensor, "batch max_len"] = (new_logprobs - old_logprobs_chunk).exp()

                pg_term = ratio * advantages_chunk
                clipped_term = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages_chunk
                objective = torch.minimum(pg_term, clipped_term)

                print(f"Ratios min/max: {ratio.min().item():.3f}/{ratio.max().item():.3f}")
                print(
                    f"Clipped fraction: {((ratio > 1 + CLIP_EPSILON) | (ratio < 1 - CLIP_EPSILON)).float().mean().item():.3f}")

                policy_loss = -1 * objective[mini_mask].sum() / mini_mask.sum().float()
                value_loss = F.mse_loss(new_values, returns_chunk, reduction="none")[mini_mask].mean().float()
                # entropy_loss = (new_logprobs.exp() * new_logprobs).sum(dim=-1)[mini_mask].mean()
                entropy_loss = 0.5 * torch.log(2 * torch.pi * torch.e * dist_std.pow(2))[mini_mask].mean()
                # -- backward on loss and step
                # print("policy_loss", policy_loss.dtype, "value_loss", value_loss.dtype, "entropy", entropy_loss.dtype)
                combined_loss = policy_loss + VALUE_LOSS_COEFF * value_loss + ENTROPY_LOSS_COEFF * entropy_loss
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()

                # -- end ppo update
                total_value_loss += (value_loss * MINIBATCH_SIZE / NUM_SAMPLES / NUM_INNER_STEPS).item()

        all_value_losses.append(total_value_loss)
        print("outer step", i, "returns", avg_return, "val loss", total_value_loss)

    save_graph(all_returns, all_value_losses, all_means, all_stds)
