import time
import signal
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
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

    for loop in trange(num_episodes // num_envs, desc="Collecting episodes..."):
        # States come as (num_envs, 4) array
        observation, _ = env.reset()
        done = False
        states: list[Tensor] = [torch.from_numpy(observation)]
        actions: list[Tensor] = []
        action_logprobs: list[Tensor] = []
        rewards: list[Tensor] = []
        termination_idxs: list[int | None] = [None] * num_envs

        step = 0
        while True:
            _, logits = policy(
                torch.from_numpy(observation)
            )
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprobs: Tensor = dist.log_prob(action)

            observation, reward, terminated, truncated, info = env.step(action.numpy())
            action_logprobs.append(logprobs) # already a tensor
            actions.append(action) # already a tensor
            rewards.append(torch.from_numpy(reward))
            states.append(torch.from_numpy(observation))

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


def apply_policy(
    policy,
    padded_states,
    trajectory_lengths
):
    mask = torch.arange(padded_states.shape[1]).view(1, -1) >= torch.tensor(trajectory_lengths).view(-1, 1)
    values, logits = policy(padded_states)
    logprobs = torch.log_softmax(logits, dim=-1)
    values = values.squeeze(-1)
    values.masked_fill_(mask, 0)

    return logprobs, values

if __name__ == "__main__":
    NUM_ENVS = 32
    NUM_SAMPLES = 1024
    NUM_OUTER_STEPS = 10_000
    NUM_INNER_STEPS = 5
    CLIP_EPSILON = 0.2
    VALUE_LOSS_COEFF = 0.5
    ENTROPY_LOSS_COEFF = 0.01
    MINIBATCH_SIZE = 64
    env = gym.make_vec('CartPole-v1', num_envs=NUM_ENVS)
    policy = PolicyNetwork(4, 2, 1)
    optimizer = torch.optim.AdamW(
        policy.parameters()
    )
    all_returns = []
    all_value_losses = []

    signal.signal(signal.SIGINT, lambda sig, frame: save_graph(all_returns, all_value_losses) or exit(0))

    for i in range(NUM_OUTER_STEPS):
        (
            padded_states, # (jagged) num_samples * traj_len * 4
            padded_actions, # (jagged) num_samples * traj_len
            padded_logprobs, # (jagged) num_samples * traj_len * 2
            padded_rewards, # (jagged) num_samples * traj_len
            trajectory_lengths
        ) = collect_episodes(env, policy, NUM_ENVS, NUM_SAMPLES)
        padded_returns = compute_returns(padded_rewards)
        mask = torch.arange(padded_returns.shape[1]).view(1, -1) < torch.tensor(trajectory_lengths).view(-1, 1)

        # compute advantages once from the episodes
        with torch.no_grad():
            _, padded_values = apply_policy(policy, padded_states, trajectory_lengths)
            fixed_advantages = compute_advantage(padded_rewards, padded_values)
            # NEW: normalize advantages
            fixed_advantages = (fixed_advantages - fixed_advantages[mask].mean()) / (fixed_advantages[mask].std() + 1e-8)
        avg_return = np.mean(trajectory_lengths)
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
                new_logprobs, new_values = apply_policy(
                    policy,
                    states_chunk,
                    lengths_chunk
                )
                mini_mask = torch.arange(states_chunk.shape[1]).view(1, -1) < \
                           torch.tensor(lengths_chunk).view(-1, 1)

                action_new_logprobs: Float32[Tensor, "batch max_len"] = \
                    new_logprobs.gather(2, actions_chunk.unsqueeze(-1).long()).squeeze(2)
                ratio: Float32[Tensor, "batch max_len"] = (action_new_logprobs - old_logprobs_chunk).exp()
                pg_term = ratio * advantages_chunk
                clipped_term = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages_chunk
                objective = torch.minimum(pg_term, clipped_term)

                policy_loss = -1 * objective[mini_mask].sum() / mini_mask.sum()
                value_loss = F.mse_loss(new_values, returns_chunk, reduction="none")[mini_mask].mean()
                entropy_loss = (new_logprobs.exp() * new_logprobs).sum(dim=-1)[mini_mask].mean()

                # -- backward on loss and step
                combined_loss = policy_loss + VALUE_LOSS_COEFF * value_loss + ENTROPY_LOSS_COEFF * entropy_loss
                combined_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # -- end ppo update
                total_value_loss += (value_loss * MINIBATCH_SIZE / NUM_SAMPLES / NUM_INNER_STEPS).item()

        all_value_losses.append(total_value_loss)
        print("outer step", i, "returns", avg_return, "val loss", total_value_loss)

    save_graph(all_returns, all_value_losses)
