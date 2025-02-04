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

class PolicyNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim = 128
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.classification_head = nn.Linear(hidden_dim, out_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Float32[Tensor, "batch in_dim"]):
        h1: Float32[Tensor, "batch hidden_dim"] = self.linear1(x)
        h2: Float32[Tensor, "batch hidden_dim"] = self.linear2(F.relu(h1))
        h3: Float32[Tensor, "batch hidden_dim"] = self.linear3(F.relu(h2))
        logits: Float32[Tensor, "batch out_dim"] = self.classification_head(F.relu(h3))
        value: Float32[Tensor, "batch 1"] = self.value_head(h3)
        return value, logits

def collect_episode(
    env,
    policy: PolicyNetwork,
    num_envs = 8
) -> tuple:
    # States come as (num_envs, 4) array
    observation, _ = env.reset()
    done = False

    # We probably want to collect:
    states = [observation]
    actions = []
    rewards = []
    termination_idxs: list[int | None] = [None] * num_envs

    step = 0
    while True:
        _, logits = policy(
            torch.from_numpy(observation)
        )
        probs = logits.softmax(dim=-1)
        action = Categorical(probs=probs).sample().numpy()
        actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        states.append(observation)

        for i in range(num_envs):
            if (terminated[i] or truncated[i]) and termination_idxs[i] is None:
                termination_idxs[i] = step

        step += 1
        if all([x is not None for x in termination_idxs]):
            break

    return (
        states, # trajectory x num_envs x 4 x
        actions, # num_envs x trajectory
        rewards, # num_envs x trajectory
        termination_idxs # num_envs
    )

def compute_returns(
    rewards,
    gamma=0.99
):
    result = np.array(rewards)
    for i in range(len(rewards)):
        idx = len(rewards) - 1 - i
        if i > 0:
            result[idx] += gamma * result[idx + 1]
    return result

def apply_gradient_gae(
    policy,
    states,
    actions,
    rewards,
    termination_idxs,
    gamma = 0.99,
    gae_lambda = 0.97,
    entropy_coef = 0.01
):
    all_returns = []
    all_value_losses = []
    total_steps = sum(termination_idxs)
    # print(len(states), "states")
    total_value_loss = torch.tensor(0.0)
    total_policy_loss = torch.tensor(0.0)
    total_entropy = torch.tensor(0.0)
    for i, idx in enumerate(termination_idxs):
        s_single = np.array(states)[:idx + 1, i, :]
        a_single = np.array(actions)[:idx + 1, i]
        r_single = np.array(rewards)[:idx + 1, i]
        # print(s_single.shape, a_single.shape, r_single.shape)
        returns = compute_returns(r_single)
        all_returns.append(returns[0])
        in_tensor = torch.from_numpy(s_single)
        values, batch_logits = policy(in_tensor)
        values = values.squeeze(1)
        probs = F.softmax(batch_logits, dim=-1)
        log_probs = F.log_softmax(batch_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        total_entropy += entropy[:-1].sum()
        action_idxs = torch.tensor(a_single).long().unsqueeze(1)  # shape (T, 1)
        taken_logprobs = log_probs.gather(1, action_idxs).squeeze(1)


        # NEW: estimate advantages with GAE
        next_values = torch.cat([values[1:], torch.tensor([0.0])])
        td_errors = torch.from_numpy(r_single) + next_values * gamma - values
        advantages = td_errors
        for j in range(len(advantages) - 2, -1, -1):
            advantages[j] += advantages[j + 1] * gae_lambda * gamma

        scaled = taken_logprobs * advantages
        total_policy_loss -= scaled.sum()
        total_value_loss += F.mse_loss(values, torch.from_numpy(returns), reduction="sum")

    avg_policy_loss = total_policy_loss / total_steps
    avg_value_loss = total_value_loss / total_steps
    entropy_loss = -total_entropy / total_steps * entropy_coef
    (avg_policy_loss + avg_value_loss + entropy_loss).backward()

    return np.mean(all_returns), avg_value_loss.item()

def apply_gradient(
    policy,
    states,
    actions,
    rewards,
    termination_idxs
):
    all_returns = []
    all_value_losses = []
    total_steps = sum(termination_idxs)
    # print(len(states), "states")
    total_value_loss = torch.tensor(0.0)
    total_policy_loss = torch.tensor(0.0)
    for i, idx in enumerate(termination_idxs):
        s_single = np.array(states)[:idx + 1, i, :]
        a_single = np.array(actions)[:idx + 1, i]
        r_single = np.array(rewards)[:idx + 1, i]
        # print(s_single.shape, a_single.shape, r_single.shape)
        returns = compute_returns(r_single)
        all_returns.append(returns[0])
        in_tensor = torch.from_numpy(s_single)
        values, batch_logits = policy(in_tensor)
        values = values.squeeze(1)
        batch_logprobs = torch.log_softmax(batch_logits, dim=-1)
        action_idxs = torch.tensor(a_single).long().unsqueeze(1)  # shape (T, 1)
        taken_logprobs = batch_logprobs.gather(1, action_idxs).squeeze(1)

        # NEW: estimate advantages
        advantages = torch.from_numpy(returns) - values
        scaled = taken_logprobs * advantages
        total_policy_loss -= scaled.sum()
        total_value_loss += F.mse_loss(values, torch.from_numpy(returns), reduction="sum")

    avg_policy_loss = total_policy_loss / total_steps
    avg_value_loss = total_value_loss / total_steps
    (avg_policy_loss + avg_value_loss).backward()

    return np.mean(all_returns), avg_value_loss.item()

if __name__ == "__main__":
    env = gym.make_vec('CartPole-v1', num_envs=16)
    policy = PolicyNetwork(4, 2)
    optimizer = torch.optim.AdamW(policy.parameters())
    returns = []
    value_losses = []

    signal.signal(signal.SIGINT, lambda sig, frame: save_graph(returns, value_losses) or exit(0))

    for i in range(100_000):
        states, actions, rewards, termination_idxs = collect_episode(env, policy)
        avg_return, value_loss = apply_gradient_gae(policy, states, actions, rewards, termination_idxs)
        optimizer.step()
        optimizer.zero_grad()
        avg_return = np.mean(termination_idxs)
        print("return:", avg_return, "value loss:", value_loss)
        returns.append(avg_return)
        value_losses.append(value_loss)

    save_graph(returns, value_losses)
