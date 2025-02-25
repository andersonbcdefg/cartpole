import argparse
import os
import fire
import random
import time
from distutils.util import strtobool
from types import SimpleNamespace
from gymnasium import make_vec
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from gymnasium.wrappers.rendering import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def seed_everything(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_hiddens,
        last_layer_init_scale: float = 1.
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            layer_init(nn.Linear(in_dim, hidden_dim)),
            *[
                layer_init(nn.Linear(hidden_dim, hidden_dim))
                for _ in range(num_hiddens)
            ],
            layer_init(nn.Linear(hidden_dim, out_dim), std=last_layer_init_scale)
        ])
        self.num_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = F.tanh(self.layers[i](x))
        return self.layers[-1](x)

class Agent(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        policy_num_hiddens = 2,
        policy_hidden_dim = 64,
        critic_num_hiddens = 2,
        critic_hidden_dim = 128
    ):
        super().__init__()
        self.policy = MLP(
            observation_dim,
            policy_hidden_dim,
            action_dim,
            policy_num_hiddens
        )
        self.critic = MLP(
            observation_dim,
            critic_hidden_dim,
            1,
            critic_num_hiddens,
            0.01
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.policy(x)
        value = self.critic(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

def get_scheduler(optimizer, total_steps: int):
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: (total_steps - step + 1.0e-8) / (total_steps + 1.0e-8)
    )

def monte_carlo_returns(
    rewards: torch.Tensor, # num_steps, num_envs
    dones: torch.Tensor, # num_steps, num_envs (is step t terminal)
    next_dones: torch.Tensor, # 1, num_envs
    next_values: torch.Tensor, # 1, num_envs
    gamma: float = 0.99
):
    # if we wanted to be fully accurate, we should separate terminated and truncated,
    # and treat them differently. but OAI's PPO doesn't even do this
    num_steps, num_envs = rewards.shape
    returns = torch.zeros_like(rewards)
    # this is what we want: dones_shifted[t] is like dones[t+1],
    # which tells us if NEXT state is terminal. if so, there's no more rewards,
    # so we don't bootstrap. but if we used dones[t], that would tell us if the
    # CURRENT state is terminal, aka. it's already the first state of a new trajectory,
    # so we shouldn't cut off the reward here.
    dones_shifted = torch.cat([dones, next_dones], dim=0)[1:]
    for st in reversed(range(num_steps)):
        if st == num_steps - 1:
            returns[st, :] = rewards[st, :] + gamma * next_values * (1 - dones_shifted[st, :]) # only bootstrap if not done
        else:
            returns[st, :] = rewards[st, :] + gamma * returns[st + 1, :] * (1 - dones_shifted[st, :])

    return returns

def n_step_returns(
    rewards: torch.Tensor, # num_steps, num_envs
    dones: torch.Tensor, # num_steps, num_envs
    next_dones: torch.Tensor, # 1, num_envs
    values: torch.Tensor, # num_steps, num_envs
    next_values: torch.Tensor, # 1, num_envs
    gamma: float = 0.99,
    n: int = 5,
    baseline: bool = True
):
    num_steps, num_envs = rewards.shape
    returns = torch.zeros_like(rewards)
    # a cleaner way to handle all these edge cases would be padding +
    # concatnating next_values to values...
    for st in reversed(range(num_steps)):
        returns[st, :] = rewards[st, :] # always includes current reward
        active = torch.ones_like(next_values).squeeze()
        for st2 in range(st + 1, st + n):
            # zero out locations where the prior state was terminal
            active *= (1 - dones[st2 - 1, :])
            if st2 >= num_steps: # guard
                # bootstrap with next_values
                returns[st, :] += gamma**(st2 - st) * next_values.squeeze() * active
                break
            returns[st, :] += gamma**(st2 - st) * rewards[st2, :] * active
        # executes if we didn't complete the loop, so still need to bootstrap
        else:
            # update active with the previous done state
            active *= (1 - dones[st + n - 1])
            if st + n < num_steps:
                returns[st, :] += gamma**n * values[st + n, :] * active
            else:
                returns[st, :] += gamma**n * next_values.squeeze() * active

    # if baseline, we need to subtract the value estimates for each state
    if baseline:
        returns -= values
    return returns

def n_step_returns_td(
    rewards: torch.Tensor, # num_steps, num_envs
    dones: torch.Tensor, # num_steps, num_envs
    values: torch.Tensor, # num_steps, num_envs
    next_values: torch.Tensor, # 1, num_envs
    gamma: float = 0.99,
    n: int = 5,
    baseline: bool = True
):
    num_steps, num_envs = rewards.shape
    # first calculate td errors: r_t + gamma * V_t+1 - V_t. the initial V_t is the one that doesn't
    # cancel in the telescoping sum; that means the baseline is "built in"!
    values_offset = torch.cat([values[1:, :], next_values.view(1, -1)])
    td_errors = rewards + gamma * values_offset * (1 - dones) - values
    padded_td_errors = torch.cat([td_errors, torch.zeros((n, num_envs))], dim=0)
    padded_dones = torch.cat([dones, torch.ones((n, num_envs))], dim=0)
    # now do the telescoping sum from back to front
    advantages = torch.zeros_like(td_errors)
    for st in reversed(range(num_steps)):
        # slice out the tds to sum
        td_slice = padded_td_errors[st:st + n, :]
        done_slice = padded_dones[st: st + n - 1, :]
        # calculate the mask of how much to weight each td error
        discount_mask = gamma**torch.arange(td_slice.shape[0]).view(-1, 1)
        # always include current reward
        dones_mask = torch.cat([
            torch.ones(num_envs).view(1, num_envs),
            torch.cumprod(1 - done_slice, dim=0)
        ], dim=0)
        mask = discount_mask * dones_mask

        advantages[st, :] += torch.sum(mask * td_slice, dim=0)

    if baseline:
        return advantages
    return advantages + values

# should be the same as above but more vectorization
def n_step_returns_td_vec(
    rewards: torch.Tensor, # num_steps, num_envs
    dones: torch.Tensor, # num_steps, num_envs
    values: torch.Tensor, # num_steps, num_envs
    next_values: torch.Tensor, # 1, num_envs
    gamma: float = 0.99,
    n: int = 5,
    baseline: bool = True
):
    num_steps, num_envs = rewards.shape
    # first calculate td errors: r_t + gamma * V_t+1 - V_t. the initial V_t is the one that doesn't
    # cancel in the telescoping sum; that means the baseline is "built in"!
    values_offset = torch.cat([values[1:, :], next_values.view(1, -1)])
    # (1 - dones) to ensure that terminal states don't include a "next value" in the "reality" term
    td_errors = rewards + gamma * values_offset * (1 - dones) - values
    # now do the telescoping sum from back to front
    if n == 1:
        dones_mask = torch.ones((num_steps, num_envs, 1))
    else:
        dones_padded = torch.cat([dones, torch.ones((n - 2, num_envs))]) # treat padded states as done
        dones_windowed = dones_padded.unfold(dimension=0, size=n - 1, step=1) # num_steps, num_envs, n - 1
        dones_mask = torch.cat([
            torch.ones((num_steps, num_envs, 1)),
            torch.cumprod(1 - dones_windowed, dim=2)
        ], dim=2)
    tds_padded = torch.cat([td_errors, torch.zeros((n - 1, num_envs))])
    tds_windowed = tds_padded.unfold(dimension=0, size=n, step=1) # num_steps, num_envs, n
    discounts = gamma**torch.arange(n).view(1, 1, n)
    advantages = (tds_windowed * dones_mask * discounts).sum(dim=-1)

    if baseline:
        return advantages
    return advantages + values

def test_n_step():
    torch.manual_seed(42)
    num_steps = 50
    num_envs = 4
    n = 5
    gamma = 0.99

    # Generate random test data.
    rewards = torch.randn(num_steps, num_envs)
    # Generate random dones as 0/1 floats.
    dones = torch.randint(0, 2, (num_steps, num_envs)).float()
    values = torch.randn(num_steps, num_envs)
    next_values = torch.randn(1, num_envs)

    # Calculate n-step returns using three different implementations.
    adv_loop = n_step_returns(rewards, dones, values, next_values, gamma, n, baseline=True)
    adv_td_loop = n_step_returns_td(rewards, dones, values, next_values, gamma, n, baseline=True)
    adv_vec = n_step_returns_td_vec(rewards, dones, values, next_values, gamma, n, baseline=True)

    # Use torch.allclose to compare the outputs.
    if not torch.allclose(adv_loop, adv_td_loop, atol=1e-5):
        print("Mismatch between loop and td implementations!")
    if not torch.allclose(adv_loop, adv_vec, atol=1e-5):
        print("Mismatch between loop and vectorized implementations!")
    if not torch.allclose(adv_td_loop, adv_vec, atol=1e-5):
        print("Mismatch between td and vectorized implementations!")

    print("All n-step return implementations match.")

def gae_returns(
    rewards: torch.Tensor, # num_steps, num_envs
    dones: torch.Tensor, # num_steps, num_envs
    values: torch.Tensor, # num_steps, num_envs
    next_values: torch.Tensor, # 1, num_envs
    gamma: float = 0.99,
    gae_lambda: float = 0.97
):
    num_steps, num_envs = rewards.shape
    # first calculate td errors: r_t + gamma * V_t+1 - V_t. the initial V_t is the one that doesn't
    # cancel in the telescoping sum; that means the baseline is "built in"!
    values_offset = torch.cat([values[1:, :], next_values.view(1, -1)])
    td_errors = rewards + gamma * values_offset * (1 - dones) - values

    # now do the telescoping sum from back to front
    advantages = td_errors.clone()
    for st in reversed(range(num_steps - 1)):
        advantages[st, :] += advantages[st + 1, :] * gae_lambda * gamma * (1 - dones[st, :])
    return advantages

def test_gae():
    torch.manual_seed(42)
    num_steps = 50
    num_envs = 4
    gamma = 0.99

    # Generate random test data
    rewards = torch.randn(num_steps, num_envs)
    dones = torch.randint(0, 2, (num_steps, num_envs)).float()
    values = torch.randn(num_steps, num_envs)
    next_values = torch.randn(1, num_envs)

    # GAE with lambda=1 should match n-step with n=num_steps
    gae = gae_returns(rewards, dones, values, next_values, gamma, gae_lambda=1.0)
    n_step = n_step_returns_td_vec(rewards, dones, values, next_values, gamma, n=num_steps, baseline=True)

    if not torch.allclose(gae, n_step, atol=1e-5):
        print("GAE with lambda=1 doesn't match n-step with full horizon!")
        print("Max difference:", torch.abs(gae - n_step).max().item())
    else:
        print("GAE with lambda=1 matches n-step with full horizon")

    # GAE with lambda=0 should match 1-step TD errors
    gae_0 = gae_returns(rewards, dones, values, next_values, gamma, gae_lambda=0.0)
    td_1step = n_step_returns_td_vec(rewards, dones, values, next_values, gamma, n=1, baseline=True)

    if not torch.allclose(gae_0, td_1step, atol=1e-5):
        print("GAE with lambda=0 doesn't match 1-step TD!")
        print("Max difference:", torch.abs(gae_0 - td_1step).max().item())
    else:
        print("GAE with lambda=0 matches 1-step TD")

def estimate_advantages(
    values: torch.Tensor
):
    # naive implementation with nested loops.
    num_steps, num_envs = values.shape
    for t in range(num_steps):
        for e in range(num_envs):
            pass

def main():
    args = parse_args()
    seed_everything(args)
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
    )
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_vec(
        "CartPole-v1",
        num_envs=args.num_envs,
    )
    env = RecordEpisodeStatistics(env)

    agent = Agent(
        np.array(env.single_observation_space.shape).prod(),
        env.single_action_space.n # pyright: ignore
    ).to(device)
    print(agent)

    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1.0e-5
    )

    global_step = 0
    start_time = time.time()
    next_obs, _ = env.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # linear decay
    scheduler = get_scheduler(optimizer, total_steps=num_updates)

    # initialize storage
    obs_batch = torch.zeros((args.num_steps, args.num_envs) + env.single_observation_space.shape).to(device) # pyright: ignore
    actions = torch.zeros((args.num_steps, args.num_envs) + env.single_action_space.shape).to(device) # pyright: ignore
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # outer loop
    for _ in range(num_updates):

        # collect episodes
        for step in range(args.num_steps):
            obs_batch[step, :, :] = torch.from_numpy(next_obs)
            dones[step, :] = next_done
            obs = torch.from_numpy(next_obs)
            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(obs)
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            next_done = (torch.from_numpy(terminated).bool() | torch.from_numpy(truncated).bool()).float()
            actions[step, ...] = action
            logprobs[step, :] = logprob
            rewards[step, :] = torch.from_numpy(reward)
            values[step, :] = value.squeeze()

            if "episode" in info.keys():
                for i, x in enumerate(info['episode']['r']):
                    if next_done[i]:
                        print("episode return", x)

        # print mean return
        print("mean reward:", rewards.mean().item())

        # estimate advantages
        with torch.no_grad():
            # get next value for bootstrap
            next_value = agent.get_value(torch.from_numpy(next_obs)).reshape(1, -1)
            # YOU DONT UNDERSTAND THIS. FIGURE IT OUT DUMDUM
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # learn. flatten and randomize over steps not just trajectories
        b_obs = obs_batch.reshape((-1,) + env.single_observation_space.shape) # pyright: ignore
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.single_action_space.shape) # pyright: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # randomize and iterate minibatches
        for ep in range(args.update_epochs):
            perm = torch.randperm(b_obs.shape[0])
            for batch_idxs in perm.chunk(args.num_minibatches, dim=0):
                obs_chunk = b_obs[batch_idxs]
                actions_chunk = b_actions[batch_idxs]
                old_logprobs_chunk = b_logprobs[batch_idxs]
                advantages_chunk = b_advantages[batch_idxs]
                returns_chunk = b_returns[batch_idxs]
                _, newlogprob, entropy, value = agent.get_action_and_value(obs_chunk)

                pg_ratio = (newlogprob - old_logprobs_chunk).exp()
                pg_term = pg_ratio * advantages_chunk
                clipped_pg_term = torch.clamp(pg_ratio, 1 - args.clip_coef, 1 + args.clip_coef) * advantages_chunk
                objective = torch.minimum(pg_term, clipped_pg_term)
                policy_loss = -objective.mean()
                value_loss = 0.5 * F.mse_loss(value.flatten(), returns_chunk)
                entropy_bonus = entropy.mean()
                loss = policy_loss - args.ent_coef * entropy_bonus + value_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
               # print("loss", loss)

if __name__ == "__main__":
    fire.Fire()
