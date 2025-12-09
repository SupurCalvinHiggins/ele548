import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Iterable, List, Optional
from warnings import catch_warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from memory import Memory
from model import DQN
from running_scaler import RunningScaler
from torch import Tensor
from tqdm import tqdm

with catch_warnings():
    import compiler_gym
    import gym
    from compiler_gym.envs import CompilerEnv
    from compiler_gym.wrappers import (
        ConstrainedCommandline,
        RandomOrderBenchmarks,
        TimeLimit,
    )

device = torch.device("cuda")


@dataclass
class System:
    # Environment.
    envs: List[CompilerEnv]
    memory: Memory
    observation_scaler: RunningScaler
    # Network.
    policy_net: DQN
    target_net: DQN
    opt: optim.AdamW
    # Scheduling.
    step: int


@dataclass
class Config:
    # Epsilon-greedy parameters.
    eps_start: float = 0.9
    eps_end: float = 0.01
    eps_decay: int = 25_000
    # Model parameters.
    hidden_size: int = 256
    # Training parameters.
    episodes: int = 10_000
    envs: int = 32
    steps_per_episode: int = 250  # 25
    batch_size: int = 128
    gamma: float = 0.99
    # Optimizer parameters.
    lr: float = 3e-5
    # Memory parameters.
    memory_capacity: int = 10_000
    memory_min_capacity: int = 1_000
    # Gradient clipping.
    gradient_max_norm: float = 80.0
    # Soft updates.
    tau: float = 0.005
    # Plotting.
    episodes_per_plot: int = 100
    window_size: int = 100


# TODO: things to track
# reward, Q-value
# action distribution
# action had no effect (in info)
# Use geomean for sliding windows


@dataclass
class Statistics:
    # Average loss from each step.
    losses: List[float] = field(default_factory=list)
    # Improvement factor from each episode.
    improvement_factors: List[float] = field(default_factory=list)
    # -Oz improvement factor from each episode.
    oz_improvement_factors: List[float] = field(default_factory=list)
    # Epsilon-greedy threshold from each step.
    eps_thresholds: List[float] = field(default_factory=list)
    # Gradient norm from each step.
    gradient_norms: List[float] = field(default_factory=list)
    # Scaled states mean from each step.
    scaled_states_means: List[float] = field(default_factory=list)
    # Scaled states var from each step.
    scaled_states_vars: List[float] = field(default_factory=list)


@torch.no_grad()
def train_step_batch(sys: System, cfg: Config, stat: Statistics) -> None:
    if len(sys.memory) < cfg.memory_min_capacity:
        return

    states, actions, next_states, rewards, dones = sys.memory.sample_batch(
        cfg.batch_size
    )
    # print(states.shape, actions.shape, next_states.shape, rewards.shape, dones.shape)
    scaled_states = sys.observation_scaler.scale(states)
    scaled_next_states = sys.observation_scaler.scale(next_states)

    stat.scaled_states_means.append(scaled_states.mean().item())
    stat.scaled_states_vars.append(scaled_states.var().item())

    next_state_values = torch.zeros(cfg.batch_size, device=device)
    next_state_values[~dones] = sys.target_net(scaled_next_states[~dones]).max(1).values
    expected_state_action_values = (next_state_values * cfg.gamma) + rewards

    criterion = nn.SmoothL1Loss()
    sys.opt.zero_grad()

    with torch.enable_grad():
        state_action_values = sys.policy_net(scaled_states).gather(
            1, actions.unsqueeze(1)
        )
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss.backward()

    stat.losses.append(loss.item())

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        sys.policy_net.parameters(), cfg.gradient_max_norm
    )
    stat.gradient_norms.append(gradient_norm.item())
    sys.opt.step()

    for target_param, param in zip(
        sys.target_net.parameters(), sys.policy_net.parameters()
    ):
        target_param.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

    sys.step += 1


@torch.no_grad()
def select_action_batch(
    sys: System, cfg: Config, stat: Statistics, states: Tensor
) -> Tensor:
    eps_threshold = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(
        -1.0 * sys.step / cfg.eps_decay
    )
    stat.eps_thresholds.append(eps_threshold)

    scaled_states = sys.observation_scaler.scale(states)
    policy_actions = sys.policy_net(scaled_states).argmax(dim=-1).view(-1)

    random_actions = torch.tensor(
        [env.action_space.sample() for env in sys.envs], device=device, dtype=torch.long
    )

    mask = torch.rand(cfg.envs, device=device, dtype=torch.float) > eps_threshold

    actions = torch.zeros(cfg.envs, device=device, dtype=torch.long)
    actions[mask] = policy_actions[mask]
    actions[~mask] = random_actions[~mask]

    return actions


@torch.no_grad()
def train_episode_batch(sys: System, cfg: Config, stat: Statistics) -> None:
    states = torch.tensor(
        [env.reset() for env in sys.envs], dtype=torch.float, device=device
    )
    initial_costs = [env.observation["IrInstructionCount"] for env in sys.envs]
    oz_costs = [env.observation["IrInstructionCountOz"] for env in sys.envs]

    for _ in range(cfg.steps_per_episode):
        actions = select_action_batch(sys, cfg, stat, states)
        cpu_actions = actions.cpu()

        next_states, rewards, dones, _ = zip(
            *[env.step(action.item()) for env, action in zip(sys.envs, cpu_actions)]
        )

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)

        sys.memory.push_batch(states, actions, next_states, rewards, dones)

        # TODO: Does this make sense?
        for _ in range(cfg.envs):
            train_step_batch(sys, cfg, stat)

        states = next_states

    episode_costs = [env.observation["IrInstructionCount"] for env in sys.envs]

    stat.improvement_factors.extend(
        initial_cost / episode_cost
        for initial_cost, episode_cost in zip(initial_costs, episode_costs)
    )
    stat.oz_improvement_factors.extend(
        initial_cost / oz_cost for initial_cost, oz_cost in zip(initial_costs, oz_costs)
    )


def plot_with_moving_average(
    cfg: Config,
    data: Iterable,
    label: str,
    color: Optional[str] = None,
    ma_color: Optional[str] = None,
) -> None:
    y = torch.tensor(data)
    plt.plot(y.numpy(), label=label, color=color, zorder=0)
    if len(y) >= cfg.window_size:
        pad_value = y[: cfg.window_size].mean()
        ma = y.unfold(0, cfg.window_size, 1).mean(1).view(-1)
        ma = torch.cat((torch.full((cfg.window_size - 1,), pad_value), ma))
        plt.plot(ma.numpy(), label=f"{label} Moving Average", color=ma_color, zorder=1)


def plot_statistics(cfg: Config, stat: Statistics) -> None:
    plt.clf()
    plot_with_moving_average(
        cfg,
        stat.improvement_factors,
        label="DQN",
        color="lightblue",
        ma_color="darkblue",
    )
    plot_with_moving_average(
        cfg,
        stat.oz_improvement_factors,
        label="-Oz",
        color="orange",
        ma_color="red",
    )
    plt.xlabel("Episode")
    plt.ylabel("IR Instruction Count Improvement Factor over -O0")
    plt.legend()
    plt.savefig("improvement_factor.png")

    plt.clf()
    plt.plot(stat.scaled_states_means, label="Scaled Mean")
    plt.plot(stat.scaled_states_vars, label="Scaled Var")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig("scaled_states.png")

    plt.clf()
    plot_with_moving_average(
        cfg,
        stat.gradient_norms,
        label="Gradient Norm",
    )
    plt.xlabel("Step")
    plt.legend()
    plt.savefig("grad_norm.png")

    plt.clf()
    plot_with_moving_average(cfg, stat.losses, label="Loss")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig("loss.png")

    plt.clf()
    plt.plot(stat.eps_thresholds, label="Epsilon-Greedy Threshold")
    plt.xlabel("Step")
    plt.legend()
    plt.savefig("eps_threshold.png")


@torch.no_grad()
def train(sys: System, cfg: Config, stat: Statistics) -> None:
    plot_threshold = cfg.episodes_per_plot
    for episode in tqdm(range(0, cfg.episodes, cfg.envs)):
        train_episode_batch(sys, cfg, stat)

        if (episode + 1) > plot_threshold:
            plot_statistics(cfg, stat)
            plot_threshold = episode + cfg.episodes_per_plot


def make_env(env: CompilerEnv, cfg: Config) -> CompilerEnv:
    env = TimeLimit(env, max_episode_steps=cfg.steps_per_episode)
    # env.reset(benchmark="benchmark://cbench-v1/qsort")
    # env = RandomOrderBenchmarks(env, env.datasets["benchmark://chstone-v0"])
    # env = RandomOrderBenchmarks(env, env.datasets["benchmark://cbench-v1"])
    # env = RandomOrderBenchmarks(env, env.datasets["benchmark://mibench-v1"])
    env = RandomOrderBenchmarks(env, env.datasets["benchmark://tensorflow-v0"])
    # env = RandomOrderBenchmarks(env, env.datasets["benchmark://jotaibench-v0"])
    env.observation_space = "Autophase"
    env.reward_space = "IrInstructionCount"
    return env


def main(cfg: Config) -> None:
    with ExitStack() as stack:
        # This is a *very* hacky way to ensure we get different RNG in each env.
        env = gym.make("llvm-v0")
        envs = [env] + [env.fork() for _ in range(cfg.envs - 1)]
        envs = [stack.enter_context(make_env(env, cfg)) for env in envs]
        env = envs[0]

        n_observations = env.observation_space.shape[0]
        n_actions = env.action_space.n

        policy_net = DQN(n_observations, n_actions, cfg.hidden_size).to(device)
        target_net = DQN(n_observations, n_actions, cfg.hidden_size).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        opt = optim.AdamW(policy_net.parameters(), lr=cfg.lr, amsgrad=True)

        state_shape = env.observation_space.shape
        observation_scaler = RunningScaler(shape=state_shape).to(device)
        memory = Memory(capacity=cfg.memory_capacity, state_shape=state_shape).to(
            device
        )

        sys = System(
            envs=envs,
            policy_net=policy_net,
            target_net=target_net,
            opt=opt,
            observation_scaler=observation_scaler,
            memory=memory,
            step=0,
        )
        stat = Statistics()

        train(sys, cfg, stat)


if __name__ == "__main__":
    cfg = Config()
    main(cfg)
