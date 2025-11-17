import math
import random
from dataclasses import dataclass, field
from typing import List, Optional
from warnings import catch_warnings

import matplotlib.pyplot as plt
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
class TrainConfig:
    ma_window_size: int = 100


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
    eps_decay: int = 10_000
    # Model parameters.
    hidden_size: int = 256
    # Training parameters.
    episodes: int = 10_000
    envs: int = 32
    steps_per_episode: int = 25
    batch_size: int = 128
    gamma: float = 0.99
    # Optimizer parameters.
    lr: float = 3e-5
    # Memory parameters.
    memory_capacity: int = 10_000
    memory_min_capacity: int = 1_000
    # Gradient clipping.
    gradient_max_value: float = 100.0
    # Soft updates.
    tau: float = 0.005
    # Plotting.
    episodes_per_plot: int = 100


# TODO: things to track
# loss, reward, grad, Q-value, input dist. after norm statistics
# action distribution
# epsilon greedy value
# action had no effect (in info)
# make a new Statistics structure, rename existing ones


@dataclass
class TrainState:
    finals: List[float] = field(default_factory=list)
    oz_finals: List[float] = field(default_factory=list)
    rewards: List[List[float]] = field(default_factory=list)
    losses: List[List[float]] = field(default_factory=list)


@dataclass
class Statistics:
    losses: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)


def train_episode(ts: TrainState, cfg: TrainConfig) -> None:
    ts.env.reset()
    ts.rewards.append([])
    start = float(ts.env.observation["IrInstructionCount"])
    ts.oz_finals.append(start / float(ts.env.observation["IrInstructionCountOz"]))

    while True:
        state = torch.tensor(
            ts.env.observation["Autophase"],
            dtype=torch.float,
            device=device,
        ).unsqueeze(0)

        action = select_action(ts, cfg)

        observation, reward, done, info = ts.env.step(action.item())

        ts.rewards[-1].append(reward)
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(
            observation, dtype=torch.float, device=device
        ).unsqueeze(0)
        ts.memory.emplace(state, action, next_state, reward)

        if done:
            ts.finals.append(start / float(ts.env.observation["IrInstructionCount"]))
            return

        train_step(ts, cfg)


@torch.no_grad()
def train_step_batch(sys: System, cfg: Config, stat: Statistics) -> None:
    if len(sys.memory) < cfg.memory_min_capacity:
        return

    states, actions, next_states, rewards, dones = sys.memory.sample(cfg.batch_size)
    scaled_states = sys.observation_scaler.scale(states)
    scaled_next_states = sys.observation_scaler.scale(next_states)

    next_state_values = torch.zeros(cfg.batch_size, device=device)
    next_state_values[~dones] = sys.target_net(scaled_next_states[~dones]).max(1).values
    expected_state_action_values = (next_state_values * cfg.gamma) + rewards

    criterion = nn.SmoothL1Loss()
    sys.opt.zero_grad()

    with torch.enable_grad():
        state_action_values = sys.policy_net(scaled_states).gather(1, actions)
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss.backward()

    torch.nn.utils.clip_grad_value_(sys.policy_net.parameters(), cfg.gradient_max_value)
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

    scaled_states = sys.observation_scaler.scale(states)
    policy_actions = sys.policy_net(scaled_states).argmax(dim=-1).view(-1, 1)

    random_actions = torch.tensor(
        [env.action_space.sample() for env in sys.envs], device=device, dtype=torch.long
    )

    mask = torch.rand(len(sys.envs), device=device, dtype=torch.float) > eps_threshold

    actions = torch.tensor(len(sys.envs), device=device, dtype=torch.long)
    actions[mask] = policy_actions
    actions[~mask] = random_actions

    return actions


@torch.no_grad()
def train_episode_batch(sys: System, cfg: Config, stat: Statistics) -> None:
    states = torch.tensor(
        [env.reset() for env in sys.envs], dtype=torch.float, device=device
    )

    for _ in range(cfg.steps_per_episode):
        actions = select_action_batch(sys, cfg, stat, states).cpu()

        next_states, rewards, dones, _ = zip(
            *[env.step(action) for env, action in zip(sys.envs, actions)]
        )

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)

        sys.memory.push_batch(states, actions, next_states, rewards, dones)

        train_step_batch(sys, cfg, stat)

        states = next_states


def plot_statistics(cfg: Config, stat: Statistics) -> None:
    plt.clf()

    finals = torch.tensor(ts.finals, device=device)
    plt.plot(finals.cpu().numpy(), label="DQN")

    oz_finals = torch.tensor(ts.oz_finals, device=device)
    plt.plot(oz_finals.cpu().numpy(), label="-Oz")

    if finals.size(0) >= cfg.ma_window_size:
        ma_finals = finals.unfold(0, cfg.ma_window_size, 1).mean(1).view(-1)
        ma_finals = torch.cat(
            (torch.ones(cfg.ma_window_size - 1, device=device), ma_finals)
        )
        plt.plot(ma_finals.cpu().numpy(), label="DQN Moving Average")

        ma_oz_finals = oz_finals.unfold(0, cfg.ma_window_size, 1).mean(1).view(-1)
        ma_oz_finals = torch.cat(
            (torch.ones(cfg.ma_window_size - 1, device=device), ma_oz_finals)
        )
        plt.plot(ma_oz_finals.cpu().numpy(), label="-Oz Moving Average")

    plt.xlabel("Episode")
    plt.ylabel("IR Instruction Count Improvement Factor over -O0")
    plt.legend()
    plt.savefig("out.png")


@torch.no_grad()
def train(sys: System, cfg: Config, stat: Statistics) -> None:
    for episode in tqdm(range(0, cfg.episodes, cfg.envs)):
        train_episode_batch(sys, cfg, stat)

        if (episode + 1) % cfg.episodes_per_plot == 0:
            plot_statistics(cfg, stat)


def main(cfg: TrainConfig) -> None:
    with gym.make("llvm-v0") as env:
        env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        env = RandomOrderBenchmarks(env, env.datasets["benchmark://jotaibench-v0"])
        """env = ConstrainedCommandline(
            env,
            flags=[
                "-break-crit-edges",
                "-early-cse-memssa",
                "-gvn-hoist",
                "-gvn",
                "-instcombine",
                "-instsimplify",
                "-jump-threading",
                "-loop-reduce",
                "-loop-rotate",
                "-loop-versioning",
                "-mem2reg",
                "-newgvn",
                "-reg2mem",
                "-simplifycfg",
                "-sroa",
            ],
        )"""
        env.observation_space = "Autophase"
        env.reward_space = "IrInstructionCount"

        n_observations = env.observation_space.shape[0]
        n_actions = env.action_space.n

        policy_net = DQN(n_observations, n_actions, cfg.hidden_size).to(device)
        target_net = DQN(n_observations, n_actions, cfg.hidden_size).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        opt = optim.AdamW(policy_net.parameters(), lr=cfg.lr, amsgrad=True)
        observation_scaler = RunningScaler(shape=env.observation_space.shape)
        reward_scaler = RunningScaler(shape=(1,))
        memory = ReplayMemory(cfg.replay_memory_capacity)

        ts = TrainState(
            env=env,
            policy_net=policy_net,
            target_net=target_net,
            opt=opt,
            observation_scaler=observation_scaler,
            reward_scaler=reward_scaler,
            memory=memory,
            step=0,
        )

        train(ts, cfg)


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)
