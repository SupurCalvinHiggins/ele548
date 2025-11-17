import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from warnings import catch_warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


class RunningScaler:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        Adapted from stable_baselines3.

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = torch.zeros(shape, dtype=torch.float, device=device)
        self.var = torch.ones(shape, dtype=torch.float, device=device)
        self.count = epsilon

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(
        self, batch_mean: Tensor, batch_var: Tensor, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    @torch.no_grad()
    def scale(self, x: Tensor) -> Tensor:
        eps = 1e-5
        scaled = (x - self.mean) / torch.sqrt(self.var + eps)
        return scaled.clamp(-5.0, 5.0)


@dataclass
class Transition:
    state: Tensor
    action: Tensor
    next_state: Tensor
    reward: Tensor


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        assert capacity > 0
        self.memory = deque([], maxlen=capacity)

    def emplace(
        self, state: Tensor, action: Tensor, next_state: Tensor, reward: Tensor
    ) -> None:
        self.push(Transition(state, action, next_state, reward))

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, k: int) -> List[Transition]:
        return random.sample(self.memory, k)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# TODO: things to track
# loss, reward, grad, Q-value, input dist. after norm statistics
# action distribution
# epsilon greedy value
# action had no effect (in info)
# make a new Statistics structure, rename existing ones


@dataclass
class TrainState:
    env: CompilerEnv
    policy_net: DQN
    target_net: DQN
    opt: optim.AdamW
    memory: ReplayMemory
    observation_scaler: RunningScaler
    reward_scaler: RunningScaler
    state: Optional[Tensor] = None
    step: int = 0
    finals: List[float] = field(default_factory=list)
    oz_finals: List[float] = field(default_factory=list)
    rewards: List[List[float]] = field(default_factory=list)
    losses: List[List[float]] = field(default_factory=list)


@dataclass
class TrainConfig:
    eps_start: float = 0.9
    eps_end: float = 0.01
    eps_decay: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-5
    episodes: int = 100_000
    max_episode_steps: int = 25
    replay_memory_capacity: int = 100_000
    max_grad_value: int = 100
    ma_window_size: int = 100
    hidden_size: int = 256


def select_action(ts: TrainState, cfg: TrainConfig) -> Tensor:
    sample = random.random()
    eps_threshold = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(
        -1.0 * ts.step / cfg.eps_decay
    )
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.tensor(
                ts.env.observation["Autophase"],
                dtype=torch.float,
                device=device,
            ).unsqueeze(0)
            return (
                ts.policy_net(ts.observation_scaler.scale(state))
                .argmax(dim=-1)
                .view(1, 1)
            )
    else:
        return torch.tensor(
            [[ts.env.action_space.sample()]], device=device, dtype=torch.long
        )


def train_step(ts: TrainState, cfg: TrainConfig) -> None:
    # TODO: move to config
    if len(ts.memory) < max(10 * cfg.batch_size, 10_000):
        return

    batch = ts.memory.sample(cfg.batch_size)

    non_final_mask = torch.tensor(
        tuple(map(lambda t: t.next_state is not None, batch)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat(
        [t.next_state for t in batch if t.next_state is not None]
    )

    state_batch = torch.cat([t.state for t in batch])
    action_batch = torch.cat([t.action for t in batch])
    reward_batch = torch.cat([t.reward for t in batch])
    ts.observation_scaler.update(state_batch)
    # print("reward mean/std:", reward_batch.mean().item(), reward_batch.std().item())

    state_action_values = ts.policy_net(
        ts.observation_scaler.scale(state_batch)
    ).gather(1, action_batch)

    next_state_values = torch.zeros(cfg.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            ts.target_net(ts.observation_scaler.scale(non_final_next_states))
            .max(1)
            .values
        )
    expected_state_action_values = (next_state_values * cfg.gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    ts.opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(ts.policy_net.parameters(), cfg.max_grad_value)
    ts.opt.step()

    ts.step += 1

    target_net_state_dict = ts.target_net.state_dict()
    policy_net_state_dict = ts.policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[
            key
        ] * cfg.tau + target_net_state_dict[key] * (1 - cfg.tau)
    ts.target_net.load_state_dict(target_net_state_dict)


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


def train(ts: TrainState, cfg: TrainConfig) -> None:
    for episode in tqdm(range(cfg.episodes)):
        train_episode(ts, cfg)

        if episode % 100 != 0:
            continue

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
