import math
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
from warnings import catch_warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

with catch_warnings():
    import compiler_gym
    import gym
    from compiler_gym.envs import CompilerEnv
    from compiler_gym.wrappers import RandomOrderBenchmarks, TimeLimit

device = torch.device("cuda")


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
    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@dataclass
class TrainState:
    env: CompilerEnv
    policy_net: DQN
    target_net: DQN
    opt: optim.AdamW
    memory: ReplayMemory
    state: Optional[Tensor] = None
    step: int = 0


@dataclass
class TrainConfig:
    eps_start: float = 0.9
    eps_end: float = 0.01
    eps_decay: int = 2500
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    episodes: int = 1000
    max_episode_steps: int = 100
    replay_memory_capacity: int = 10_000
    max_grad_value: int = 100


EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500


def select_action(ts: TrainState, cfg: TrainConfig) -> Tensor:
    sample = random.random()
    eps_threshold = cfg.eps_start + (cfg.eps_start - cfg.eps_end) * math.exp(
        -1.0 * ts.step / cfg.eps_decay
    )
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.tensor(
                ts.env.observation[ts.env.observation_space],
                dtype=torch.float,
                device=device,
            ).unsqueeze(0)
            return ts.policy_net(state).argmax(dim=-1).view(1, 1)
    else:
        return torch.tensor(
            [[ts.env.action_space.sample()]], device=device, dtype=torch.long
        )


def train_step(ts: TrainState, cfg: TrainConfig) -> None:
    if len(ts.memory) < cfg.batch_size:
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

    state_action_values = ts.policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(cfg.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            ts.target_net(non_final_next_states).max(1).values
        )
    expected_state_action_values = (next_state_values * cfg.gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    ts.opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(ts.policy_net.parameters(), cfg.max_grad_value)
    ts.opt.step()

    ts.step += 1


def train_episode(ts: TrainState, cfg: TrainConfig) -> None:
    ts.env.reset()

    while True:
        state = torch.tensor(
            ts.env.observation[ts.env.observation_space],
            dtype=torch.float,
            device=device,
        ).unsqueeze(0)
        action = select_action(ts, cfg)

        observation, reward, done, info = ts.env.step(action.item())
        print(reward)
        reward = torch.tensor([reward], device=device)
        if done:
            return

        next_state = torch.tensor(
            observation, dtype=torch.float, device=device
        ).unsqueeze(0)
        ts.memory.emplace(state, action, next_state, reward)

        train_step(ts, cfg)

        target_net_state_dict = ts.target_net.state_dict()
        policy_net_state_dict = ts.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * cfg.tau + target_net_state_dict[key] * (1 - cfg.tau)
        ts.target_net.load_state_dict(target_net_state_dict)


def train(ts: TrainState, cfg: TrainConfig) -> None:
    for episode in range(cfg.episodes):
        train_episode(ts, cfg)


def main(cfg: TrainConfig) -> None:
    with gym.make("llvm-v0") as env:
        env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        env = RandomOrderBenchmarks(env, env.datasets["benchmark://jotaibench-v0"])
        env.observation_space = "Autophase"
        env.reward_space = "IrInstructionCount"

        n_observations = env.observation[env.observation_space].shape[0]
        n_actions = env.action_space.n

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        opt = optim.AdamW(policy_net.parameters(), lr=cfg.lr, amsgrad=True)

        ts = TrainState(
            env=env,
            policy_net=policy_net,
            target_net=target_net,
            opt=opt,
            memory=ReplayMemory(cfg.replay_memory_capacity),
            step=0,
        )

        train_episode(ts, cfg)


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)
