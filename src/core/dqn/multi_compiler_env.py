from warnings import catch_warnings
from typing import Tuple, List

with catch_warnings():
    import gym
    from compiler_gym.envs import CompilerEnv
    from compiler_gym.wrappers import (
        RandomOrderBenchmarks,
    )


import torch
import numpy as np
from torch import Tensor


device = torch.device("cuda")


class MultiCompilerEnv:
    def __init__(self, num_envs: int, observation_space: str, reward_space: str, cost_name: str, baseline_cost_name: str, steps_per_episode: int, dataset: str) -> None:
        self.envs = []
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.reward_space = reward_space
        self.cost_name = cost_name
        self.baseline_cost_name = baseline_cost_name
        self.steps_per_episode = steps_per_episode
        self.dataset = dataset

    def __enter__(self) -> None:
        env = gym.make("llvm-v0")
        envs = [env] + [env.fork() for _ in range(self.num_envs - 1)]

        def configure_env(env: CompilerEnv) -> CompilerEnv:
            env = RandomOrderBenchmarks(env, env.datasets[self.dataset])
            env.observation_space = self.observation_space
            env.reward_space = self.reward_space
            return env

        self.envs = [configure_env(env) for env in envs]
        return self

    def __exit__(self, type, value, exc_traceback) -> None:
        for env in self.envs:
            env.close()
        self.envs = []

    def __len__(self) -> int:
        return len(self.envs)
    
    def convert_observations(self, x: List[np.ndarray]) -> Tensor:
        if x[0].ndim == 2:
            x = [r.mean(0) for r in x]
        x = torch.tensor(np.array(x), dtype=torch.float, device=device)
        return x
    
    def convert_reward(self, x: List[np.ndarray]) -> Tensor:
        x = torch.tensor(np.array(x), dtype=torch.float, device=device)
        return x

    def convert_costs_or_actions(self, x: List[int]) -> Tensor:
        return torch.tensor(x, dtype=torch.long, device=device)

    @property
    def observations(self) -> Tensor:
        observations = [env.observation[self.observation_space] for env in self.envs]
        return self.convert_observations(observations)

    @property
    def costs(self) -> Tensor:
        costs = [env.observation[self.cost_name] for env in self.envs]
        return self.convert_costs_or_actions(costs)

    @property
    def baseline_costs(self) -> Tensor:
        baseline_costs = [env.observation[self.baseline_cost_name] for env in self.envs]
        return self.convert_costs_or_actions(baseline_costs)

    @property
    def n_observations(self) -> int:
        return self.envs[0].observation[self.observation_space].shape[-1]

    @property
    def n_actions(self) -> int:
        return self.envs[0].action_space.n

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tuple[str]]:
        count = actions.size(0)
        assert count == len(self)

        actions = actions.cpu()

        next_observations, rewards, _, _ = zip(
            *[env.step(action.item()) for env, action in zip(self.envs, actions)]
        )

        next_observations = self.convert_observations(next_observations)
        rewards = self.convert_reward(rewards)
        uris = (env.benchmark.uri for env in self.envs)

        return next_observations, rewards, uris

    def reset(self) -> Tensor:
        observations = [env.reset() for env in self.envs]
        return self.convert_observations(observations)

    def sample_actions(self) -> Tensor:
        actions = [env.action_space.sample() for env in self.envs]
        return self.convert_costs_or_actions(actions)