from tqdm import tqdm
from dataclasses import asdict
import itertools as it
import time
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from model import DQN
from state import State
from config import Config
from logger import Logger
from memory import Memory
from multi_compiler_env import MultiCompilerEnv

device = torch.device("cuda")


def linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    slope = (end - start) / duration
    return max(slope * t + start, end)


def sample_policy_actions(model: DQN, observations: Tensor) -> Tensor:
    policy_actions = model.policy(observations)[0].argmax(dim=-1).view(-1)
    return policy_actions


def sample_epsilon_greedy_actions(policy_actions: Tensor, random_actions: Tensor, threshold: float) -> Tensor:
    assert policy_actions.size(0) == random_actions.size(0)

    count = policy_actions.size(0)    
    mask = torch.rand(count, device=device, dtype=torch.float) > threshold

    actions = torch.zeros(count, device=device, dtype=torch.long)
    actions[mask] = policy_actions[mask]
    actions[~mask] = random_actions[~mask]

    return actions


@torch.no_grad()
def rollout_step(state: State, cfg: Config, logger: Logger) -> None:
    state.model.eval()
    observations = state.env.observations
    policy_actions = sample_policy_actions(state.model, observations)
    # logger.log("actions", policy_actions.tolist())
    random_actions = state.env.sample_actions()

    threshold = linear_schedule(cfg.epsilon_greedy_start, cfg.epsilon_greedy_end, cfg.epsilon_greedy_duration, state.step)
    logger.log("epsilon_greedy_threshold", threshold)

    actions = sample_epsilon_greedy_actions(policy_actions, random_actions, threshold)
    next_observations, rewards, dones, uris = state.env.step(actions)
    if cfg.use_binary_reward:
        rewards = rewards.sign()
    # logger.log("rewards", rewards.tolist())
    if any(dones):
        return True

    costs = state.env.costs
    if (costs == 0).any():
        idx = (costs == 0).nonzero().tolist()
        for i in idx:
            print("zero_cost:", uris[i[0]])
            logger.log("zero_cost", str(uris[i[0]]))
        return True
        
    state.memory.push(observations, actions, next_observations, rewards, uris)
    return False


@torch.no_grad()
def train_step(state: State, cfg: Config, logger: Logger) -> None:
    state.model.train()
    observations, actions, next_observations, rewards, _ = state.memory.sample(cfg.batch_size)
    
    next_observation_values = state.model.target(next_observations)[0].max(1).values
    expected_observation_action_values = (next_observation_values * cfg.gamma) + rewards

    criterion = nn.SmoothL1Loss()
    state.opt.zero_grad()

    with torch.enable_grad():
        out, encoder_out = state.model.policy(observations)
        observation_action_values = out.gather(
            1, actions.unsqueeze(1)
        )
        loss = criterion(observation_action_values, expected_observation_action_values.unsqueeze(1))
        if encoder_out is not None:
            loss += 0.1 * (F.mse_loss(encoder_out, observations, reduction='none') / (observations.var(dim=0, unbiased=False, keepdim=True) + 1e-6)).mean()
        loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        state.model.policy.parameters(), cfg.gradient_norm_max
    )
    logger.log("gradient_norm", gradient_norm.item())

    state.opt.step()

    for target_param, param in zip(
        state.model.target.parameters(), state.model.policy.parameters()
    ):
        target_param.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
    state.step += 1


def train(state: State, cfg: Config, logger: Logger) -> None:
    cfg.output_path.mkdir(exist_ok=True)
    d = asdict(cfg)
    d.pop("output_path")
    (cfg.output_path / "cfg.json").write_text(json.dumps(d))

    for episode in tqdm(range(0, cfg.episodes, cfg.envs)):
        state.env.reset()
        
        start_costs = state.env.costs
        baseline_end_costs = state.env.baseline_costs
        end_costs = state.env.costs

        for _ in range(cfg.steps_per_episode):
            if rollout_step(state, cfg, logger):
                break
            end_costs = state.env.costs

            if len(state.memory) < cfg.memory_capacity_min:
                continue
            
            for _ in range(cfg.batchs_per_episode):
                train_step(state, cfg, logger)
        
        improvement_factor = (start_costs / (end_costs + 1e-5)).log().mean().exp().item()
        baseline_improvement_factor = (start_costs / (baseline_end_costs + 1e-5)).log().mean().exp().item()
        print("improvement_factor =", improvement_factor)
        print("baseline_improvement_factor =", baseline_improvement_factor)

        logger.log("improvement_factor", improvement_factor)
        logger.log("baseline_improvement_factor", baseline_improvement_factor)

        logger.flush()
        torch.save(state.model, cfg.output_path / "model.pt")


def main(cfg: Config, start: Path = None) -> None:
    with MultiCompilerEnv(cfg.envs, cfg.observation_space, cfg.reward_space, cfg.cost, cfg.baseline_cost, cfg.steps_per_episode, cfg.dataset) as env:
        n_observations = env.n_observations
        n_actions = env.n_actions
        
        if start is not None:
            model = DQN(n_observations, n_actions, cfg.hidden_size, cfg.use_autoencoder).to(device)
        else:
            model = torch.load(start).to(device)
        opt = optim.AdamW(model.policy.parameters(), lr=cfg.lr, amsgrad=True)
        
        memory = Memory(capacity=cfg.memory_capacity, observation_shape=(n_observations,)).to(
            device
        )

        state = State(
            env=env,
            model=model, 
            opt=opt,
            memory=memory,
            step=0,
        )
        logger = Logger(cfg.output_path / "log.json")

        train(state, cfg, logger)


# TODO:
# performance
# test script
# sweep


def sweep(result_path: Path) -> None:
    result_path.mkdir(exist_ok=True)

    hps = {
        "epsilon_greedy_start": [0.9],
        "epsilon_greedy_end": [0.01],
        "epsilon_greedy_duration": [32 * 1024],
        "observation_space": ["Autophase"],
        "reward_space": ["IrInstructionCount", "IrInstructionCountNorm"],
        "cost": ["IrInstructionCount"],
        "baseline_cost": ["IrInstructionCountOz"],
        "dataset": ["benchmark://anghabench-v1"],
        "use_binary_reward": [True, False],
        "use_autoencoder": [True, False],
        "steps_per_episode": [64],
        "hidden_size": [256, 512],
        "episodes": [1024],
        "envs": [32],
        "batchs_per_episode": [32],
        "batch_size": [128],
        "gamma": [0.99],
        "lr": [5e-4, 5e-5],
        "memory_capacity": [8 * 1024],
        "memory_capacity_min": [4 * 1024],
        "gradient_norm_max": [10.0],
        "tau": [0.001],
    }

    combs = [dict(zip(hps.keys(), values)) for values in it.product(*hps.values())]
    for comb in combs:
        output_path = result_path / str(time.time())
        cfg = Config(output_path=output_path, **comb)
        print(cfg)
        main(cfg)


if __name__ == "__main__":
    # sweep(Path("results"))
    main(Config(), start=Path("results/checkpoint2/model.pt"))
