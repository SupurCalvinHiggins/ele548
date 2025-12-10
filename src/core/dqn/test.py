import torch
import gym
import json
from pathlib import Path
from typing import Iterator, Optional, Dict, List
from tqdm import tqdm
import torch.nn as nn
import multiprocessing as mp

from warnings import catch_warnings


with catch_warnings():
    from compiler_gym.envs import CompilerEnv
    from compiler_gym.wrappers import IterateOverBenchmarks


# device = torch.device("cuda")
device = torch.device("cpu")


def make_env(dataset_uri: str) -> CompilerEnv:
    env = gym.make("llvm-v0")
    env = IterateOverBenchmarks(env, env.datasets[dataset_uri])
    return env


def iter_env(env: CompilerEnv) -> Iterator[None]:
    while True:
        try:
            env.reset()
            yield
        except StopIteration:
            break


@torch.no_grad()
def benchmark_metrics_with_improvement_factors(benchmark_metrics: Dict) -> Dict:
    costs = torch.tensor(benchmark_metrics["costs"])
    o0_cost = benchmark_metrics["costs"][0]
    oz_cost = benchmark_metrics["oz_cost"]
    benchmark_metrics["o0_rl_improvement_factors"] = (o0_cost / costs).cummax(dim=0).values.tolist()
    benchmark_metrics["rl_oz_improvement_factors"] = (oz_cost / costs).cummax(dim=0).values.tolist()
    benchmark_metrics["o0_oz_improvement_factor"] = o0_cost / oz_cost
    return benchmark_metrics


@torch.no_grad()
def evaluate_model_on_benchmark(model: nn.Module, env: CompilerEnv, max_steps: int) -> Optional[Dict]:
    assert max_steps > 0

    costs = [env.observation["IrInstructionCount"]]
    oz_cost = env.observation["IrInstructionCountOz"]
    if costs[0] == 0 or oz_cost == 0:
        return None

    for _ in range(max_steps):
        observation = torch.tensor(env.observation["Autophase"], dtype=torch.float, device=device)
        action = model.policy(observation.unsqueeze(0))[0].argmax(dim=-1).item()
        _, _, done, _ = env.step(action)
        if done:
            return None
        cost = env.observation["IrInstructionCount"]
        if cost == 0:
            return None
        costs.append(cost)
    
    return benchmark_metrics_with_improvement_factors({"costs": costs, "oz_cost": oz_cost})


@torch.no_grad()
def env_metrics_with_improvement_factors(env_metrics: Dict) -> Dict:
    o0_rl_improvement_factors = torch.tensor([benchmark_metrics["o0_rl_improvement_factors"] for _, benchmark_metrics in env_metrics.items()])
    rl_oz_improvement_factors = torch.tensor([benchmark_metrics["rl_oz_improvement_factors"] for _, benchmark_metrics in env_metrics.items()])
    o0_oz_improvement_factor = torch.tensor([benchmark_metrics["o0_oz_improvement_factor"] for _, benchmark_metrics in env_metrics.items()])

    # [b, s]
    env_metrics["mean_o0_rl_improvement_factors"] = o0_rl_improvement_factors.log().mean(dim=0).exp().tolist()
    env_metrics["min_o0_rl_improvement_factors"] = o0_rl_improvement_factors.min(dim=0).values.tolist()
    env_metrics["max_o0_rl_improvement_factors"] = o0_rl_improvement_factors.max(dim=0).values.tolist()

    env_metrics["mean_rl_oz_improvement_factors"] = rl_oz_improvement_factors.log().mean(dim=0).exp().tolist()
    env_metrics["min_rl_oz_improvement_factors"] = rl_oz_improvement_factors.min(dim=0).values.tolist()
    env_metrics["max_rl_oz_improvement_factors"]= rl_oz_improvement_factors.max(dim=0).values.tolist()

    env_metrics["mean_o0_oz_improvement_factor"] = o0_oz_improvement_factor.log().mean(dim=0).exp().tolist()
    env_metrics["min_o0_oz_improvement_factor"] = o0_oz_improvement_factor.min(dim=0).values.tolist()
    env_metrics["max_o0_oz_improvement_factor"] = o0_oz_improvement_factor.max(dim=0).values.tolist()

    return env_metrics


def evaluate_model_on_dataset(model: nn.Module, dataset_uri: str, max_steps: int) -> Dict:
    assert max_steps > 0

    metrics = {}
    with make_env(dataset_uri) as env:
        num_benchmarks = len(env.datasets[dataset_uri])
        for _ in tqdm(iter_env(env), total=num_benchmarks):
            benchmark_uri = str(env.benchmark.uri)
            benchmark_metrics = evaluate_model_on_benchmark(model, env, max_steps)
            if benchmark_metrics is None:
                print(f"Failed: dataset_uri = {dataset_uri}, benchmark_uri = {benchmark_uri}")
                continue
            metrics[benchmark_uri] = benchmark_metrics
    
    return env_metrics_with_improvement_factors(metrics)


def evaluate_model_on_datasets(model: nn.Module, dataset_uris: List[str], max_steps: int) -> Dict:
    assert max_steps > 0

    metrics = {}
    for dataset_uri in dataset_uris:
        dataset_metrics = evaluate_model_on_dataset(model, dataset_uri, max_steps)
        metrics[dataset_uri] = dataset_metrics
    
    return metrics


def worker_fn(model_path: Path, dataset_uris: List[str], max_steps: int) -> None:
    model = torch.load(model_path).to(device)
    metrics = evaluate_model_on_datasets(model, dataset_uris, max_steps)
    metrics_path = model_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))


def evaluate_all(results_path: Path, dataset_uris: List[str], max_steps: int) -> None:
    with mp.Pool(8) as pool:
        pool.starmap(worker_fn, ((model_path, dataset_uris, max_steps) for model_path in  results_path.glob("**/*.pt")))

        

if __name__ == "__main__":
    evaluate_all(
        Path("results"), 
        [
            # "benchmark://blas-v0",
            "benchmark://cbench-v1",
            # "benchmark://chstone-v0",
            # "benchmark://linux-v0",
            # "benchmark://mibench-v1",
            # "benchmark://npb-v0",
            # "benchmark://opencv-v0",
            # "benchmark://tensorflow-v0"
        ],
        256
    )