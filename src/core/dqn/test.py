import torch
import gym
import json
from pathlib import Path

from warnings import catch_warnings


with catch_warnings():
    from compiler_gym.envs import CompilerEnv
    from compiler_gym.wrappers import IterateOverBenchmarks


from model import DQN


device = torch.device("cuda")


def make_env(dataset: str) -> CompilerEnv:
    env = gym.make("llvm-v0")
    env = IterateOverBenchmarks(env, env.datasets[dataset])
    return env


@torch.no_grad()
def test_model(model, dataset, steps: int = 64):
    ifs = []
    baseline_ifs = []
    with make_env(dataset) as env:
        while True:
            try:
                env.reset()
            except:
                break
            cost_start = env.observation["IrInstructionCount"]
            baseline = env.observation["IrInstructionCountOz"]
            cost_end = env.observation["IrInstructionCount"]
            for _ in range(steps):
                cost_end = env.observation["IrInstructionCount"]
                observations = torch.tensor(env.observation["Autophase"], dtype=torch.float, device=device).unsqueeze(0)
                action = model.policy(observations)[0].argmax(dim=-1).view(-1).item()
                _, _, done, _ = env.step(action)
                if done:
                    break
            ifs.append(cost_start / cost_end)
            baseline_ifs.append(cost_start / baseline)
        
    geomean_if = torch.tensor(ifs).log().mean().exp().item()
    geomean_baseline = torch.tensor(baseline_ifs).log().mean().exp().item()
    return geomean_if, geomean_baseline


def test(results_path: Path, dataset: str):
    for model_path in results_path.glob("**/*.pt"):
        try:
            model = torch.load(model_path)
            improvement_factor, baseline_improvement_factor = test_model(model, dataset)
            print(model_path, improvement_factor, baseline_improvement_factor)
            output_path = model_path.parent / f"metrics.json"
            output_path.write_text(json.dumps({"improvement_factor": improvement_factor, "baseline_improvement_factor": baseline_improvement_factor}))
        except Exception as e:
            print(e)
    # get the model file
    # load the model
    # set up envs
    # for each env, do full rollout using model
    # get final score with baseline
    # 
    # pass

if __name__ == "__main__":
    test(Path("results"), "benchmark://chstone-v0")