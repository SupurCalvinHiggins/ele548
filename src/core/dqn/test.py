import torch
import gym

from warnings import catch_warnings


with catch_warnings():
    from compiler_gym.envs import CompilerEnv


from model import DQN


def make_env(dataset: str) -> CompilerEnv:
    env = gym.make("llvm-v0")
    env.res
    return env


def test(results_path: Path):
    for model_path in results_path.glob("**/*.pt"):
        model = torch.load(model_path)
        with make_env() as env:
            pass

    # get the model file
    # load the model
    # set up envs
    # for each env, do full rollout using model
    # get final score with baseline
    # 
    pass