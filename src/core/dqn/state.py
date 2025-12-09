from dataclasses import dataclass

from memory import Memory
from model import DQN
from multi_compiler_env import MultiCompilerEnv

import torch.optim as optim
from warnings import catch_warnings


with catch_warnings():
    from compiler_gym.envs import CompilerEnv


@dataclass
class State:
    env: MultiCompilerEnv
    memory: Memory
    model: DQN
    opt: optim.AdamW
    step: int