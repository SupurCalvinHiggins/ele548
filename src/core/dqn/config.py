from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Output
    output_path: Path = Path("test")

    # Epsilon-greedy parameters.
    epsilon_greedy_start: float = 0.9
    epsilon_greedy_end: float = 0.01
    epsilon_greedy_duration: int = 25_000

    # Environment parameters.
    observation_space: str = "Autophase"
    reward_space: str = "IrInstructionCount"
    cost: str = "IrInstructionCount"
    baseline_cost: str = "IrInstructionCountOz"
    
    dataset: str = "benchmark://jotaibench-v0" # "benchmark://anghabench-v1" # "benchmark://chstone-v0"
    use_binary_reward: bool = False
    steps_per_episode: int = 256

    # Model parameters.
    hidden_size: int = 256
    
    # Training parameters.
    episodes: int = 8 * 1024
    envs: int = 32
    batchs_per_episode: int = 64
    batch_size: int = 128
    gamma: float = 0.99

    # Optimizer parameters.
    lr: float = 3e-5

    # Memory parameters.
    memory_capacity: int = 10_000
    memory_capacity_min: int = 1_000

    # Gradient clipping.
    gradient_norm_max: float = 80.0

    # Soft updates.
    tau: float = 0.005
