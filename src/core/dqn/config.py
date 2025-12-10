from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Output
    output_path: Path = Path("test")

    # Epsilon-greedy parameters.
    epsilon_greedy_start: float = 0.9
    epsilon_greedy_end: float = 0.01
    epsilon_greedy_duration: int = 32 * 1024

    # Environment parameters.
    observation_space: str = "Autophase"
    reward_space: str = "IrInstructionCount"
    cost: str = "IrInstructionCount"
    baseline_cost: str = "IrInstructionCountOz"
    
    dataset: str = "benchmark://anghabench-v1" # "benchmark://chstone-v0"
    use_binary_reward: bool = False
    steps_per_episode: int = 256

    # Model parameters.
    hidden_size: int = 512
    use_autoencoder: bool = True
    
    # Training parameters.
    episodes: int = 512 * 1024
    envs: int = 32
    batchs_per_episode: int = 64
    batch_size: int = 128
    gamma: float = 0.99

    # Optimizer parameters.
    lr: float = 5e-5

    # Memory parameters.
    memory_capacity: int = 256 * 1024
    memory_capacity_min: int = 256 * 1024

    # Gradient clipping.
    gradient_norm_max: float = 10.0

    # Soft updates.
    tau: float = 0.001
