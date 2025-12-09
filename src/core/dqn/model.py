import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PolicyNetwork(nn.Module):
    def __init__(self, n_observations: int, n_actions, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


class DQN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.policy = PolicyNetwork(*args, **kwargs)
        self.target = PolicyNetwork(*args, **kwargs)
        self.target.load_state_dict(self.policy.state_dict())