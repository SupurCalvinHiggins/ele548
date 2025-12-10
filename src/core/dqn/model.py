import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AutoEncoder(nn.Module):
    def __init__(self, n_observations: int) -> None:
        super().__init__()
        self.embedding_size = n_observations // 4
        embedding_size = self.embedding_size
        self.fc1 = nn.Linear(n_observations, 2 * embedding_size)
        self.ln1 = nn.LayerNorm(2 * embedding_size)
        self.fc2 = nn.Linear(2 * embedding_size, embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.fc3 = nn.Linear(embedding_size, 2 * embedding_size)
        self.ln3 = nn.LayerNorm(2 * embedding_size)
        self.fc4 = nn.Linear(2 * embedding_size, n_observations)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        out = x
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        encoder_out = x
        return out, encoder_out 


class PolicyNetwork(nn.Module):
    def __init__(self, n_observations: int, n_actions, hidden_size: int, use_autoencoder: bool) -> None:
        super().__init__()
        if use_autoencoder:
            self.enc = AutoEncoder(n_observations)
        else:
            self.enc = None
        
        self.fc1 = nn.Linear(n_observations if not use_autoencoder else self.enc.embedding_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "enc") and self.enc is not None:
            x, encoder_out = self.enc(x)
        else:
            encoder_out = None
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x, encoder_out 


class DQN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.policy = PolicyNetwork(*args, **kwargs)
        self.target = PolicyNetwork(*args, **kwargs)
        self.target.load_state_dict(self.policy.state_dict())
