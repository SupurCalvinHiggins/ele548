from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor


class Memory(nn.Module):
    def __init__(self, capacity: int, observation_shape: Tuple[int]) -> None:
        super().__init__()
        assert capacity > 0

        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.register_buffer(
            "states", torch.zeros((capacity, *observation_shape), dtype=torch.float)
        )
        self.register_buffer(
            "next_states", torch.zeros((capacity, *observation_shape), dtype=torch.float)
        )
        self.register_buffer("actions", torch.zeros((capacity,), dtype=torch.long))
        self.register_buffer("rewards", torch.zeros((capacity,), dtype=torch.float))
        self.uris = ["" for _ in range(capacity)]

    def push(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
        rewards: Tensor,
        uris: List[str]
    ) -> None:
        batch_size = states.size(0)

        device = self.states.device
        idxs = (torch.arange(batch_size, device=device) + self.position) % self.capacity

        self.states[idxs] = states
        self.next_states[idxs] = next_states
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        
        for offset, uri in enumerate(uris):
            idx = (offset + self.position) % self.capacity
            self.uris[idx] = uri         

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(
        self, batch_size: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = self.states.device
        idxs = torch.randint(0, self.size, (batch_size,), device=device)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.next_states[idxs],
            self.rewards[idxs],
            idxs,
        )
    
    def uris(self, idxs: Tensor) -> List[str]:
        return [self.uris[idx] for idx in idxs.tolist()]

    def __len__(self) -> int:
        return self.size
