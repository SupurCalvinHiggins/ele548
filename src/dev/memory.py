import random
from collections import deque
from dataclasses import dataclass
from typing import List

from torch import Tensor


@dataclass
class Transition:
    state: Tensor
    action: Tensor
    next_state: Tensor
    reward: Tensor


class Memory:
    def __init__(self, capacity: int) -> None:
        assert capacity > 0
        self.memory = deque([], maxlen=capacity)

    def emplace(
        self, state: Tensor, action: Tensor, next_state: Tensor, reward: Tensor
    ) -> None:
        self.push(Transition(state, action, next_state, reward))

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, k: int) -> List[Transition]:
        return random.sample(self.memory, k)

    def push_batch(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
        rewards: Tensor,
        dones: Tensor,
    ) -> None:
        pass

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...

    def __len__(self) -> int:
        return len(self.memory)
