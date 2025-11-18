from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class RunningScaler(nn.Module):
    def __init__(
        self,
        shape: Tuple[int, ...],
        epsilon: float = 1e-4,
    ) -> None:
        """
        Calculates the running mean and variance of a data stream. Adapted from
        Stable-Baselines3.
        """
        super().__init__()
        self.shape = shape
        self.epsilon = epsilon

        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float))
        self.count = epsilon

    @torch.no_grad()
    def update(self, batch: Tensor) -> None:
        batch_size = batch.size(0)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        self.update_from_moments(batch_mean, batch_var, batch_size)

    @torch.no_grad()
    def update_from_moments(
        self, batch_mean: Tensor, batch_var: Tensor, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    @torch.no_grad()
    def scale(self, x: Tensor) -> Tensor:
        scaled = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return scaled.clamp(-5.0, 5.0)
