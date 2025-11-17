from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class RunningScaler(nn.Module):
    def __init__(
        self,
        shape: Tuple[int, ...],
        batch_size: int,
        epsilon: float = 1e-4,
    ) -> None:
        """
        Calculates the running mean and variance of a data stream. Adapted from
        Stable-Baselines3.
        """
        self.shape = shape
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.batch_n = 0
        self.batch = self.register_buffer(
            torch.zeroes((batch_size, *shape), dtype=torch.float)
        )

        self.mean = self.register_buffer(torch.zeros(shape, dtype=torch.float))
        self.var = self.register_buffer(torch.ones(shape, dtype=torch.float))
        self.count = epsilon

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        assert x.shape == self.shape

        self.batch[self.batch_n] = x
        self.batch_n += 1

        if self.batch_n == self.batch_size:
            self.batch_n = 0
            batch_mean = self.batch.mean(axis=0)
            batch_var = self.batch.var(axis=0)
            self.update_from_moments(batch_mean, batch_var, self.batch_size)

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
        eps = 1e-5
        scaled = (x - self.mean) / torch.sqrt(self.var + eps)
        return scaled.clamp(-5.0, 5.0)
