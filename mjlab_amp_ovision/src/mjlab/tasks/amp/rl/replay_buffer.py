"""Fixed-size replay buffer for AMP (s, s_next) transitions."""

from __future__ import annotations

import numpy as np
import torch


class AmpReplayBuffer:
  def __init__(self, obs_dim: int, buffer_size: int, device: str | torch.device) -> None:
    self.device = torch.device(device)
    self.buffer_size = int(buffer_size)
    self.states = torch.zeros(self.buffer_size, obs_dim, device=self.device)
    self.next_states = torch.zeros(self.buffer_size, obs_dim, device=self.device)
    self.step = 0
    self.num_samples = 0

  def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
    states = states.to(self.device)
    next_states = next_states.to(self.device)
    n = states.shape[0]
    end = self.step + n
    if end > self.buffer_size:
      first = self.buffer_size - self.step
      self.states[self.step :] = states[:first]
      self.next_states[self.step :] = next_states[:first]
      self.states[: n - first] = states[first:]
      self.next_states[: n - first] = next_states[first:]
    else:
      self.states[self.step : end] = states
      self.next_states[self.step : end] = next_states
    self.num_samples = min(self.buffer_size, max(end, self.num_samples))
    self.step = (self.step + n) % self.buffer_size

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    n = max(self.num_samples, 1)
    for _ in range(num_mini_batch):
      idxs = np.random.choice(n, size=mini_batch_size)
      yield self.states[idxs], self.next_states[idxs]
