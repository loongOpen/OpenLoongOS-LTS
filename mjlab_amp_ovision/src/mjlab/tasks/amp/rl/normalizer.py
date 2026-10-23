"""Running mean/std normalizer used for AMP observations."""

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
  def __init__(self, shape: int | tuple[int, ...], eps: float = 1e-2, until: int | None = None):
    super().__init__()
    if isinstance(shape, int):
      shape = (shape,)
    self.eps = eps
    self.until = until
    self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
    self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
    self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
    self.register_buffer("count", torch.tensor(0, dtype=torch.long))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.training:
      self.update(x)
    return (x - self._mean) / (self._std + self.eps)

  @torch.jit.unused
  def update(self, x: torch.Tensor) -> None:
    if self.until is not None and int(self.count) >= self.until:
      return
    count_x = x.shape[0]
    self.count += count_x
    rate = count_x / self.count.float()
    var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
    mean_x = torch.mean(x, dim=0, keepdim=True)
    delta_mean = mean_x - self._mean
    self._mean += rate * delta_mean
    self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
    self._std = torch.sqrt(torch.clamp(self._var, min=1e-8))
