"""Distributed running statistics for AMP, with unchanged checkpoint buffers."""

from __future__ import annotations

import torch
import torch.distributed as dist

from rsl_rl.modules import EmpiricalNormalization


@torch.no_grad()
def update_normalization(
    normalizer: EmpiricalNormalization, samples: torch.Tensor, *, distributed: bool
) -> None:
    """Update from the global batch without gathering observations across GPUs.

    All ranks must start with identical buffers and call this at the same times.
    Count, sum and squared deviations are reduced together. Centering around
    the existing mean and accumulating in float64 avoids cancellation for
    observations with large offsets. Local batch sizes may differ.
    """
    if not distributed:
        normalizer.update(samples)
        return
    if not normalizer.training or (normalizer.until is not None and normalizer.count >= normalizer.until):
        return

    centered = samples.double() - normalizer._mean.double()
    width = normalizer._mean.numel()
    moments = torch.cat(
        [
            centered.new_tensor([samples.shape[0]]),
            centered.sum(dim=0).flatten(),
            centered.square().sum(dim=0).flatten(),
        ]
    )
    dist.all_reduce(moments, op=dist.ReduceOp.SUM)
    count = moments[0]
    if count == 0:
        return
    delta = (moments[1 : 1 + width] / count).view_as(normalizer._mean)
    variance = (moments[1 + width :] / count).view_as(delta) - delta.square()
    variance = variance.clamp_min_(0)
    total_count = normalizer.count + count
    rate = count / total_count
    # Parallel population-variance merge, equivalent to a single global update.
    normalizer._mean.add_((rate * delta).to(normalizer._mean.dtype))
    normalizer._var.copy_(
        (1 - rate) * normalizer._var.double() + rate * variance + rate * (1 - rate) * delta.square()
    )
    normalizer._std.copy_(normalizer._var.sqrt())
    normalizer.count.copy_(total_count)
