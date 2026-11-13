"""Tensor-only turning sampling, independent of the simulation runtime."""

import torch


def validate_turning_config(cfg):
    standing, turning = cfg.rel_standing_envs, cfg.rel_turning_envs
    if not (0.0 <= standing <= 1.0 and 0.0 <= turning <= 1.0 and standing + turning <= 1.0):
        raise ValueError("Standing and turning probabilities must be in [0, 1] and sum to at most 1.")
    if turning > 0.0:
        low, high = cfg.turning_ang_vel_range
        yaw_low, yaw_high = cfg.ranges.ang_vel_z
        if not (0.0 < low <= high and -high >= yaw_low and high <= yaw_high):
            raise ValueError("Turning magnitudes must be positive and fit both signs of ranges.ang_vel_z.")


def resample_turning_commands(
    cfg, env_ids, velocity, is_standing, is_heading, *, standing_probability=None
):
    """Replace a fraction of non-standing commands with constant yaw and zero XY velocity.

    The configured turning probability is over all samples, not just the non-standing ones.
    Reference-aligned resets never stand, so their caller passes standing_probability=0.
    Disabled sampling leaves the parent command and random generator unchanged.
    """
    if cfg.rel_turning_envs == 0.0 or env_ids.numel() == 0:
        return env_ids[:0]
    if standing_probability is None:
        standing_probability = cfg.rel_standing_envs
    probability = cfg.rel_turning_envs / (1.0 - standing_probability)
    selected = (torch.rand(env_ids.numel(), device=velocity.device) < probability) & ~is_standing[env_ids]
    turning_ids = env_ids[selected]
    magnitude = torch.empty(turning_ids.numel(), device=velocity.device).uniform_(*cfg.turning_ang_vel_range)
    sign = 2 * torch.randint(2, (turning_ids.numel(),), device=velocity.device) - 1
    velocity[turning_ids, :2] = 0.0
    velocity[turning_ids, 2] = magnitude * sign
    # The parent's heading controller must not overwrite the sampled yaw rate.
    is_heading[turning_ids] = False
    return turning_ids
