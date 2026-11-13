from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

from .sampling import resample_turning_commands, validate_turning_config


class AmpVelocityCommand(UniformVelocityCommand):
    """Sample walking, standing and turn-in-place commands on reset and at intervals."""

    def __init__(self, cfg, env):
        validate_turning_config(cfg)
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device).flatten()
        if env_ids.numel() == 0:
            return
        super()._resample_command(env_ids)
        resample_turning_commands(
            self.cfg, env_ids, self.vel_command_b, self.is_standing_env, self.is_heading_env
        )
