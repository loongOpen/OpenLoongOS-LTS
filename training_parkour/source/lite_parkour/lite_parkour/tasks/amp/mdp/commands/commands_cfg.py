from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .velocity_command import AmpVelocityCommand


@configclass
class AmpVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`AmpVelocityCommand`.

    Kept free of any import that pulls ``isaaclab.markers`` / pxr, so this module is safe to
    import at package load time (before ``SimulationApp`` starts). The command term is
    referenced by a string ``class_type`` path and only imported at runtime (after the app is
    up), mirroring the stock ``UniformVelocityCommandCfg``.
    """

    class_type: type[AmpVelocityCommand] | str = "{DIR}.velocity_command:AmpVelocityCommand"

    rel_turning_envs: float = 0.0
    """Probability of turning in place per resample, exclusive of standing and walking."""

    turning_ang_vel_range: tuple[float, float] = (0.3, 1.0)
    """Magnitude of the turning yaw command [rad/s]; left and right are equally likely."""

