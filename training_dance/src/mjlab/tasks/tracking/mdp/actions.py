"""Tracking-specific action terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class MotionReferenceJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position residuals around the current motion reference."""

  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> MotionReferenceJointPositionAction:
    return MotionReferenceJointPositionAction(self, env)


class MotionReferenceJointPositionAction(JointPositionAction):
  """Apply ``reference_joint_pos + scale * action`` position targets."""

  cfg: MotionReferenceJointPositionActionCfg

  def __init__(
    self, cfg: MotionReferenceJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    command = cast(MotionCommand, env.command_manager.get_term(cfg.command_name))
    reference_names = tuple(command.robot.joint_names)
    missing = tuple(name for name in self.target_names if name not in reference_names)
    if missing:
      raise ValueError(f"Motion reference is missing action joints: {missing}")
    self._reference_joint_ids = torch.tensor(
      [reference_names.index(name) for name in self.target_names],
      dtype=torch.long,
      device=self.device,
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    command = cast(
      MotionCommand, self._env.command_manager.get_term(self.cfg.command_name)
    )
    # Refresh the offset every policy step so it follows the current reference frame.
    self._offset = command.joint_pos[:, self._reference_joint_ids]
    super().process_actions(actions)
