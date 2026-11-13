"""53-D AMP observations for the 20-DoF LingLong.

Layout (must match AmpMotionLoader):
  joint_q20 | joint_dq20 | base_lin3 | base_ang3 | foot6 | root_z1
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

AMP_OBS_DIM = 53
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def compute_amp_observations(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Online AMP observation, aligned with the motion JSON -> AMP loader layout."""
  robot: Entity = env.scene["robot"]
  foot_ids, _ = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)

  root_pos = robot.data.root_link_pos_w
  root_quat = robot.data.root_link_quat_w
  foot_pos_w = robot.data.body_link_pos_w[:, foot_ids, :]
  left_foot = quat_apply_inverse(root_quat, foot_pos_w[:, 0] - root_pos)
  right_foot = quat_apply_inverse(root_quat, foot_pos_w[:, 1] - root_pos)

  amp_obs = torch.cat(
    (
      robot.data.joint_pos,
      robot.data.joint_vel,
      robot.data.root_link_lin_vel_b,
      robot.data.root_link_ang_vel_b,
      left_foot,
      right_foot,
      root_pos[:, 2:3],
    ),
    dim=-1,
  )
  if amp_obs.shape[-1] != AMP_OBS_DIM:
    raise RuntimeError(
      f"AMP obs dim {amp_obs.shape[-1]} != {AMP_OBS_DIM}. "
      f"joint_pos={tuple(robot.data.joint_pos.shape)} "
      f"joint_vel={tuple(robot.data.joint_vel.shape)}"
    )
  return amp_obs
