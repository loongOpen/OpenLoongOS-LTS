from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def _motion_reference_at(
  command: MotionCommand, frame_indices: torch.Tensor
) -> torch.Tensor:
  """Encode reference frames as [lin vel, ang vel, gravity, joint position]."""
  batch_size, sequence_length = frame_indices.shape
  flat_indices = frame_indices.reshape(-1)
  anchor_index = command.motion_anchor_body_index

  joint_pos = command.motion.joint_pos[flat_indices].view(
    batch_size, sequence_length, -1
  )
  anchor_quat_w = command.motion.body_quat_w[flat_indices, anchor_index].view(
    batch_size, sequence_length, 4
  )
  anchor_lin_vel_w = command.motion.body_lin_vel_w[flat_indices, anchor_index].view(
    batch_size, sequence_length, 3
  )
  anchor_ang_vel_w = command.motion.body_ang_vel_w[flat_indices, anchor_index].view(
    batch_size, sequence_length, 3
  )

  flat_quat_w = anchor_quat_w.reshape(-1, 4)
  base_lin_vel_b = quat_apply_inverse(
    flat_quat_w, anchor_lin_vel_w.reshape(-1, 3)
  ).view(batch_size, sequence_length, 3)
  base_ang_vel_b = quat_apply_inverse(
    flat_quat_w, anchor_ang_vel_w.reshape(-1, 3)
  ).view(batch_size, sequence_length, 3)

  gravity_w = command.robot.data.gravity_vec_w[:, None, :].expand(
    -1, sequence_length, -1
  )
  gravity_b = quat_apply_inverse(flat_quat_w, gravity_w.reshape(-1, 3)).view(
    batch_size, sequence_length, 3
  )

  return torch.cat([base_lin_vel_b, base_ang_vel_b, gravity_b, joint_pos], dim=-1)


def centered_motion_reference(
  env: ManagerBasedRlEnv,
  command_name: str,
  num_past_frames: int,
  num_future_frames: int,
  step_size: int = 1,
) -> torch.Tensor:
  """Return a clamped past/current/future local-reference window."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  offsets = (
    torch.arange(
      -num_past_frames,
      num_future_frames + 1,
      device=command.device,
      dtype=torch.long,
    )
    * step_size
  )
  frame_indices = command.time_steps[:, None] + offsets[None, :]
  frame_indices = frame_indices.clamp(0, command.motion.time_step_total - 1)
  reference = _motion_reference_at(command, frame_indices)
  return reference.reshape(env.num_envs, -1)


def current_motion_reference(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return the current ``9 + num_joints`` local reference frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return _motion_reference_at(command, command.time_steps[:, None]).squeeze(1)


def proprioception(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one ``6 + 3 * num_joints`` real-robot proprioceptive frame.

  Layout: projected gravity, gyroscope, relative joint position, joint velocity,
  and previous policy action. History is managed by ``ObservationManager``.
  """
  return torch.cat(
    [
      env.scene["robot"].data.projected_gravity_b,
      env.scene["robot/imu_ang_vel"].data,
      env.scene["robot"].data.joint_pos_biased
      - env.scene["robot"].data.default_joint_pos,
      env.scene["robot"].data.joint_vel - env.scene["robot"].data.default_joint_vel,
      env.action_manager.action,
    ],
    dim=-1,
  )
