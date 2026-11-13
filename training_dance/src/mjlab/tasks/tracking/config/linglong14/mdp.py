"""Linglong 1.4-specific behavior from its original PPO tracking task."""

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat, quat_apply_inverse, quat_from_euler_xyz, quat_inv, quat_mul,
)


class Linglong14MotionCommand(MotionCommand):
  """Append a static 3 s tail and preserve reset/reference/history timing."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.real_motion_frames = self.motion.time_step_total
    with np.load(cfg.motion_file) as source:
      fps = float(source["fps"].reshape(-1)[0])
    if abs(fps * env.step_dt - 1.) > 1e-6:
      raise ValueError("motion sampling rate must match the control period")
    count = round(cfg.terminal_hold_time_s * fps)
    if count:
      for name in ("joint_pos", "joint_vel", "_body_pos_w", "_body_quat_w", "_body_lin_vel_w", "_body_ang_vel_w"):
        data = getattr(self.motion, name)
        tail = data[-1:].expand(count, *data.shape[1:]).clone()
        if "vel" in name:
          tail.zero_()
        setattr(self.motion, name, torch.cat([data, tail], dim=0))
      for public, private in (("body_pos_w", "_body_pos_w"), ("body_quat_w", "_body_quat_w"),
                              ("body_lin_vel_w", "_body_lin_vel_w"), ("body_ang_vel_w", "_body_ang_vel_w")):
        setattr(self.motion, public, getattr(self.motion, private)[:, self.body_indexes])
      self.motion.time_step_total += count
      self.bin_count = int(self.motion.time_step_total // fps) + 1
      self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
      self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
    self._previous_time_steps = self.time_steps.clone()
    self.switch_steps_since = torch.full_like(self.time_steps, 1_000_000)

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    self._previous_time_steps[env_ids] = self.time_steps[env_ids]
    self.switch_steps_since[env_ids] = 1_000_000

  def _switch_clips(self, env_ids):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      raise ValueError("sampling_mode must be uniform or start")
    self._previous_time_steps[env_ids] = self.time_steps[env_ids]
    # Source mid-episode switches use exact reference states, without reset noise.
    self._write_reference_state_to_sim(env_ids, self.body_pos_w[env_ids, 0].clone(),
      self.body_quat_w[env_ids, 0].clone(), self.body_lin_vel_w[env_ids, 0].clone(),
      self.body_ang_vel_w[env_ids, 0].clone(), self.joint_pos[env_ids].clone(), self.joint_vel[env_ids].clone())
    self._env.action_manager.reset(env_ids)
    self._env.observation_manager.reset(env_ids)
    self.switch_steps_since[env_ids] = 0
    self._pending_forward = True

  def _update_command(self, env_ids=None):
    # Explicit and automatic resets retain the selected first target frame.
    if env_ids is None:
      active = ~self._env.reset_buf
      self.switch_steps_since[active] = (self.switch_steps_since[active] + 1).clamp_max(1_000_000)
      self._previous_time_steps[active] = self.time_steps[active]
      self.time_steps[active] += 1
      wraps = (self.time_steps >= self.motion.time_step_total).nonzero().squeeze(-1)
      if len(wraps):
        self._switch_clips(wraps)
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()
    self.update_relative_body_poses()


@dataclass(kw_only=True)
class Linglong14MotionCommandCfg(MotionCommandCfg):
  terminal_hold_time_s: float = 3.
  switch_grace_steps: int = 5
  switch_threshold_ramp_steps: int = 20

  def build(self, env):
    return Linglong14MotionCommand(self, env)


def reference_target_rate(env, command_name="motion"):
  """Keep the source task's unscaled residual-plus-reference rate penalty."""
  cmd = env.command_manager.get_term(command_name)
  previous = cmd.motion.joint_pos[cmd._previous_time_steps]
  previous = previous * (cmd.time_steps > 0).unsqueeze(-1)
  return ((cmd.joint_pos + env.action_manager.action - previous - env.action_manager.prev_action)**2).sum(-1)


def zero_base_xy_velocity(env, command_name="motion"):
  return env.command_manager.get_term(command_name).robot_anchor_lin_vel_w[:, :2].square().sum(-1)


def knee_separation(env, command_name="motion", std=.06):
  c = env.command_manager.get_term(command_name)
  c.update_relative_body_poses()
  left, right = (c.cfg.body_names.index(n) for n in ("left_knee_link", "right_knee_link"))
  target = (c.body_pos_relative_w[:, left, :2] - c.body_pos_relative_w[:, right, :2]).norm(dim=-1)
  actual = (c.robot_body_pos_w[:, left, :2] - c.robot_body_pos_w[:, right, :2]).norm(dim=-1)
  return torch.exp(-(target-actual).clamp_min(0).square() / std**2)


def contact_penalty(env, sensor_name, threshold=1.):
  data = env.scene[sensor_name].data
  forces = data.force_history
  if forces is not None:
    magnitude = forces.norm(dim=-1).max(dim=-1).values
  else:
    magnitude = data.force.norm(dim=-1)
  return (magnitude > threshold).float().sum(-1)


def illegal_contact(env, sensor_name="illegal_contact", threshold=1.):
  command = env.command_manager.get_term("motion")
  return (contact_penalty(env, sensor_name, threshold) > 0) & (command.switch_steps_since >= command.cfg.switch_grace_steps)


def _switch_threshold(command, ordinary, initial):
  progress = ((command.switch_steps_since.float() - command.cfg.switch_grace_steps)
              / max(command.cfg.switch_threshold_ramp_steps, 1)).clamp(0, 1)
  return initial + (ordinary - initial) * progress


def anchor_height_termination(env, command_name="motion"):
  c = env.command_manager.get_term(command_name)
  error = (c.anchor_pos_w[:, 2] - c.robot_anchor_pos_w[:, 2]).abs()
  return (error > _switch_threshold(c, .25, .35)) & (c.switch_steps_since >= c.cfg.switch_grace_steps)


def anchor_tilt_termination(env, command_name="motion"):
  c = env.command_manager.get_term(command_name)
  gravity = c.robot.data.gravity_vec_w
  error = (quat_apply_inverse(c.anchor_quat_w, gravity)[:, 2] - quat_apply_inverse(c.robot_anchor_quat_w, gravity)[:, 2]).abs()
  return (error > _switch_threshold(c, .8, 1.)) & (c.switch_steps_since >= c.cfg.switch_grace_steps)


def body_height_termination(env, body_names, command_name="motion"):
  c = env.command_manager.get_term(command_name)
  c.update_relative_body_poses()
  ids = [c.cfg.body_names.index(n) for n in body_names]
  error = (c.body_pos_relative_w[:, ids, 2] - c.robot_body_pos_w[:, ids, 2]).abs().max(dim=-1).values
  return (error > _switch_threshold(c, .25, .35)) & (c.switch_steps_since >= c.cfg.switch_grace_steps)


def robot_base_height(env):
  return env.scene["robot"].data.root_link_pos_w[:, 2:3]


def critic_motion_command(env, command_name="motion"):
  c = env.command_manager.get_term(command_name)
  return torch.cat([quat_apply_inverse(c.anchor_quat_w, c.anchor_lin_vel_w), c.anchor_pos_w[:, 2:3],
                    quat_apply_inverse(c.anchor_quat_w, c.anchor_ang_vel_w),
                    matrix_from_quat(c.anchor_quat_w)[..., :2].reshape(env.num_envs, 6),
                    c.joint_pos, c.joint_vel], dim=-1)


def anchor_ori_without_yaw(env, command_name="motion"):
  c = env.command_manager.get_term(command_name)
  q = quat_mul(quat_inv(c.robot_anchor_quat_w), c.anchor_quat_w)
  w, x, y, z = q.unbind(-1)
  roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
  pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
  return matrix_from_quat(quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll)))[..., :2].reshape(env.num_envs, 6)


def current_relative_body_position_reward(env, command_name, std, body_names=None):
  """Refresh target anchoring before a reward, as in the source task."""
  from mjlab.tasks.tracking.mdp.rewards import motion_relative_body_position_error_exp
  env.command_manager.get_term(command_name).update_relative_body_poses()
  return motion_relative_body_position_error_exp(env, command_name, std, body_names)


def current_relative_body_orientation_reward(env, command_name, std, body_names=None):
  from mjlab.tasks.tracking.mdp.rewards import motion_relative_body_orientation_error_exp
  env.command_manager.get_term(command_name).update_relative_body_poses()
  return motion_relative_body_orientation_error_exp(env, command_name, std, body_names)

