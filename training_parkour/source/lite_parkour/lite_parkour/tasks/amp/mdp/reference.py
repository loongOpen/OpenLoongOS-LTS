"""Reference-motion demonstration observations for the AMP discriminator."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from lite_parkour.envs import ManagerBasedAnimationEnv
    from lite_parkour.managers import AnimationTerm


def key_body_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=MISSING, preserve_order=True),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    key_body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # shape: (num_envs, M, 3)
    root_pos_w = robot.data.root_pos_w  # shape: (num_envs, 3).
    root_quat = robot.data.root_quat_w.torch  # v3: ProxyArray -> Tensor; shape (num_envs, 4)

    num_key_bodies = key_body_pos_w.shape[1]
    num_envs = root_pos_w.shape[0]

    key_body_pos_b = math_utils.quat_apply_inverse(
        root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1),
        key_body_pos_w - root_pos_w.unsqueeze(1).expand(-1, num_key_bodies, -1),
    )

    return key_body_pos_b.reshape(num_envs, -1)


def ref_root_ang_vel_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_ang_vel_w = animation_term.get_root_ang_vel_w()  # shape: (num_envs, num_steps, 3)
    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_root_ang_vel = math_utils.quat_apply_inverse(ref_root_quat, ref_root_ang_vel_w)

    if flatten_steps_dim:
        return ref_root_ang_vel.reshape(num_envs, -1)
    else:
        return ref_root_ang_vel


def ref_joint_pos(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_dof_pos = animation_term.get_dof_pos()  # shape: (num_envs, num_steps, num_dofs)

    if flatten_steps_dim:
        return ref_dof_pos.reshape(env.num_envs, -1)
    else:
        return ref_dof_pos


def ref_joint_vel(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_dof_vel = animation_term.get_dof_vel()  # shape: (num_envs, num_steps, num_dofs)

    if flatten_steps_dim:
        return ref_dof_vel.reshape(env.num_envs, -1)
    else:
        return ref_dof_vel
