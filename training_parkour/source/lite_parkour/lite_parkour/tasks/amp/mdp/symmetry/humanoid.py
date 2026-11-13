"""Left-right symmetry of observations and actions for the Lite AMP-depth task.

Each batch is augmented with its left-right mirror. Term layouts are read from the observation manager and joint / key-body mirrors are built by
name, so histories, joint counts and key bodies follow the task configuration. Every term needs
an explicit rule; an unknown term raises instead of silently passing through unmirrored.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states", "joint_mirror", "mirror_flat_group"]

# Per-frame signs under the reflection y -> -y.
_VECTOR_SIGNS = {
    "base_lin_vel": (1.0, -1.0, 1.0),  # polar vector
    "base_ang_vel": (-1.0, 1.0, -1.0),  # axial vector
    "velocity_commands": (1.0, -1.0, -1.0),  # (vx, vy, yaw rate)
    "root_local_rot_tan_norm": (1.0, -1.0, 1.0, 1.0, -1.0, 1.0),  # tangent and normal columns
}
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")


def _partner(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name


def joint_mirror(joint_names) -> tuple[list[int], list[float]]:
    """``mirrored[k] = signs[k] * data[indices[k]]`` for a block of joint values.

    Left/right joints swap; roll and yaw angles change sign, pitch angles do not. This holds for
    Lite because paired joints share axis directions and mirrored origins in the URDF.
    """
    name_to_index = {name: i for i, name in enumerate(joint_names)}
    indices, signs = [], []
    for name in joint_names:
        if _partner(name) not in name_to_index:
            raise ValueError(f"Symmetry: no mirror partner for joint '{name}'.")
        indices.append(name_to_index[_partner(name)])
        signs.append(-1.0 if ("_roll_joint" in name or "_yaw_joint" in name) else 1.0)
    return indices, signs


def _key_body_mirror(body_names) -> tuple[list[int], list[float]]:
    """Mirror for concatenated body-frame positions: swap left/right bodies and negate y."""
    name_to_index = {name: i for i, name in enumerate(body_names)}
    indices, signs = [], []
    for name in body_names:
        if _partner(name) not in name_to_index:
            raise ValueError(f"Symmetry: no mirror partner for key body '{name}'.")
        source = name_to_index[_partner(name)]
        indices += [3 * source, 3 * source + 1, 3 * source + 2]
        signs += [1.0, -1.0, 1.0]
    return indices, signs


def mirror_flat_group(obs: torch.Tensor, layout, joint_names, key_body_names=()) -> torch.Tensor:
    """Mirror a concatenated observation group.

    Args:
        obs: ``(batch, dim)`` group tensor.
        layout: ``[(term_name, term_dim), ...]`` in concatenation order; histories are flattened
            frame by frame, so each term is a repetition of one per-frame block.
        joint_names: Joint order of joint-valued terms.
        key_body_names: Body order of ``key_body_pos_b``.
    """
    frame_rules = {name: (list(range(len(signs))), list(signs)) for name, signs in _VECTOR_SIGNS.items()}
    frame_rules.update({name: joint_mirror(joint_names) for name in _JOINT_TERMS})
    if key_body_names:
        frame_rules["key_body_pos_b"] = _key_body_mirror(key_body_names)

    index, sign, offset = [], [], 0
    for name, dim in layout:
        if name not in frame_rules:
            raise ValueError(f"Symmetry: no mirror rule for observation term '{name}'.")
        frame_index, frame_sign = frame_rules[name]
        width = len(frame_index)
        if dim % width:
            raise ValueError(f"Symmetry: term '{name}' has {dim} elements, not a multiple of {width}.")
        for _ in range(dim // width):
            index += [offset + i for i in frame_index]
            sign += frame_sign
            offset += width
    if offset != obs.shape[-1]:
        raise ValueError(f"Symmetry: layout covers {offset} elements, group has {obs.shape[-1]}.")
    index_t = torch.tensor(index, dtype=torch.long, device=obs.device)
    sign_t = torch.tensor(sign, dtype=obs.dtype, device=obs.device)
    return obs.index_select(-1, index_t) * sign_t


def _layout(env: ManagerBasedRLEnv, group: str):
    manager = env.observation_manager
    return [(name, math.prod(shape)) for name, shape in zip(manager.active_terms[group], manager.group_obs_term_dim[group])]


def _key_body_names(env: ManagerBasedRLEnv):
    term = getattr(env.cfg.observations.critic, "key_body_pos_b", None)
    return tuple(term.params["asset_cfg"].body_names) if term is not None else ()


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return the batch followed by its left-right mirror, for observations and/or actions."""
    env = env.unwrapped
    joint_names = tuple(env.cfg.actions.joint_pos.joint_names)

    obs_aug = None
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)  # first half keeps the original samples
        for group in ("policy", "critic"):
            if group in obs.keys():
                obs_aug[group][batch_size:] = mirror_flat_group(
                    obs[group], _layout(env, group), joint_names, _key_body_names(env)
                )
        if "depth" in obs.keys():
            # (batch, frames, height, width): the camera sits on the sagittal plane and the crop is
            # left-right symmetric, so mirroring the scene flips image columns.
            obs_aug["depth"][batch_size:] = obs["depth"].flip(dims=[-1])

    actions_aug = None
    if actions is not None:
        indices, signs = joint_mirror(joint_names)
        mirrored = actions[:, indices] * torch.tensor(signs, dtype=actions.dtype, device=actions.device)
        actions_aug = torch.cat((actions, mirrored), dim=0)

    return obs_aug, actions_aug
