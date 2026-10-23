"""Load 20-DoF LingLong motion JSON (77-D frames) as 53-D AMP expert transitions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


class AmpMotionLoader:
  """Expert dataset for AMP.

  Raw frame (77):
      root_pos3 | root_rot_xyzw4 | joint_q20 | end12 | lin3 | ang3 | joint_dq20 | end_vel12
      end12 = toeL3 + toeR3 + handL3 + handR3

  AMP observation (53):
      joint_q20 | joint_dq20 | base_lin3 | base_ang3 | foot6 | root_z1
  """

  RAW_FRAME_SIZE = 77
  RAW_POS_Z_IDX = 2
  RAW_JOINT_POS_START = 7
  RAW_JOINT_POS_END = 27
  RAW_FOOT_POS_START = 27
  RAW_FOOT_POS_END = 33
  RAW_LIN_VEL_START = 39
  RAW_LIN_VEL_END = 42
  RAW_ANG_VEL_START = 42
  RAW_ANG_VEL_END = 45
  RAW_JOINT_VEL_START = 45
  RAW_JOINT_VEL_END = 65

  JOINT_POS_SIZE = 20
  JOINT_VEL_SIZE = 20
  LINEAR_VEL_SIZE = 3
  ANGULAR_VEL_SIZE = 3
  FOOT_POS_SIZE = 6
  Z_POS_SIZE = 1
  TOTAL_DATA_SIZE = 53

  def __init__(
    self,
    device: str | torch.device,
    time_between_frames: float,
    motion_files: Sequence[str],
    preload_transitions: bool = True,
    num_preload_transitions: int = 200_000,
  ) -> None:
    self.device = torch.device(device)
    self.time_between_frames = float(time_between_frames)

    self.trajectories: list[torch.Tensor] = []
    self.trajectory_names: list[str] = []
    self.trajectory_idxs: list[int] = []
    self.trajectory_lens: list[float] = []
    self.trajectory_weights: list[float] = []
    self.trajectory_frame_durations: list[float] = []
    self.trajectory_num_frames: list[int] = []

    if not motion_files:
      raise ValueError("AmpMotionLoader requires at least one motion file")

    for i, motion_file in enumerate(motion_files):
      path = Path(motion_file)
      if not path.exists():
        raise FileNotFoundError(f"AMP motion file not found: {path}")
      with path.open() as f:
        motion_json = json.load(f)
      motion_data = np.asarray(motion_json["Frames"], dtype=np.float64)
      if motion_data.ndim != 2 or motion_data.shape[1] != self.RAW_FRAME_SIZE:
        raise ValueError(
          f"{path}: expect raw frame dim={self.RAW_FRAME_SIZE}, got {motion_data.shape}"
        )

      target_data = np.concatenate(
        [
          motion_data[:, self.RAW_JOINT_POS_START : self.RAW_JOINT_POS_END],
          motion_data[:, self.RAW_JOINT_VEL_START : self.RAW_JOINT_VEL_END],
          motion_data[:, self.RAW_LIN_VEL_START : self.RAW_LIN_VEL_END],
          motion_data[:, self.RAW_ANG_VEL_START : self.RAW_ANG_VEL_END],
          motion_data[:, self.RAW_FOOT_POS_START : self.RAW_FOOT_POS_END],
          motion_data[:, self.RAW_POS_Z_IDX : self.RAW_POS_Z_IDX + 1],
        ],
        axis=1,
      )
      if target_data.shape[1] != self.TOTAL_DATA_SIZE:
        raise RuntimeError(
          f"AMP obs dim error: expect {self.TOTAL_DATA_SIZE}, got {target_data.shape[1]}"
        )

      self.trajectories.append(
        torch.tensor(target_data, dtype=torch.float32, device=self.device)
      )
      self.trajectory_names.append(path.stem)
      self.trajectory_idxs.append(i)
      self.trajectory_weights.append(float(motion_json.get("MotionWeight", 1.0)))
      frame_duration = float(motion_json["FrameDuration"])
      self.trajectory_frame_durations.append(frame_duration)
      traj_len = (motion_data.shape[0] - 1) * frame_duration
      self.trajectory_lens.append(traj_len)
      self.trajectory_num_frames.append(int(motion_data.shape[0]))
      print(
        f"[AMP] loaded {path.name}: {traj_len:.2f}s, "
        f"{motion_data.shape[0]} frames, AMP dim {target_data.shape[1]}"
      )

    weights = np.asarray(self.trajectory_weights, dtype=np.float64)
    self.trajectory_weights = weights / weights.sum()
    self.trajectory_frame_durations = np.asarray(
      self.trajectory_frame_durations, dtype=np.float64
    )
    self.trajectory_lens = np.asarray(self.trajectory_lens, dtype=np.float64)
    self.trajectory_num_frames = np.asarray(self.trajectory_num_frames, dtype=np.int64)
    self.trajectory_idxs = np.asarray(self.trajectory_idxs, dtype=np.int64)

    self.preload_transitions = preload_transitions
    if preload_transitions:
      print(f"[AMP] preloading {num_preload_transitions} transitions")
      traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
      times = self.trajectory_time_sample_batch(traj_idxs)
      self.preloaded_s = self.get_frame_at_time_batch(traj_idxs, times)
      self.preloaded_s_next = self.get_frame_at_time_batch(
        traj_idxs, times + self.time_between_frames
      )

  @property
  def observation_dim(self) -> int:
    return self.TOTAL_DATA_SIZE

  def weighted_traj_idx_sample_batch(self, size: int) -> np.ndarray:
    return np.random.choice(
      self.trajectory_idxs, size=size, p=self.trajectory_weights, replace=True
    )

  def trajectory_time_sample_batch(self, traj_idxs: np.ndarray) -> np.ndarray:
    subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
    time_samples = (
      self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
    )
    return np.maximum(np.zeros_like(time_samples), time_samples)

  def get_frame_at_time_batch(
    self, traj_idxs: np.ndarray, times: np.ndarray
  ) -> torch.Tensor:
    p = times / np.maximum(self.trajectory_lens[traj_idxs], 1e-8)
    n = self.trajectory_num_frames[traj_idxs]
    idx_low = np.floor(p * n).astype(np.int64)
    idx_high = np.ceil(p * n).astype(np.int64)
    idx_low = np.clip(idx_low, 0, n - 1)
    idx_high = np.clip(idx_high, 0, n - 1)

    batch_size = len(traj_idxs)
    starts = torch.zeros(batch_size, self.TOTAL_DATA_SIZE, device=self.device)
    ends = torch.zeros(batch_size, self.TOTAL_DATA_SIZE, device=self.device)
    for traj_idx in set(traj_idxs.tolist()):
      trajectory = self.trajectories[int(traj_idx)]
      mask = traj_idxs == traj_idx
      starts[mask] = trajectory[idx_low[mask]]
      ends[mask] = trajectory[idx_high[mask]]
    blend = torch.as_tensor(
      p * n - idx_low, device=self.device, dtype=torch.float32
    ).unsqueeze(-1)
    return (1.0 - blend) * starts + blend * ends

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    for _ in range(num_mini_batch):
      if self.preload_transitions:
        idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)
        yield self.preloaded_s[idxs], self.preloaded_s_next[idxs]
      else:
        traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
        times = self.trajectory_time_sample_batch(traj_idxs)
        yield (
          self.get_frame_at_time_batch(traj_idxs, times),
          self.get_frame_at_time_batch(traj_idxs, times + self.time_between_frames),
        )
