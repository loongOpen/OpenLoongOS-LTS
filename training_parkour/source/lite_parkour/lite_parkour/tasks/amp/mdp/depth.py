"""Depth-camera observation terms for AMP locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg


def pinhole_camera_pattern_analytic(cfg, intrinsic_matrices: torch.Tensor, device: str):
    """Generate pinhole rays without a batched 3x3 matrix multiplication.

    This is mathematically equivalent to Isaac Lab's ``pinhole_camera_pattern`` for
    the standard, zero-skew intrinsic matrices produced by a
    :class:`PinholeCameraPatternCfg`. Avoiding the tiny batched GEMM also works
    around a CUBLAS ``INVALID_VALUE`` error seen with the PyTorch/CUDA build
    shipped in some Isaac Lab v3 environments.
    """
    pixel_x, pixel_y = torch.meshgrid(
        torch.arange(cfg.width, dtype=torch.float32, device=device) + 0.5,
        torch.arange(cfg.height, dtype=torch.float32, device=device) + 0.5,
        indexing="xy",
    )
    pixel_x = pixel_x.flatten().unsqueeze(0)
    pixel_y = pixel_y.flatten().unsqueeze(0)

    focal_x = intrinsic_matrices[:, 0, 0].unsqueeze(1)
    focal_y = intrinsic_matrices[:, 1, 1].unsqueeze(1)
    center_x = intrinsic_matrices[:, 0, 2].unsqueeze(1)
    center_y = intrinsic_matrices[:, 1, 2].unsqueeze(1)

    image_x = (pixel_x - center_x) / focal_x
    image_y = (pixel_y - center_y) / focal_y
    ray_directions = torch.stack((torch.ones_like(image_x), -image_x, -image_y), dim=-1)
    ray_directions = F.normalize(ray_directions, dim=-1)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions


class DepthImageHistory(ManagerTermBase):
    """Pre-process and temporally subsample depth images from a camera sensor.

    Ported from the perception path of InstinctLab's G1 parkour task.
    Crop the raw depth image, apply a Gaussian blur, normalize metric depth, keep
    a long per-environment history, and return a sparse sequence ending at the
    current frame without adding sensor delay. The output layout is ``(N, frames, height, width)`` so
    the frame dimension can be consumed directly as Conv2d input channels.
    """

    def __init__(self, cfg: ObservationTermCfg, env):
        super().__init__(cfg, env)

        self.sensor_cfg: SceneEntityCfg = cfg.params.get("sensor_cfg", SceneEntityCfg("depth_camera"))
        self.data_type: str = cfg.params.get("data_type", "distance_to_image_plane")
        self.crop_region: tuple[int, int, int, int] = tuple(cfg.params.get("crop_region", (0, 0, 0, 0)))
        self.blur_kernel_size: int = cfg.params.get("blur_kernel_size", 3)
        self.blur_sigma: float = cfg.params.get("blur_sigma", 1.0)
        self.depth_range: tuple[float, float] = tuple(cfg.params.get("depth_range", (0.0, 2.5)))
        self.history_length: int = cfg.params.get("history_length", 37)
        self.history_skip_frames: int = cfg.params.get("history_skip_frames", 5)
        self.num_output_frames: int = cfg.params.get("num_output_frames", 8)

        if self.blur_kernel_size < 1 or self.blur_kernel_size % 2 == 0:
            raise ValueError(f"blur_kernel_size must be a positive odd number, got {self.blur_kernel_size}.")
        if self.blur_sigma <= 0.0:
            raise ValueError(f"blur_sigma must be positive, got {self.blur_sigma}.")
        if self.depth_range[1] <= self.depth_range[0]:
            raise ValueError(f"Invalid depth_range: {self.depth_range}.")
        if self.history_skip_frames < 1 or self.num_output_frames < 1:
            raise ValueError("history_skip_frames and num_output_frames must be at least one.")
        frames_needed = (self.num_output_frames - 1) * self.history_skip_frames + 1
        if frames_needed > self.history_length:
            raise ValueError(
                "Depth history is too short: need at least "
                f"{frames_needed} frames, got {self.history_length}."
            )

        self._history: torch.Tensor | None = None
        self._history_pointer = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._num_pushes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._batch_indices = torch.arange(env.num_envs, device=env.device)

        coords = torch.arange(self.blur_kernel_size, device=env.device, dtype=torch.float32)
        coords -= (self.blur_kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-(coords.square()) / (2.0 * self.blur_sigma**2))
        kernel_1d /= kernel_1d.sum()
        self._blur_kernel = torch.outer(kernel_1d, kernel_1d).view(1, 1, self.blur_kernel_size, self.blur_kernel_size)

        # Oldest-to-newest output offsets, expressed as lags from the current frame.
        self._frame_lags = torch.flip(
            torch.arange(
                0,
                self.num_output_frames * self.history_skip_frames,
                self.history_skip_frames,
                device=env.device,
            ),
            dims=(0,),
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids_t = self._batch_indices
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self._batch_indices.device)

        self._history_pointer[env_ids_t] = -1
        self._num_pushes[env_ids_t] = 0

    def _read_and_process(self, env) -> torch.Tensor:
        sensor = env.scene.sensors[self.sensor_cfg.name]
        image_buffer = sensor.data.output[self.data_type]
        images = image_buffer.torch if hasattr(image_buffer, "torch") else image_buffer
        images = images.clone()

        if images.ndim != 4 or images.shape[-1] != 1:
            raise ValueError(
                f"Expected depth image shape (N, H, W, 1), got {tuple(images.shape)} for '{self.data_type}'."
            )

        crop_up, crop_down, crop_left, crop_right = self.crop_region
        end_h = images.shape[1] - crop_down if crop_down > 0 else images.shape[1]
        end_w = images.shape[2] - crop_right if crop_right > 0 else images.shape[2]
        if crop_up >= end_h or crop_left >= end_w:
            raise ValueError(f"crop_region {self.crop_region} removes the entire image {tuple(images.shape[1:3])}.")
        images = images[:, crop_up:end_h, crop_left:end_w, :]

        # NHWC -> NCHW. Reflection padding matches torchvision's GaussianBlur behavior.
        images = images.permute(0, 3, 1, 2)
        pad = self.blur_kernel_size // 2
        if pad > 0:
            images = F.pad(images, (pad, pad, pad, pad), mode="reflect")
            images = F.conv2d(images, self._blur_kernel)

        depth_min, depth_max = self.depth_range
        images = torch.nan_to_num(images, nan=depth_max, posinf=depth_max, neginf=depth_min)
        images = images.clamp(depth_min, depth_max)
        images = (images - depth_min) / (depth_max - depth_min)
        return images.squeeze(1)

    def _append(self, frame: torch.Tensor) -> None:
        if self._history is None:
            self._history = torch.empty(
                (frame.shape[0], self.history_length, *frame.shape[1:]),
                dtype=frame.dtype,
                device=frame.device,
            )

        first_push = self._num_pushes == 0
        self._history_pointer = (self._history_pointer + 1) % self.history_length
        self._history[self._batch_indices, self._history_pointer] = frame
        if torch.any(first_push):
            self._history[first_push] = frame[first_push].unsqueeze(1)
        self._num_pushes += 1

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
        data_type: str = "distance_to_image_plane",
        crop_region: tuple[int, int, int, int] = (0, 0, 0, 0),
        blur_kernel_size: int = 3,
        blur_sigma: float = 1.0,
        depth_range: tuple[float, float] = (0.0, 2.5),
        history_length: int = 37,
        history_skip_frames: int = 5,
        num_output_frames: int = 8,
    ) -> torch.Tensor:
        del sensor_cfg, data_type, crop_region, blur_kernel_size, blur_sigma
        del depth_range, history_length, history_skip_frames, num_output_frames

        frame = self._read_and_process(env)
        self._append(frame)
        assert self._history is not None

        lags = self._frame_lags.unsqueeze(0)
        frame_indices = (self._history_pointer.unsqueeze(1) - lags) % self.history_length
        return self._history[self._batch_indices.unsqueeze(1), frame_indices].clone()
