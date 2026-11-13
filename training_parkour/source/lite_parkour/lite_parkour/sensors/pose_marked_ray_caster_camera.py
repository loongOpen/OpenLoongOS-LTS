"""Ray-cast depth camera whose debug visualization also marks the camera pose.

The stock debug view only draws ray hits, which shows where the camera looks but
not where it sits or how it is rolled. With ``debug_vis`` enabled this sensor
additionally draws, for every environment:

* a dark box at the optical centre;
* a red arrow along the optical axis (+X in the ``world`` camera convention);
* a blue stub along the image up direction (+Z);
* a yellow wire frustum through the four corner pixel rays.

This module imports USD-backed marker code, so it must only be loaded after the
simulation app starts. Reference it from a sensor config with a string
``class_type``, e.g. ``ResolvableString("lite_parkour.sensors.pose_marked_ray_caster_camera:
PoseMarkedMultiMeshRayCasterCamera")``.
"""

from __future__ import annotations

import importlib

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera import MultiMeshRayCasterCamera

AXIS_LENGTH = 0.25
AXIS_TIP_LENGTH = 0.04
UP_LENGTH = 0.08
FRUSTUM_LENGTH = 0.35

# Unit-length line prototypes are scaled along X per instance. Primitive shapes
# keep the markers independent of Nucleus assets.
POSE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/DepthCameraPose",
    markers={
        "body": sim_utils.CuboidCfg(
            size=(0.03, 0.09, 0.025),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        "axis_line": sim_utils.CylinderCfg(
            radius=0.006, height=1.0, axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "axis_tip": sim_utils.ConeCfg(
            radius=0.015, height=AXIS_TIP_LENGTH, axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "up_line": sim_utils.CylinderCfg(
            radius=0.004, height=1.0, axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.3, 1.0)),
        ),
        "frustum_line": sim_utils.CylinderCfg(
            radius=0.002, height=1.0, axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
        ),
    },
)
_BODY, _AXIS_LINE, _AXIS_TIP, _UP_LINE, _FRUSTUM_LINE = range(5)


def _quat_x_to(direction: torch.Tensor) -> torch.Tensor:
    """Shortest rotation (x, y, z, w) taking +X onto unit ``direction``; not valid for -X."""
    q = torch.stack(
        (torch.zeros_like(direction[:, 0]), -direction[:, 2], direction[:, 1], 1.0 + direction[:, 0]), dim=-1
    )
    return math_utils.normalize(q)


class _CameraPoseMarkerMixin:
    """Adds the pose marker on top of the backend camera's ray-hit visualization."""

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "pose_visualizer"):
                self.pose_visualizer = VisualizationMarkers(POSE_MARKER_CFG)
            self.pose_visualizer.set_visibility(True)
        elif hasattr(self, "pose_visualizer"):
            self.pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if not self.is_initialized or not hasattr(self, "pose_visualizer"):
            return
        translations, orientations, scales, indices = self.pose_marker_transforms()
        self.pose_visualizer.visualize(translations, orientations, scales, indices)

    def _pose_marker_layout(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Marker offsets in the camera frame, built once from the corner pixel rays."""
        if hasattr(self, "_pose_marker_local"):
            return self._pose_marker_local
        width, height = self.cfg.pattern_cfg.width, self.cfg.pattern_cfg.height
        directions = self.ray_directions.torch if hasattr(self.ray_directions, "torch") else self.ray_directions
        # Rays are stored row-major; walk the corners around the image border.
        corners = directions[0, [0, width - 1, height * width - 1, (height - 1) * width]].float()
        corners = math_utils.normalize(corners)
        far = corners * FRUSTUM_LENGTH

        device = corners.device
        x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        # Segments as start + vector: optical axis, up stub, 4 frustum edges from
        # the optical centre, then the 4 rim edges joining consecutive far corners.
        rim = far.roll(-1, dims=0) - far
        starts = torch.cat((torch.zeros(6, 3, device=device), far))
        vectors = torch.cat((x_axis * AXIS_LENGTH, z_axis * UP_LENGTH, far, rim))
        lengths = vectors.norm(dim=-1)
        line_pos = starts + 0.5 * vectors
        line_quat = _quat_x_to(vectors / lengths.unsqueeze(-1))
        line_scale = torch.stack((lengths, torch.ones_like(lengths), torch.ones_like(lengths)), dim=-1)
        line_index = torch.tensor([_AXIS_LINE, _UP_LINE] + [_FRUSTUM_LINE] * 8, device=device)

        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
        ones = torch.ones(1, 3, device=device)
        tip_pos = x_axis * (AXIS_LENGTH + 0.5 * AXIS_TIP_LENGTH)
        self._pose_marker_local = (
            torch.cat((torch.zeros(1, 3, device=device), tip_pos, line_pos)),
            torch.cat((identity, identity, line_quat)),
            torch.cat((ones, ones, line_scale)),
            torch.cat((torch.tensor([_BODY, _AXIS_TIP], device=device), line_index)),
        )
        return self._pose_marker_local

    def pose_marker_transforms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-frame marker translations, orientations (x, y, z, w), scales and prototype indices."""
        local_pos, local_quat, local_scale, local_index = self._pose_marker_layout()
        cam_pos = self._data.pos_w.torch
        cam_quat = self._data.quat_w_world.torch
        num_envs, num_markers = cam_pos.shape[0], local_pos.shape[0]
        quat = cam_quat.unsqueeze(1).expand(num_envs, num_markers, 4)
        translations = cam_pos.unsqueeze(1) + math_utils.quat_apply(quat, local_pos.expand(num_envs, -1, -1))
        orientations = math_utils.quat_mul(quat.reshape(-1, 4), local_quat.repeat(num_envs, 1))
        return (
            translations.reshape(-1, 3),
            orientations,
            local_scale.repeat(num_envs, 1),
            local_index.repeat(num_envs),
        )


class PoseMarkedMultiMeshRayCasterCamera:
    """:class:`MultiMeshRayCasterCamera` for the active physics backend, with the pose marker mixed in.

    The stock class is a backend factory whose ``__new__`` returns the backend
    implementation, so subclassing it directly would drop any override. This
    resolves that implementation the same way and extends it instead.
    """

    _classes: dict[type, type] = {}

    def __new__(cls, cfg):
        factory = MultiMeshRayCasterCamera
        backend = factory._get_backend(cfg)
        if backend not in factory._registry:
            module = importlib.import_module(factory._get_module_name(backend))
            class_name = getattr(factory, "_backend_class_names", {}).get(backend, factory.__name__)
            factory.register(backend, getattr(module, class_name))
        base = factory._registry[backend]
        if base not in cls._classes:
            cls._classes[base] = type(
                f"PoseMarked{base.__name__}", (_CameraPoseMarkerMixin, base), {"__module__": __name__}
            )
        return cls._classes[base](cfg)
