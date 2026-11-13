# Adapted from:
#   instinctlab (CC BY-NC 4.0)
#     https://github.com/project-instinct/instinctlab
#   Isaac Lab (BSD-3-Clause)
#     https://github.com/isaac-sim/IsaacLab

from __future__ import annotations

import numpy as np
import os
import scipy.spatial.transform as tf
import torch
import trimesh
import yaml
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

from ..height_field.hf_terrains import generate_perlin_noise
from .utils import crop_terrain_mesh_aabb, generate_wall

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


@generate_wall
def motion_matched_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a motion-matched terrain based on the difficulty level and configuration.

    Args:
        difficulty (float): The difficulty level for the terrain.
        cfg (mesh_terrains_cfg.MotionMatchedTerrainCfg): Configuration for the motion-matched terrain.

    Returns:
        tuple: A tuple containing a list of trimesh objects and an array of poses.
    """
    # Load the YAML file containing terrain and motion data
    with open(cfg.metadata_yaml) as file:
        data = yaml.safe_load(file)

    # Extract terrains and motions from the YAML data
    terrains = data["terrains"]  # order matters

    terrain_idx = int(np.clip(difficulty * len(terrains), 0, len(terrains) - 1))
    selected_terrain = terrains[terrain_idx]

    terrain_file = selected_terrain["terrain_file"]
    terrain_abspath = os.path.join(cfg.path, terrain_file)
    terrain_mesh = trimesh.load(terrain_abspath, force="mesh")

    # crop terrain mesh by cfg.size
    # This does not change the terrain origin.
    terrain_mesh = crop_terrain_mesh_aabb(
        terrain_mesh,
        x_max=cfg.size[0] / 2,
        x_min=-cfg.size[0] / 2,
        y_max=cfg.size[1] / 2,
        y_min=-cfg.size[1] / 2,
    )

    # Find border height offset w.r.t current center of this terrain mesh.
    # This is used to align the terrain mesh with the origin.
    # NOTE: Assuming the border is flat, we take the mean height of the vertices
    border_height = np.mean(
        terrain_mesh.vertices[
            np.logical_or(
                np.abs(terrain_mesh.vertices[:, 0]) > (cfg.size[0] / 2 - 0.05),
                np.abs(terrain_mesh.vertices[:, 1]) > (cfg.size[1] / 2 - 0.05),
            )
        ][:, 2]
    )
    if np.isnan(border_height):
        print(f"Warning: Terrain {terrain_file} does not have a valid border height. Using 0 as the border height.")
        border_height = 0.0

    # To follow the terrain_generator convention, we move the terrain mesh to (size[0]/2, size[1]/2, -border_height).
    move_terrain_transform = np.eye(4)
    move_terrain_transform[:2, 3] = np.array(cfg.size) / 2
    move_terrain_transform[2, 3] = -border_height
    terrain_mesh.apply_transform(move_terrain_transform)
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, -border_height])
    return terrain_mesh, origin


@generate_wall
def floating_box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.PerlinMeshFloatingBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generates a floating box terrain."""

    # resolve the terrain configuration
    # height of the floating box above the ground
    if isinstance(cfg.floating_height, (tuple, list)):
        floating_height = cfg.floating_height[1] - difficulty * (cfg.floating_height[1] - cfg.floating_height[0])
    else:
        floating_height = cfg.floating_height

    # length of the floating box
    if isinstance(cfg.box_length, (tuple, list)):
        box_length = cfg.box_length[1] - difficulty * (cfg.box_length[1] - cfg.box_length[0])
    else:
        box_length = cfg.box_length

    # height of the floating box
    if isinstance(cfg.box_height, (tuple, list)):
        box_height = np.random.uniform(*cfg.box_height)
    else:
        box_height = cfg.box_height

    # width of the floating box
    if cfg.box_width is None:
        box_width = cfg.size[0]
    else:
        box_width = cfg.box_width

    # initialize the list of meshes
    meshes_list = list()

    # extract quantities
    total_height = floating_height + box_height
    # constants for terrain generation
    terrain_height = 0.0

    # generate the box mesh
    dim = (box_width, box_length, box_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], floating_height + box_height / 2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)

    # generate the ground

    if cfg.perlin_cfg is None:
        dim = (cfg.size[0], cfg.size[1], terrain_height)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(ground_mesh)
    else:
        clean_ground_height_field = np.zeros(
            (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1), dtype=np.int16
        )
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        h, w = perlin_noise.shape
        ground_h, ground_w = clean_ground_height_field.shape
        pad_h_left = max(0, (ground_h - h) // 2)
        pad_h_right = max(0, ground_h - h - pad_h_left)
        pad_w_left = max(0, (ground_w - w) // 2)
        pad_w_right = max(0, ground_w - w - pad_w_left)
        pad_width = ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        perlin_noise = np.pad(perlin_noise, pad_width, mode="constant", constant_values=0)
        if cfg.no_perlin_at_obstacle is True:
            box_width_px = int(box_width / cfg.horizontal_scale)
            box_length_px = int(box_length / cfg.horizontal_scale)
            box_width_start_px = int((cfg.size[0] - box_width) / 2 / cfg.horizontal_scale)
            box_length_start_px = int((cfg.size[1] - box_length) / 2 / cfg.horizontal_scale)
            perlin_noise[
                box_width_start_px : box_width_start_px + box_width_px,
                box_length_start_px : box_length_start_px + box_length_px,
            ] = 0
        ground_height_field = clean_ground_height_field + perlin_noise
        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            ground_height_field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        ground_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], total_height])

    return meshes_list, origin


@generate_wall
def random_multi_box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.PerlinMeshRandomMultiBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generates a terrain containing multiple boxes with random size and orientation."""

    box_height_range = cfg.box_height_range
    box_length_range = cfg.box_length_range
    box_width_range = cfg.box_width_range

    if isinstance(cfg.box_height_mean, (tuple, list)):
        if cfg.box_height_mean[0] < box_height_range:
            raise RuntimeError("The minimum box height mean is smaller than the box height half range.")
        box_height_mean = cfg.box_height_mean[0] + difficulty * (cfg.box_height_mean[1] - cfg.box_height_mean[0])
    else:
        box_height_mean = cfg.box_height_mean
        if box_height_mean < box_height_range:
            raise RuntimeError("The minimum box height mean is smaller than the box height half range.")

    if isinstance(cfg.box_length_mean, (tuple, list)):
        if cfg.box_length_mean[0] < box_length_range:
            raise RuntimeError("The minimum box length mean is smaller than the box length half range.")
        box_length_mean = cfg.box_length_mean[0] + difficulty * (cfg.box_length_mean[1] - cfg.box_length_mean[0])
    else:
        box_length_mean = cfg.box_length_mean
        if box_length_mean < box_length_range:
            raise RuntimeError("The minimum box length mean is smaller than the box length half range.")

    if isinstance(cfg.box_width_mean, (tuple, list)):
        if cfg.box_width_mean[0] < box_width_range:
            raise RuntimeError("The minimum box width mean is smaller than the box width half range.")
        box_width_mean = cfg.box_width_mean[0] + difficulty * (cfg.box_width_mean[1] - cfg.box_width_mean[0])
    else:
        box_width_mean = cfg.box_width_mean
        if box_width_mean < box_width_range:
            raise RuntimeError("The minimum box width mean is smaller than the box width half range.")

    generation_ratio = cfg.generation_ratio

    width = cfg.size[0]
    length = cfg.size[1]

    mesh_list = []

    num_boxes = int(generation_ratio * (width * length) / (box_length_mean * box_width_mean))
    num_boxes = max(1, num_boxes)
    if cfg.perlin_cfg is None:
        dim = (cfg.size[0], cfg.size[1], 0.0)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0)
        ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        mesh_list.append(ground_mesh)
    else:
        clean_ground_height_field = np.zeros(
            (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1), dtype=np.int16
        )
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        h, w = perlin_noise.shape
        ground_h, ground_w = clean_ground_height_field.shape
        pad_h_left = max(0, (ground_h - h) // 2)
        pad_h_right = max(0, ground_h - h - pad_h_left)
        pad_w_left = max(0, (ground_w - w) // 2)
        pad_w_right = max(0, ground_w - w - pad_w_left)
        pad_width = ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        perlin_noise = np.pad(perlin_noise, pad_width, mode="constant", constant_values=0)
        ground_height_field = clean_ground_height_field + perlin_noise
        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            ground_height_field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        ground_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh_list.append(ground_mesh)

    if cfg.box_perlin_cfg is not None and cfg.no_perlin_at_obstacle is False:
        box_perlin_cfg = cfg.box_perlin_cfg
        box_perlin_cfg.horizontal_scale = (
            cfg.horizontal_scale if box_perlin_cfg.horizontal_scale is None else box_perlin_cfg.horizontal_scale
        )
        box_perlin_cfg.vertical_scale = (
            cfg.vertical_scale if box_perlin_cfg.vertical_scale is None else box_perlin_cfg.vertical_scale
        )
        box_perlin_cfg.slope_threshold = (
            cfg.slope_threshold if box_perlin_cfg.slope_threshold is None else box_perlin_cfg.slope_threshold
        )

    platform_width = cfg.platform_width

    for i in range(num_boxes):
        box_width = box_width_mean + np.random.uniform(-1, 1) * box_width_range
        box_length = box_length_mean + np.random.uniform(-1, 1) * box_length_range
        box_height = box_height_mean + np.random.uniform(-1, 1) * box_height_range
        dim = (box_width, box_length, box_height)
        x = np.random.uniform(box_width / 2, width - box_width / 2)
        y = np.random.uniform(box_length / 2, length - box_length / 2)
        if (
            x > width / 2 - platform_width / 2 - box_width / 2 and x < width / 2 + platform_width / 2 + box_width / 2
        ) and (
            y > length / 2 - platform_width / 2 - box_length / 2
            and y < length / 2 + platform_width / 2 + box_length / 2
        ):
            continue
        pos = (x, y, box_height / 2)
        theta = np.random.uniform(0, 2 * np.pi)
        translation_matrix = trimesh.transformations.translation_matrix(pos)
        rotation_matrix = trimesh.transformations.rotation_matrix(theta, (0, 0, 1))
        transform = translation_matrix @ rotation_matrix
        box_mesh = trimesh.creation.box(extents=dim)
        # top_z=box_mesh.vertices[:, 2].max()
        # top_face_mask=np.all(box_mesh.vertices[box_mesh.faces][:,:,2] == top_z, axis=1)
        # box_mesh.update_faces(~top_face_mask)
        # box_mesh.remove_unreferenced_vertices()
        box_mesh.apply_transform(transform)
        mesh_list.append(box_mesh)
        if cfg.box_perlin_cfg is not None and cfg.no_perlin_at_obstacle is False:
            box_perlin_cfg.size = (box_width, box_length)
            perlin_noise = generate_perlin_noise(
                difficulty,
                box_perlin_cfg,  # type: ignore[arg-type]
            )
            vertices, triangles = convert_height_field_to_mesh(
                perlin_noise,
                box_perlin_cfg.horizontal_scale,
                box_perlin_cfg.vertical_scale,
                box_perlin_cfg.slope_threshold,
            )
            box_noise = trimesh.Trimesh(vertices=vertices, faces=triangles)
            center_offset = (-box_width / 2, -box_length / 2, 0)
            center_translation = trimesh.transformations.translation_matrix(center_offset)
            box_noise.apply_transform(center_translation)
            noise_pos = (x, y, box_height)
            translation_matrix = trimesh.transformations.translation_matrix(noise_pos)
            transform = translation_matrix @ rotation_matrix
            box_noise.apply_transform(transform)
            mesh_list.append(box_noise)

    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return mesh_list, origin


def _interpolate_competition_range(value_range: tuple[float, float], difficulty: float, name: str) -> float:
    """Interpolate a competition obstacle parameter over terrain difficulty."""
    lower, upper = value_range
    if lower > upper:
        raise ValueError(f"{name} must be ordered as (min, max), got {value_range}.")
    return lower + float(np.clip(difficulty, 0.0, 1.0)) * (upper - lower)


def _make_competition_side_pits(
    size: tuple[float, float], track_width: float, ground_thickness: float, side_depth: float
) -> list[trimesh.Trimesh]:
    """Create lowered ground on both sides of a centered obstacle track."""
    if not 0.0 < track_width <= size[1]:
        raise ValueError(f"track_width must be in (0, {size[1]}], got {track_width}.")
    if side_depth >= 0.0:
        raise ValueError(f"side_depth must be negative, got {side_depth}.")
    side_width = 0.5 * (size[1] - track_width)
    if side_width <= 0.0:
        return []
    dimensions = (size[0], side_width, ground_thickness)
    center_z = side_depth - 0.5 * ground_thickness
    return [
        trimesh.creation.box(
            dimensions,
            trimesh.transformations.translation_matrix((0.5 * size[0], 0.5 * side_width, center_z)),
        ),
        trimesh.creation.box(
            dimensions,
            trimesh.transformations.translation_matrix((0.5 * size[0], size[1] - 0.5 * side_width, center_z)),
        ),
    ]


def _make_competition_track_ground(
    size: tuple[float, float], track_width: float, ground_thickness: float, side_depth: float
) -> list[trimesh.Trimesh]:
    """Create a centered z=0 track with lowered side-pit floors."""
    bottom_z = side_depth - ground_thickness
    center_ground = trimesh.creation.box(
        (size[0], track_width, -bottom_z),
        trimesh.transformations.translation_matrix((0.5 * size[0], 0.5 * size[1], 0.5 * bottom_z)),
    )
    return [
        center_ground,
        *_make_competition_side_pits(size, track_width, ground_thickness, side_depth),
    ]


def _make_competition_ramp_wedge(
    low_center: tuple[float, float],
    high_center: tuple[float, float],
    width: float,
    height: float,
) -> trimesh.Trimesh:
    """Create a watertight triangular prism rising from low to high center."""
    low = np.asarray(low_center, dtype=float)
    high = np.asarray(high_center, dtype=float)
    direction = high - low
    direction /= np.linalg.norm(direction)
    lateral = np.array([-direction[1], direction[0]])
    half_width = 0.5 * width
    low_left, low_right = low - half_width * lateral, low + half_width * lateral
    high_left, high_right = high - half_width * lateral, high + half_width * lateral
    vertices = np.array(
        [
            (*low_left, 0.0),
            (*low_right, 0.0),
            (*high_left, height),
            (*high_right, height),
            (*high_left, 0.0),
            (*high_right, 0.0),
        ]
    )
    faces = np.array(
        [
            (0, 2, 3),
            (0, 3, 1),
            (0, 1, 5),
            (0, 5, 4),
            (0, 4, 2),
            (1, 3, 5),
            (4, 5, 3),
            (4, 3, 2),
        ]
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def continuous_ramps_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshContinuousRampsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate repeated up-platform-down ramp units along the forward axis."""
    angle_deg = _interpolate_competition_range(cfg.slope_angle_range, difficulty, "slope_angle_range")
    if not 0.0 < angle_deg < 90.0:
        raise ValueError(f"Resolved slope angle must be in (0, 90), got {angle_deg}.")
    angle = np.deg2rad(angle_deg)
    horizontal_run = cfg.slope_length * np.cos(angle)
    ramp_height = cfg.slope_length * np.sin(angle)
    obstacle_length = cfg.num_ramps * (2.0 * horizontal_run + cfg.transition_length)
    if obstacle_length > cfg.size[0]:
        raise ValueError(f"Continuous ramps need {obstacle_length:.3f} m, but tile length is {cfg.size[0]:.3f} m.")

    approach_length = 0.5 * (cfg.size[0] - obstacle_length)
    profile_x = [0.0, approach_length]
    profile_z = [0.0, 0.0]
    x_position = approach_length
    for _ in range(cfg.num_ramps):
        x_position += horizontal_run
        profile_x.append(x_position)
        profile_z.append(ramp_height)
        x_position += cfg.transition_length
        profile_x.append(x_position)
        profile_z.append(ramp_height)
        x_position += horizontal_run
        profile_x.append(x_position)
        profile_z.append(0.0)
    profile_x.append(cfg.size[0])
    profile_z.append(0.0)

    y_min = 0.5 * (cfg.size[1] - cfg.ramp_width)
    y_max = y_min + cfg.ramp_width
    bottom_z = cfg.side_depth - cfg.ground_thickness
    vertices = []
    for x_position, height in zip(profile_x, profile_z):
        vertices.extend(
            [
                (x_position, y_min, height),
                (x_position, y_max, height),
                (x_position, y_min, bottom_z),
                (x_position, y_max, bottom_z),
            ]
        )
    faces = []
    for index in range(len(profile_x) - 1):
        start, stop = 4 * index, 4 * (index + 1)
        faces.extend(
            [
                (start, stop, stop + 1),
                (start, stop + 1, start + 1),
                (start + 2, start + 3, stop + 3),
                (start + 2, stop + 3, stop + 2),
                (start, start + 2, stop + 2),
                (start, stop + 2, stop),
                (start + 1, stop + 1, stop + 3),
                (start + 1, stop + 3, start + 3),
            ]
        )
    last = 4 * (len(profile_x) - 1)
    faces.extend([(0, 1, 3), (0, 3, 2), (last, last + 2, last + 3), (last, last + 3, last + 1)])
    meshes = [trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)]
    meshes.extend(_make_competition_side_pits(cfg.size, cfg.ramp_width, cfg.ground_thickness, cfg.side_depth))
    return meshes, np.array([0.5 * approach_length, 0.5 * cfg.size[1], 0.0])


def cross_ramps_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshCrossRampsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate alternating left-high/right-high transverse ramps."""
    ramp_height = _interpolate_competition_range(cfg.ramp_height_range, difficulty, "ramp_height_range")
    row_length = cfg.num_ramps * cfg.ramp_width
    if row_length > cfg.size[0] or cfg.horizontal_length > cfg.size[1]:
        raise ValueError("Cross-ramp dimensions exceed the terrain tile.")

    center_y = 0.5 * cfg.size[1]
    half_length = 0.5 * cfg.horizontal_length
    row_start_x = 0.5 * (cfg.size[0] - row_length)
    meshes = _make_competition_track_ground(cfg.size, cfg.horizontal_length, cfg.ground_thickness, cfg.side_depth)
    for index in range(cfg.num_ramps):
        center_x = row_start_x + (index + 0.5) * cfg.ramp_width
        left_is_high = cfg.start_left_high if index % 2 == 0 else not cfg.start_left_high
        low_y, high_y = (
            (center_y - half_length, center_y + half_length)
            if left_is_high
            else (center_y + half_length, center_y - half_length)
        )
        meshes.append(_make_competition_ramp_wedge((center_x, low_y), (center_x, high_y), cfg.ramp_width, ramp_height))
    return meshes, np.array([0.5 * row_start_x, center_y, 0.0])


def symmetric_ramps_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshSymmetricRampsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate two transverse ramps rising away from a low center gap."""
    ramp_height = _interpolate_competition_range(cfg.ramp_height_range, difficulty, "ramp_height_range")
    obstacle_span_y = 2.0 * cfg.horizontal_length + cfg.center_gap
    if cfg.ramp_width > cfg.size[0] or obstacle_span_y > cfg.size[1]:
        raise ValueError("Symmetric-ramp dimensions exceed the terrain tile.")

    center_x, center_y = 0.5 * cfg.size[0], 0.5 * cfg.size[1]
    half_gap = 0.5 * cfg.center_gap
    meshes = _make_competition_track_ground(cfg.size, obstacle_span_y, cfg.ground_thickness, cfg.side_depth)
    meshes.extend(
        [
            _make_competition_ramp_wedge(
                (center_x, center_y + half_gap),
                (center_x, center_y + half_gap + cfg.horizontal_length),
                cfg.ramp_width,
                ramp_height,
            ),
            _make_competition_ramp_wedge(
                (center_x, center_y - half_gap),
                (center_x, center_y - half_gap - cfg.horizontal_length),
                cfg.ramp_width,
                ramp_height,
            ),
        ]
    )
    obstacle_start_x = center_x - 0.5 * cfg.ramp_width
    return meshes, np.array([0.5 * obstacle_start_x, center_y, 0.0])


def braking_ramp_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshBrakingRampTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate ascending stairs immediately followed by a descending ramp."""
    angle_deg = _interpolate_competition_range(cfg.slope_angle_range, difficulty, "slope_angle_range")
    angle = np.deg2rad(angle_deg)
    obstacle_height = cfg.slope_surface_length * np.sin(angle)
    slope_run = cfg.slope_surface_length * np.cos(angle)
    stairs_run = cfg.num_steps * cfg.step_depth
    obstacle_length = stairs_run + slope_run
    if obstacle_length > cfg.size[0]:
        raise ValueError(f"Braking ramp needs {obstacle_length:.3f} m, but tile length is {cfg.size[0]:.3f} m.")

    center_y = 0.5 * cfg.size[1]
    start_x = 0.5 * (cfg.size[0] - obstacle_length)
    step_height = obstacle_height / cfg.num_steps
    meshes = _make_competition_track_ground(cfg.size, cfg.obstacle_width, cfg.ground_thickness, cfg.side_depth)
    for index in range(cfg.num_steps):
        height = (index + 1) * step_height
        center_x = start_x + (index + 0.5) * cfg.step_depth
        meshes.append(
            trimesh.creation.box(
                (cfg.step_depth, cfg.obstacle_width, height),
                trimesh.transformations.translation_matrix((center_x, center_y, 0.5 * height)),
            )
        )
    ramp_high_x = start_x + stairs_run
    meshes.append(
        _make_competition_ramp_wedge(
            (ramp_high_x + slope_run, center_y),
            (ramp_high_x, center_y),
            cfg.obstacle_width,
            obstacle_height,
        )
    )
    return meshes, np.array([0.5 * start_x, center_y, 0.0])


def consecutive_hurdles_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshConsecutiveHurdlesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate hollow transverse hurdles along the forward axis."""
    hurdle_height = _interpolate_competition_range(cfg.hurdle_height_range, difficulty, "hurdle_height_range")
    sequence_length = cfg.num_hurdles * cfg.hurdle_width + (cfg.num_hurdles - 1) * cfg.hurdle_spacing
    if sequence_length > cfg.size[0] or cfg.hurdle_length > cfg.size[1]:
        raise ValueError("Hurdle sequence dimensions exceed the terrain tile.")

    center_y = 0.5 * cfg.size[1]
    start_x = 0.5 * (cfg.size[0] - sequence_length)
    meshes = _make_competition_track_ground(cfg.size, cfg.hurdle_length, cfg.ground_thickness, cfg.side_depth)
    for index in range(cfg.num_hurdles):
        center_x = start_x + 0.5 * cfg.hurdle_width + index * (cfg.hurdle_width + cfg.hurdle_spacing)
        post_offset_y = 0.5 * (cfg.hurdle_length - cfg.hurdle_width)
        for offset_y in (-post_offset_y, post_offset_y):
            meshes.append(
                trimesh.creation.box(
                    (cfg.hurdle_width, cfg.hurdle_width, hurdle_height),
                    trimesh.transformations.translation_matrix((center_x, center_y + offset_y, 0.5 * hurdle_height)),
                )
            )
        meshes.append(
            trimesh.creation.box(
                (cfg.hurdle_width, cfg.hurdle_length, cfg.hurdle_width),
                trimesh.transformations.translation_matrix(
                    (center_x, center_y, hurdle_height - 0.5 * cfg.hurdle_width)
                ),
            )
        )
    return meshes, np.array([0.5 * start_x, center_y, 0.0])
