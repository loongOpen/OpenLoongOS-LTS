# Adapted from:
#   instinctlab (CC BY-NC 4.0)
#     https://github.com/project-instinct/instinctlab
#   Isaac Lab (BSD-3-Clause)
#     https://github.com/isaac-sim/IsaacLab

from dataclasses import MISSING
from typing import List

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass

from ..height_field import PerlinPlaneTerrainCfg


class WallTerrainCfgMixin:
    wall_prob: List[float] = [0.0, 0.0, 0.0, 0.0]  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls


@configclass
class MotionMatchedTerrainCfg(SubTerrainBaseCfg):
    """Configuration for motion-matched terrain generation.

    ## Terrain Mesh Requirements
    - All terrain meshes must have the a border at the bottom.
    - The terrain origin (0, 0, 0) is at the surface of the terrain center, which means that the point should
        be above the terrain at (0, 0, t) given any t > 0 and below the terrain at (0, 0, t) given any t < 0.
    - The USER should ensure that the non-flat part of the terrain is within the size of the terrain.
    """

    function: str = "{DIR}.mesh_terrains:motion_matched_terrain"

    path: str = MISSING
    """Directory containing both terrains and the motions, so that these can be matched together.
    """

    metadata_yaml: str = MISSING
    """YAML file containing the motion matching configuration.
    This file should specify the motion matching parameters, such as the motion files to be used,
    the matching criteria, and any other relevant settings.

    You may use the `scripts/motion_matched_metadata_generator.py` to generate the metadata.yaml file if you arrange your
    dataset in the structure as described in `scripts/motion_matched_metadata_generator.py`.

    ## Typical yaml file structure

    ```yaml
    terrains:
        - terrain_id: "jumpbox1" # can be any string.
          terrain_file: "path/to/terrain.stl" # path to the terrain mesh file, relative to the datasetdir.
        - terrain_id: "jumpbox2"
          terrain_file: "path/to/another_terrain.stl"
    motion_files:
        - terrain_id: "jumpbox1" # should match the terrain_id above.
          motion_file: "path/to/motion1_poses.npz" # path to the motion file, relative to the datasetdir.
          weight: (optional) 1.0
        - terrain_id: "jumpbox2"
          motion_file: "path/to/motion2_retargetted.npz"
          weight: (optional) 1.0
    ```

    """


@configclass
class PerlinMeshFloatingBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a floating box mesh terrain."""

    function: str = "{DIR}.mesh_terrains:floating_box_terrain"
    floating_height: tuple[float, float] | float = MISSING
    """The height of the box above the ground. Could be a fixed value or a range (min, max)."""
    box_length: tuple[float, float] | float = MISSING
    """The length of the box along the y-axis. Could be a fixed value or a range (min, max)."""
    box_width: float | None = None
    """The width of the box along the x-axis. If None, it will be equal to the width of the terrain."""
    box_height: tuple[float, float] | float = MISSING
    """The height of the box along the z-axis."""
    perlin_cfg: PerlinPlaneTerrainCfg | None = None

    # values used for perlin noise generation
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = True
    """If True, no perlin noise will be generated exactly below the box."""


@configclass
class PerlinMeshRandomMultiBoxTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """Configuration for a sub terrain with multiple random boxes with perlin noise."""

    function: str = "{DIR}.mesh_terrains:random_multi_box_terrain"
    box_height_mean: tuple[float, float] | float = MISSING
    box_height_range: float = MISSING
    box_length_mean: tuple[float, float] | float = MISSING
    box_length_range: float = MISSING
    box_width_mean: tuple[float, float] | float = MISSING
    box_width_range: float = MISSING
    platform_width: float = MISSING

    generation_ratio: float = MISSING

    perlin_cfg: PerlinPlaneTerrainCfg | None = None
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float | None = None
    no_perlin_at_obstacle: bool = False
    box_perlin_cfg: PerlinPlaneTerrainCfg | None = None
    """Used only when perlin_cfg is not None"""


@configclass
class MeshContinuousRampsTerrainCfg(SubTerrainBaseCfg):
    """Consecutive up-platform-down ramp units."""

    function: str = "{DIR}.mesh_terrains:continuous_ramps_terrain"

    ramp_width: float = 2.4
    slope_angle_range: tuple[float, float] = (5.0, 15.0)
    slope_length: float = 1.5
    transition_length: float = 0.8
    num_ramps: int = 2
    side_depth: float = -2.0
    ground_thickness: float = 0.1


@configclass
class MeshCrossRampsTerrainCfg(SubTerrainBaseCfg):
    """Transverse ramps arranged in a row with alternating slopes."""

    function: str = "{DIR}.mesh_terrains:cross_ramps_terrain"

    horizontal_length: float = 3.0
    ramp_height_range: tuple[float, float] = (0.2, 0.8)
    ramp_width: float = 1.5
    num_ramps: int = 4
    start_left_high: bool = False
    side_depth: float = -2.0
    ground_thickness: float = 0.1


@configclass
class MeshSymmetricRampsTerrainCfg(SubTerrainBaseCfg):
    """Two transverse ramps rising symmetrically from a center gap."""

    function: str = "{DIR}.mesh_terrains:symmetric_ramps_terrain"

    horizontal_length: float = 1.7
    ramp_width: float = 3.0
    ramp_height_range: tuple[float, float] = (0.15, 0.62)
    center_gap: float = 0.1
    side_depth: float = -2.0
    ground_thickness: float = 0.1


@configclass
class MeshBrakingRampTerrainCfg(SubTerrainBaseCfg):
    """Ascending stairs connected to a descending braking ramp."""

    function: str = "{DIR}.mesh_terrains:braking_ramp_terrain"

    obstacle_width: float = 2.0
    num_steps: int = 4
    step_depth: float = 0.3
    slope_surface_length: float = 3.0
    slope_angle_range: tuple[float, float] = (10.0, 30.0)
    side_depth: float = -2.0
    ground_thickness: float = 0.1


@configclass
class MeshConsecutiveHurdlesTerrainCfg(SubTerrainBaseCfg):
    """Transverse hurdles placed consecutively along the forward axis."""

    function: str = "{DIR}.mesh_terrains:consecutive_hurdles_terrain"

    num_hurdles: int = 2
    hurdle_spacing: float = 1.1
    hurdle_length: float = 2.4
    hurdle_width: float = 0.05
    hurdle_height_range: tuple[float, float] = (0.1, 0.3)
    side_depth: float = -2.0
    ground_thickness: float = 0.1
