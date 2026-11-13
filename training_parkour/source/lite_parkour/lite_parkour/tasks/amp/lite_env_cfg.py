"""Lite perceptive walking: AMP style reward, chest depth camera, rough-terrain curriculum."""

import hashlib
import json
import math
import re

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import ResolvableString
from isaaclab_physx.physics import PhysxCfg

import lite_parkour.tasks.amp.mdp as mdp
import lite_parkour.terrains as perlin_gen
from lite_parkour.assets.lite import (
    LITE_ACTION_SCALE,
    LITE_CFG,
    LITE_DEPTH_VISUAL_PATHS,
    LITE_JOINT_ORDER,
    LITE_KEY_BODIES,
    LITE_URDF,
)
from lite_parkour.envs.manager_based_amp_env_cfg import ManagerBasedAmpEnvCfg
from lite_parkour.managers import AnimationTermCfg as AnimTerm
from lite_parkour.managers import MotionDataTermCfg as MotionDataTerm

##
# Constants
##

MOTION_DIR = LITE_URDF.parents[2] / "MotionData/Lite/amp/reference"
STAIR_CLIP_WEIGHT = 0.3
AMP_NUM_STEPS = 4

# Chest camera on Lite's rigid torso, pitched 45 degrees down (x, y, z, w).
# Chest camera position relative to base_link.
CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
CAMERA_POS = (0.1237, 0.0, 0.40966)
CAMERA_ROT = (0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))
# String class type keeps this module importable before the simulation app starts.
CAMERA_CLASS_TYPE = "lite_parkour.sensors.pose_marked_ray_caster_camera:PoseMarkedMultiMeshRayCasterCamera"

# Contact on the torso or proximal limb bodies ends the episode.
TERMINATION_CONTACT_BODIES = [
    "base_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
]

ARM_JOINTS = [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]


def joint_cfg() -> SceneEntityCfg:
    """All 26 joints in the policy order."""
    return SceneEntityCfg("robot", joint_names=list(LITE_JOINT_ORDER), preserve_order=True)


def motion_clip_pools() -> tuple[dict[str, float], dict[str, float]]:
    """Walk and stair clips of the retargeted dataset, as ``(walk, stairs)`` weight maps.

    Run clips (``C<n>_...``) and side steps are not used: commands are forward walking only.
    """
    manifest = json.loads((MOTION_DIR / "conversion_manifest.json").read_text())
    if manifest["joint_names"] != list(LITE_JOINT_ORDER) or manifest["target_root"] != "base_link":
        raise ValueError("Motion data joint order or root link does not match the robot.")
    if manifest["urdf_sha256"][LITE_URDF.name] != hashlib.sha256(LITE_URDF.read_bytes()).hexdigest():
        raise ValueError("Lite URDF changed; update the reference trajectories and run scripts/tools/validate_motion_data.py.")
    walk, stairs = {}, {}
    for clip in manifest["clips"]:
        name = clip["name"]
        if clip["stairs"]:
            stairs[name] = STAIR_CLIP_WEIGHT
        elif not re.match(r"C\d+_", name) and "side_step" not in name.lower():
            walk[name] = 1.0
    return walk, stairs


def depth_camera_cfg() -> MultiMeshRayCasterCameraCfg:
    """64x36 ray-cast depth camera on the chest, occluded by the robot's own links."""
    return MultiMeshRayCasterCameraCfg(
        class_type=ResolvableString(CAMERA_CLASS_TYPE),
        prim_path=CAMERA_PRIM_PATH,
        mesh_prim_paths=[
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
            # One target per link keeps the per-environment pose indexing correct.
            *[
                MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                    prim_expr=f"{{ENV_REGEX_NS}}/Robot/Geometry/{path}", is_shared=True, track_mesh_transforms=True
                )
                for path in LITE_DEPTH_VISUAL_PATHS
            ],
        ],
        ray_alignment="base",
        pattern_cfg=PinholeCameraPatternCfg(
            func=mdp.pinhole_camera_pattern_analytic,
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
            width=64,
            height=36,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        max_distance=2.5,
        depth_clipping_behavior="max",
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=CAMERA_POS, rot=CAMERA_ROT, convention="world"),
    )


##
# Terrain
##

# Perlin tiles need the wall fields; walls stay off.
_NO_WALL = dict(wall_prob=[0.0, 0.0, 0.0, 0.0], wall_height=5.0, wall_thickness=0.05)


def _perlin_overlay():
    return perlin_gen.PerlinPlaneTerrainCfg(
        noise_scale=0.04, noise_frequency=20, fractal_octaves=2, fractal_lacunarity=2.0, fractal_gain=0.25,
        centering=True,
    )


LITE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=60.0, num_rows=10, num_cols=24,
    horizontal_scale=0.05, vertical_scale=0.005, slope_threshold=0.75,
    curriculum=True, use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.125, step_height_range=(0.1, 0.3), step_width=0.35,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.125, step_height_range=(0.05, 0.25), step_width=0.35,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.125, grid_width=0.45, grid_height_range=(0.05, 0.20), platform_width=2.0,
        ),
        "random_rough": perlin_gen.PerlinPlaneTerrainCfg(
            proportion=0.125, noise_scale=[0.05, 0.15], noise_frequency=20, fractal_octaves=2,
            fractal_lacunarity=2.0, fractal_gain=0.25, centering=True, border_width=0.25, **_NO_WALL,
        ),
        "hf_pyramid_slope": perlin_gen.PerlinPyramidSlopedTerrainCfg(
            proportion=0.125, slope_range=(0.1, 0.7), platform_width=2.0, border_width=0.25,
            perlin_cfg=_perlin_overlay(), **_NO_WALL,
        ),
        "hf_pyramid_slope_inv": perlin_gen.PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.125, slope_range=(0.0, 0.5), platform_width=2.0, border_width=0.25,
            perlin_cfg=_perlin_overlay(), **_NO_WALL,
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.125, rail_height_range=(0.05, 0.4), rail_thickness_range=(0.1, 0.3), platform_width=2.0,
        ),
        "repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            # object_type must be the short built-in name, not the class default import path.
            proportion=0.125, object_type="cylinder", platform_width=2.0, platform_height=0.0,
            abs_height_noise=(-0.1, 0.1), rel_height_noise=(0.5, 2.0),
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=10, height=1.0, radius=0.2, max_yx_angle=5.0, degrees=True,
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=30, height=1.0, radius=0.2, max_yx_angle=10.0, degrees=True,
            ),
        ),
    },
)


##
# Scene
##


@configclass
class LiteSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=LITE_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = LITE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # Terrain-relative base height for the fall termination; not a policy input.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    depth_camera = depth_camera_cfg()
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP
##


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LITE_JOINT_ORDER),
        preserve_order=True,
        scale=dict(LITE_ACTION_SCALE),
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.AmpVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_turning_envs=0.03,
        turning_ang_vel_range=(0.3, 1.0),
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=False,
        ranges=mdp.AmpVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception with a 5-step history."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, history_length=5, flatten_history_dim=True
        )
        root_local_rot_tan_norm = ObsTerm(
            func=mdp.root_local_rot_tan_norm, history_length=5,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, history_length=5,
            flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos, params={"asset_cfg": joint_cfg()},
            history_length=5, flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel, params={"asset_cfg": joint_cfg()},
            history_length=5, flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)

        def __post_init__(self):
            self.history_length = None
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Policy terms without noise, plus base linear velocity and key body positions."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=5, flatten_history_dim=True)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=5, flatten_history_dim=True)
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm, history_length=5, flatten_history_dim=True)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, history_length=5,
            flatten_history_dim=True,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": joint_cfg()}, history_length=5,
                            flatten_history_dim=True)
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": joint_cfg()}, history_length=5,
                            flatten_history_dim=True)
        actions = ObsTerm(func=mdp.last_action, history_length=5, flatten_history_dim=True)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=list(LITE_KEY_BODIES), preserve_order=True)},
            history_length=5,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.history_length = None
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DepthCfg(ObsGroup):
        """Cropped, blurred, normalized depth frames with a sparse history and no added delay."""

        depth_image = ObsTerm(
            func=mdp.DepthImageHistory,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "crop_region": (18, 0, 16, 16),
                "blur_kernel_size": 3,
                "blur_sigma": 1.0,
                "depth_range": (0.0, 2.5),
                "history_length": 37,
                "history_skip_frames": 5,
                "num_output_frames": 8,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = None

    @configclass
    class DiscriminatorCfg(ObsGroup):
        """Agent side of the AMP discriminator: angular velocity and joint state over 4 steps."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": joint_cfg()})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": joint_cfg()})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = AMP_NUM_STEPS
            self.flatten_history_dim = False

    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        """Reference side of the AMP discriminator, matching DiscriminatorCfg term by term."""

        ref_root_ang_vel_b = ObsTerm(
            func=mdp.ref_root_ang_vel_b, params={"animation": "animation", "flatten_steps_dim": False}
        )
        ref_joint_pos = ObsTerm(func=mdp.ref_joint_pos, params={"animation": "animation", "flatten_steps_dim": False})
        ref_joint_vel = ObsTerm(func=mdp.ref_joint_vel, params={"animation": "animation", "flatten_steps_dim": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    depth: DepthCfg = DepthCfg()
    disc: DiscriminatorCfg = DiscriminatorCfg()
    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()


@configclass
class RewardsCfg:
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # -- base motion
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # -- joint effort and smoothness
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # -- joint safety limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-1.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_by_ratio, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "limit_ratio": 0.8},
    )
    # -- posture regularization (L1 distance to the default pose)
    # Standing terms apply only when both |v_xy| and |w_z| commands are below 0.06.
    joint_deviation_hip = RewTerm(
        func=mdp.stand_still_joint_deviation_l1, weight=-0.1,
        params={
            "command_name": "base_velocity", "command_threshold": 0.06, "angular_command_threshold": 0.06,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)}
    )
    joint_deviation_arms_standing = RewTerm(
        func=mdp.stand_still_joint_deviation_l1, weight=-0.04,
        params={
            "command_name": "base_velocity", "command_threshold": 0.06, "angular_command_threshold": 0.06,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
        },
    )
    # -- feet
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped, weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "force_ratio": 4.0},
    )
    # -- contacts
    undesired_arm_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_elbow_link", ".*_wrist_.*"]), "threshold": 1.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=TERMINATION_CONTACT_BODIES), "threshold": 1.0},
    )
    # Base height above the scanned terrain.
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum_adaptive,
        params={"minimum_height": 0.55, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class EventCfg:
    # Fixed default pose and zero velocity, translated to the current terrain origin.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # Logging only: mean level per sub-terrain. Must stay after terrain_levels.
    terrain_levels_by_type = CurrTerm(func=mdp.terrain_levels_by_type)


@configclass
class MotionDataCfg:
    # Discriminator demonstrations: walk and stair clips. Weights are filled in from the manifest.
    motion_dataset = MotionDataTerm(motion_data_dir=str(MOTION_DIR), motion_data_weights={})


@configclass
class AnimationCfg:
    animation = AnimTerm(
        motion_data_term="motion_dataset",
        motion_data_components=[
            "root_pos_w", "root_quat", "root_vel_w", "root_ang_vel_w", "dof_pos", "dof_vel", "key_body_pos_b",
        ],
        num_steps_to_use=AMP_NUM_STEPS,
        random_initialize=True,
        random_fetch=True,
        enable_visualization=False,
    )



##
# Environment
##


@configclass
class LiteAmpDepthEnvCfg(ManagerBasedAmpEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15, gpu_collision_stack_size=2**27)
    )
    scene: LiteSceneCfg = LiteSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    motion_data: MotionDataCfg = MotionDataCfg()
    animation: AnimationCfg = AnimationCfg()
    terminal_obs_groups = ("disc",)

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        walk, stairs = motion_clip_pools()
        self.motion_data.motion_dataset.motion_data_weights = {**walk, **stairs}


@configclass
class LiteAmpDepthEnvCfg_PLAY(LiteAmpDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        terrain = self.scene.terrain.terrain_generator
        terrain.num_rows = 1
        terrain.num_cols = 1
        terrain.curriculum = False
        terrain.sub_terrains = {
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0,
                step_height_range=(0.20, 0.20),
                step_width=0.35,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        }
        # MeshPyramidStairsTerrainCfg places the terrain origin on the center
        # summit. Start on the low front border and face +x, so Play goes up
        # the stairs, crosses the summit, and then comes down the other side.
        # With the current 8 m terrain and 0.20 m steps, the summit is 1.20 m
        # above the low border. Subtract it from the default base height (0.99 m).
        self.scene.robot.init_state.pos = (-3.5, 0.0, -0.21)
        self.curriculum.terrain_levels = None
        self.curriculum.terrain_levels_by_type = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_turning_envs = 0.0
