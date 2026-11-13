"""Motion mimic task configuration.

This module defines the base configuration for motion mimic tasks.
Robot-specific configurations are located in the config/ directory.

This is a re-implementation of BeyondMimic (https://beyondmimic.github.io/).

Based on https://github.com/HybridRobotics/whole_body_tracking
Commit: f8e20c880d9c8ec7172a13d3a88a65e3a5a88448
"""
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base tracking task configuration."""
    actor_terms = {'command': ObservationTermCfg(func=mdp.generated_commands, params={'command_name': 'motion'}), 'motion_anchor_pos_b': ObservationTermCfg(func=mdp.motion_anchor_pos_b, params={'command_name': 'motion'}), 'motion_anchor_ori_b': ObservationTermCfg(func=mdp.motion_anchor_ori_b, params={'command_name': 'motion'}), 'base_lin_vel': ObservationTermCfg(func=mdp.builtin_sensor, params={'sensor_name': 'robot/imu_lin_vel'}), 'base_ang_vel': ObservationTermCfg(func=mdp.builtin_sensor, params={'sensor_name': 'robot/imu_ang_vel'}), 'joint_pos': ObservationTermCfg(func=mdp.joint_pos_rel, params={'biased': False}), 'joint_vel': ObservationTermCfg(func=mdp.joint_vel_rel), 'actions': ObservationTermCfg(func=mdp.last_action)}
    critic_terms = {'command': ObservationTermCfg(func=mdp.generated_commands, params={'command_name': 'motion'}), 'motion_anchor_pos_b': ObservationTermCfg(func=mdp.motion_anchor_pos_b, params={'command_name': 'motion'}), 'motion_anchor_ori_b': ObservationTermCfg(func=mdp.motion_anchor_ori_b, params={'command_name': 'motion'}), 'body_pos': ObservationTermCfg(func=mdp.robot_body_pos_b, params={'command_name': 'motion'}), 'body_ori': ObservationTermCfg(func=mdp.robot_body_ori_b, params={'command_name': 'motion'}), 'base_lin_vel': ObservationTermCfg(func=mdp.builtin_sensor, params={'sensor_name': 'robot/imu_lin_vel'}), 'base_ang_vel': ObservationTermCfg(func=mdp.builtin_sensor, params={'sensor_name': 'robot/imu_ang_vel'}), 'joint_pos': ObservationTermCfg(func=mdp.joint_pos_rel), 'joint_vel': ObservationTermCfg(func=mdp.joint_vel_rel), 'actions': ObservationTermCfg(func=mdp.last_action)}
    observations = {'actor': ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=False), 'critic': ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False)}
    actions: dict[str, ActionTermCfg] = {'joint_pos': JointPositionActionCfg(entity_name='robot', actuator_names=('.*',), scale=0.5, use_default_offset=True)}
    commands: dict[str, CommandTermCfg] = {'motion': MotionCommandCfg(entity_name='robot', resampling_time_range=(1000000000.0, 1000000000.0), debug_vis=True, pose_range={}, velocity_range={}, joint_position_range=(0.0, 0.0), motion_file='', anchor_body_name='', body_names=())}
    events = {}
    rewards: dict[str, RewardTermCfg] = {'motion_global_root_pos': RewardTermCfg(func=mdp.motion_global_anchor_position_error_exp, weight=0.5, params={'command_name': 'motion', 'std': 0.3}), 'motion_global_root_ori': RewardTermCfg(func=mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={'command_name': 'motion', 'std': 0.4}), 'motion_body_pos': RewardTermCfg(func=mdp.motion_relative_body_position_error_exp, weight=1.0, params={'command_name': 'motion', 'std': 0.3}), 'motion_body_ori': RewardTermCfg(func=mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={'command_name': 'motion', 'std': 0.4}), 'motion_body_lin_vel': RewardTermCfg(func=mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={'command_name': 'motion', 'std': 1.0}), 'motion_body_ang_vel': RewardTermCfg(func=mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={'command_name': 'motion', 'std': 3.14}), 'action_rate_l2': RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1), 'joint_limit': RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0, params={'asset_cfg': SceneEntityCfg('robot', joint_names=('.*',))}), 'self_collisions': RewardTermCfg(func=mdp.self_collision_cost, weight=-10.0, params={'sensor_name': 'self_collision', 'force_threshold': 10.0})}
    terminations: dict[str, TerminationTermCfg] = {'time_out': TerminationTermCfg(func=mdp.time_out, time_out=True), 'anchor_pos': TerminationTermCfg(func=mdp.bad_anchor_pos_z_only, params={'command_name': 'motion', 'threshold': 0.25}), 'anchor_ori': TerminationTermCfg(func=mdp.bad_anchor_ori, params={'asset_cfg': SceneEntityCfg('robot'), 'command_name': 'motion', 'threshold': 0.8}), 'ee_body_pos': TerminationTermCfg(func=mdp.bad_motion_body_pos_z_only, params={'command_name': 'motion', 'threshold': 0.25, 'body_names': ()})}
    return ManagerBasedRlEnvCfg(scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type='plane'), num_envs=1), observations=observations, actions=actions, commands=commands, events=events, rewards=rewards, terminations=terminations, viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name='robot', body_name='', distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0), sim=SimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)), decimation=4, episode_length_s=10.0)
