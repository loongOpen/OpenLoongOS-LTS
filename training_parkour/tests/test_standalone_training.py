"""Checks for Lite-only registration, packaged data, configuration and sampling."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch
import lite_parkour
from lite_parkour.assets.lite import LITE_JOINT_ORDER, LITE_URDF
from lite_parkour.tasks.amp.lite_env_cfg import LiteAmpDepthEnvCfg, LiteAmpDepthEnvCfg_PLAY
from lite_parkour.tasks.amp.agents.lite_ppo_cfg import LiteAmpDepthPPORunnerCfg
from lite_parkour.tasks.amp.mdp.commands.sampling import resample_turning_commands, validate_turning_config

ROOT = Path(__file__).resolve().parents[1]

def test_only_lite_tasks_and_current_config():
    assert str(Path(lite_parkour.__file__).resolve()).startswith(str(ROOT))
    names=[name for name in gym.registry if gym.spec(name).entry_point == "lite_parkour.envs:ManagerBasedAmpEnv"]
    assert set(names)=={"Lite-AMP-Depth-v0", "Lite-AMP-Depth-Play-v0"}
    cfg=LiteAmpDepthEnvCfg();play=LiteAmpDepthEnvCfg_PLAY();runner=LiteAmpDepthPPORunnerCfg()
    assert len(cfg.actions.joint_pos.joint_names)==26
    assert cfg.actions.joint_pos.joint_names==list(LITE_JOINT_ORDER)
    assert cfg.commands.base_velocity.rel_standing_envs==0.02 and cfg.commands.base_velocity.rel_turning_envs==0.03
    assert len(cfg.scene.terrain.terrain_generator.sub_terrains)==8
    assert all(t.proportion==0.125 for t in cfg.scene.terrain.terrain_generator.sub_terrains.values())
    assert runner.experiment_name=="lite_amp_depth" and runner.save_interval==200
    assert runner.algorithm.class_name.startswith("lite_parkour.")
    assert play.commands.base_velocity.rel_turning_envs==0
    assert len(cfg.motion_data.motion_dataset.motion_data_weights)==54
    assert runner.algorithm.symmetry_cfg.data_augmentation_func.__module__.endswith("symmetry.humanoid")

def test_reference_clips_are_self_contained():
    script=ROOT/"scripts/tools/validate_motion_data.py"
    spec=importlib.util.spec_from_file_location("_motion_check",script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.validate()
    assert report=={"clips":72,"frames":7569,"dofs":26,"key_bodies":6}

def test_command_sampling_matches_95_2_3():
    cfg=LiteAmpDepthEnvCfg().commands.base_velocity;validate_turning_config(cfg)
    torch.manual_seed(7);count=100000;ids=torch.arange(count)
    standing=torch.rand(count)<cfg.rel_standing_envs;heading=torch.ones(count,dtype=torch.bool)
    velocity=torch.zeros(count,3);velocity[:,0]=0.6
    selected=resample_turning_commands(cfg,ids,velocity,standing,heading)
    assert abs(standing.float().mean().item()-.02)<.002
    assert abs(len(selected)/count-.03)<.002
    assert not standing[selected].any() and not heading[selected].any()
    assert not velocity[selected,:2].any()
    assert torch.all((velocity[selected,2].abs()>=.3)&(velocity[selected,2].abs()<=1))


def test_training_and_play_use_nominal_parameters_and_fixed_reset():
    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab.managers import EventTermCfg, ObservationTermCfg
    from lite_parkour.tasks.amp import mdp

    for cfg in (LiteAmpDepthEnvCfg(), LiteAmpDepthEnvCfg_PLAY()):
        events = [v for v in vars(cfg.events).values() if isinstance(v, EventTermCfg)]
        assert len(events) == 1
        assert events[0].mode == "reset" and events[0].func is mdp.reset_scene_to_default
        assert not events[0].params
        for actuator in cfg.scene.robot.actuators.values():
            assert type(actuator) is IdealPDActuatorCfg
        for group in vars(cfg.observations).values():
            assert not group.enable_corruption
            assert all(v.noise is None for v in vars(group).values() if isinstance(v, ObservationTermCfg))
        assert cfg.observations.depth.depth_image.func is mdp.DepthImageHistory
        assert "delayed_frame_ranges" not in cfg.observations.depth.depth_image.params
        assert not hasattr(cfg.animation, "animation_reset")


def test_depth_history_returns_latest_frame_and_resets_without_rng(monkeypatch):
    from isaaclab.managers import ObservationTermCfg
    from lite_parkour.tasks.amp.mdp.depth import DepthImageHistory

    images = torch.full((2, 36, 64, 1), 0.5)
    env = SimpleNamespace(num_envs=2, device="cpu", scene=SimpleNamespace(sensors={
        "depth_camera": SimpleNamespace(data=SimpleNamespace(output={"distance_to_image_plane": images}))
    }))
    cfg = ObservationTermCfg(func=DepthImageHistory, params=LiteAmpDepthEnvCfg().observations.depth.depth_image.params)
    term = DepthImageHistory(cfg, env)
    def unexpected_rng(*args, **kwargs):
        raise AssertionError("Depth reset must not sample delays")
    monkeypatch.setattr(torch, "randint", unexpected_rng)
    term.reset()
    torch.testing.assert_close(term(env), torch.full((2, 8, 18, 32), 0.2))
    images.fill_(1.0)
    frames = term(env)
    torch.testing.assert_close(frames[:, -1], torch.full((2, 18, 32), 0.4))
    torch.testing.assert_close(frames[:, 0], torch.full((2, 18, 32), 0.2))
    term.reset([0])
    images[0].fill_(1.5)
    images[1].fill_(2.0)
    frames = term(env)
    torch.testing.assert_close(frames[0], torch.full((8, 18, 32), 0.6))
    torch.testing.assert_close(frames[1, 0], torch.full((18, 32), 0.2))
    torch.testing.assert_close(frames[1, -1], torch.full((18, 32), 0.8))
