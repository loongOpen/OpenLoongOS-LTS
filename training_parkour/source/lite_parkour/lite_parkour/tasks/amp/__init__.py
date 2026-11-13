"""Register the two Lite-only tasks without importing simulator modules."""
import gymnasium as gym

for task_id, config in (("Lite-AMP-Depth-v0", "LiteAmpDepthEnvCfg"),
                        ("Lite-AMP-Depth-Play-v0", "LiteAmpDepthEnvCfg_PLAY")):
    gym.register(
        id=task_id,
        entry_point="lite_parkour.envs:ManagerBasedAmpEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.lite_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.lite_ppo_cfg:LiteAmpDepthPPORunnerCfg",
        },
    )
