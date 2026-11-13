"""List the two Lite tasks without launching Isaac Sim."""
import gymnasium as gym
import lite_parkour.tasks  # noqa: F401

for name in sorted(gym.registry):
    if name.startswith("Lite-"):
        print(name, "->", gym.spec(name).kwargs["env_cfg_entry_point"])
