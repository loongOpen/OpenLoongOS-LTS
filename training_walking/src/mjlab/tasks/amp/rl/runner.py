"""AMP on-policy runner: rsl_rl PPO + separate discriminator, no changes to velocity PPO."""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.env import VecEnv

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.amp.mdp import (
  AMP_STEP_AMPLITUDE_WEIGHT,
  apply_amp_stand_stage_weights,
)
from mjlab.tasks.amp.observations import compute_amp_observations
from mjlab.tasks.amp.rl.discriminator import AmpDiscriminator
from mjlab.tasks.amp.rl.motion_loader import AmpMotionLoader
from mjlab.tasks.amp.rl.normalizer import EmpiricalNormalization
from mjlab.tasks.amp.rl.replay_buffer import AmpReplayBuffer
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class AmpOnPolicyRunner(VelocityOnPolicyRunner):
  """Velocity PPO runner with AMP style reward and discriminator updates."""

  env: RslRlVecEnvWrapper

  amp_loader_cls = AmpMotionLoader
  amp_obs_fn = staticmethod(compute_amp_observations)

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
  ) -> None:
    amp_cfg = dict(train_cfg.pop("amp", {}))
    super().__init__(env, train_cfg, log_dir, device)
    self._init_amp(amp_cfg)
    self._install_hooks()

  def _init_amp(self, amp_cfg: dict) -> None:
    device = self.device
    unwrapped = self.env.unwrapped
    motion_files = tuple(amp_cfg.get("motion_files") or ())
    if not motion_files:
      raise ValueError("AmpOnPolicyRunner requires amp.motion_files")

    self.amp_data = self.amp_loader_cls(
      device=device,
      time_between_frames=float(unwrapped.step_dt),
      motion_files=motion_files,
      preload_transitions=True,
      num_preload_transitions=int(amp_cfg.get("num_preload_transitions", 200_000)),
    )
    obs_dim = self.amp_data.observation_dim
    self.amp_normalizer = EmpiricalNormalization(shape=obs_dim).to(device)
    self.discriminator = AmpDiscriminator(
      observation_dim=obs_dim,
      hidden_dims=tuple(amp_cfg.get("hidden_dims", (1024, 512, 256))),
      reward_coef=float(amp_cfg.get("reward_coef", 0.3)),
      task_reward_lerp=float(amp_cfg.get("task_reward_lerp", 0.5)),
      style_reward_function=str(amp_cfg.get("style_reward_function", "quad_mapping")),
    ).to(device)
    self.amp_storage = AmpReplayBuffer(
      obs_dim,
      int(amp_cfg.get("replay_buffer_size", 1_000_000)),
      device,
    )
    self._disc_lr = float(amp_cfg.get("learning_rate", 3e-4))
    self._disc_wd = float(amp_cfg.get("weight_decay", 1e-3))
    self._disc_gp = float(amp_cfg.get("gradient_penalty_coef", 10.0))
    self._disc_loss = str(amp_cfg.get("loss_function", "MSELoss"))
    self._disc_minibatches = int(amp_cfg.get("num_mini_batches", 80))
    self._low_vx_task_gate = bool(amp_cfg.get("low_vx_task_gate", False))
    self._low_vx_abs = float(amp_cfg.get("low_vx_abs", 0.3))
    self._low_vy_abs = float(amp_cfg.get("low_vy_abs", 0.3))
    self._low_wz_abs = float(amp_cfg.get("low_wz_abs", 0.15))
    self._low_vx_task_reward_lerp = float(amp_cfg.get("low_vx_task_reward_lerp", 0.7))
    self._amp_mix_switch_iter = int(amp_cfg.get("mix_switch_iter", 0) or 0)
    self._stage1_reward_coef = float(amp_cfg.get("reward_coef", 0.3))
    self._stage1_task_reward_lerp = float(amp_cfg.get("task_reward_lerp", 0.5))
    self._stage2_reward_coef = float(amp_cfg.get("stage2_reward_coef", 0.5))
    self._stage2_task_reward_lerp = float(amp_cfg.get("stage2_task_reward_lerp", 0.5))
    self._amp_mix_stage2: bool | None = None
    self._stand_switch_iter = int(amp_cfg.get("stand_switch_iter", 0) or 0)
    self._amp_stand_gate_on: bool | None = None
    self.discriminator_optimizer = torch.optim.AdamW(
      self.discriminator.parameters(),
      lr=self._disc_lr,
      weight_decay=self._disc_wd,
    )
    self._prev_amp_obs = self.amp_obs_fn(unwrapped).to(device)
    # Joints with pose std=0 must be skipped from iter 0, or the pose reward
    # freezes them.
    unwrapped._amp_skip_zero_pose_std = True
    mix_msg = (
      f"mix coef={self._stage1_reward_coef} lerp={self._stage1_task_reward_lerp}"
    )
    if self._amp_mix_switch_iter > 0:
      mix_msg += (
        f" until iter {self._amp_mix_switch_iter}, then "
        f"coef={self._stage2_reward_coef} lerp={self._stage2_task_reward_lerp}"
      )
    print(
      f"[AMP] discriminator input {self.discriminator.input_dim}, "
      f"obs {obs_dim}, motions {len(motion_files)}, {mix_msg}, "
      f"low_vx_task_gate={self._low_vx_task_gate} "
      f"(from iter {self._stand_switch_iter if self._stand_switch_iter > 0 else 0}: "
      f"|vx|<{self._low_vx_abs} |vy|<{self._low_vy_abs} |wz|<{self._low_wz_abs} "
      f"→ lerp {self._low_vx_task_reward_lerp})"
    )

  def _sync_amp_mix(self) -> None:
    """Apply lerp08 → lerp05_coef05 mix after ``mix_switch_iter`` PPO updates."""
    switch_iter = self._amp_mix_switch_iter
    if switch_iter <= 0:
      return
    step = int(self.env.unwrapped.common_step_counter)
    use_stage2 = step >= switch_iter * int(self.cfg["num_steps_per_env"])
    if use_stage2 is self._amp_mix_stage2:
      return
    self._amp_mix_stage2 = use_stage2
    if use_stage2:
      self.discriminator.reward_coef = self._stage2_reward_coef
      self.discriminator.task_reward_lerp = self._stage2_task_reward_lerp
    else:
      self.discriminator.reward_coef = self._stage1_reward_coef
      self.discriminator.task_reward_lerp = self._stage1_task_reward_lerp
    stage = "stage2" if use_stage2 else "stage1"
    print(
      f"[AMP] mix {stage} at step {step}: "
      f"coef={self.discriminator.reward_coef} "
      f"lerp={self.discriminator.task_reward_lerp}"
    )

  def _stand_gate_enabled(self) -> bool:
    if not self._low_vx_task_gate:
      return False
    unwrapped = self.env.unwrapped
    if self._stand_switch_iter <= 0:
      on = True
      step = int(unwrapped.common_step_counter)
    else:
      step = int(unwrapped.common_step_counter)
      on = step >= self._stand_switch_iter * int(self.cfg["num_steps_per_env"])
    apply_amp_stand_stage_weights(unwrapped, on=on)
    if on is not self._amp_stand_gate_on:
      self._amp_stand_gate_on = on
      if on:
        print(
          f"[AMP] stand stage on at step {step}: "
          f"gate lerp={self._low_vx_task_reward_lerp}, stand_still=-1.0, "
          f"step_amplitude_symmetry={AMP_STEP_AMPLITUDE_WEIGHT} "
          f"(pure |vx|>={self._low_vx_abs}, |vy| and |wz| below {self._low_vx_abs})"
        )
    return on

  def _amp_task_reward_lerp(self) -> torch.Tensor | float:
    lerp = self.discriminator.task_reward_lerp
    cmd = self.env.unwrapped.command_manager.get_command("twist")
    assert cmd is not None
    out = torch.full_like(cmd[:, 0], lerp)
    if self._stand_gate_enabled():
      standing_like = (
        (cmd[:, 0].abs() < self._low_vx_abs)
        & (cmd[:, 1].abs() < self._low_vy_abs)
        & (cmd[:, 2].abs() < self._low_wz_abs)
      )
      out = torch.where(
        standing_like,
        torch.full_like(cmd[:, 0], self._low_vx_task_reward_lerp),
        out,
      )
    return out

  def _install_hooks(self) -> None:
    orig_step = self.env.step

    def stepped(actions: torch.Tensor):
      self._sync_amp_mix()
      self._stand_gate_enabled()
      obs, rewards, dones, extras = orig_step(actions)
      next_amp = self.amp_obs_fn(self.env.unwrapped).to(self.device)
      lerp = self._amp_task_reward_lerp()
      mixed, style = self.discriminator.predict_amp_reward(
        self._prev_amp_obs,
        next_amp,
        rewards.to(self.device),
        self.amp_normalizer,
        task_reward_lerp=lerp,
      )
      self.amp_storage.insert(self._prev_amp_obs, next_amp)
      self._prev_amp_obs = next_amp.detach()
      extras.setdefault("log", {})
      extras["log"]["amp_style_reward"] = style.mean()
      extras["log"]["amp_task_reward"] = rewards.mean()
      extras["log"]["amp_total_reward"] = mixed.mean()
      extras["log"]["amp_task_reward_lerp"] = (
        lerp.mean() if torch.is_tensor(lerp) else torch.tensor(float(lerp), device=self.device)
      )
      extras["log"]["amp_reward_coef"] = torch.tensor(
        float(self.discriminator.reward_coef), device=self.device
      )
      return obs, mixed, dones, extras

    self.env.step = stepped  # type: ignore[method-assign]

    orig_update = self.alg.update

    def update_with_disc(*args, **kwargs):
      out = orig_update(*args, **kwargs)
      amp_stats = self._update_discriminator()
      if isinstance(out, dict):
        out.update(amp_stats)
      return out

    self.alg.update = update_with_disc  # type: ignore[method-assign]

  def _update_discriminator(self) -> dict[str, float]:
    if self.amp_storage.num_samples < 2:
      return {}
    batch = (
      self.env.num_envs * int(self.cfg["num_steps_per_env"]) // self._disc_minibatches
    )
    batch = max(batch, 1)
    policy_gen = self.amp_storage.feed_forward_generator(self._disc_minibatches, batch)
    expert_gen = self.amp_data.feed_forward_generator(self._disc_minibatches, batch)

    mean_amp = 0.0
    mean_gp = 0.0
    mean_pol = 0.0
    mean_exp = 0.0
    n = 0
    for sample_policy, sample_expert in zip(policy_gen, expert_gen, strict=False):
      policy_state, policy_next = sample_policy
      expert_state, expert_next = sample_expert
      with torch.no_grad():
        policy_state = self.amp_normalizer(policy_state)
        policy_next = self.amp_normalizer(policy_next)
        expert_state = self.amp_normalizer(expert_state)
        expert_next = self.amp_normalizer(expert_next)
      amp_loss, grad_pen, policy_d, expert_d = self.discriminator.discriminator_loss(
        policy_state,
        policy_next,
        expert_state,
        expert_next,
        loss_function=self._disc_loss,
        grad_pen_coef=self._disc_gp,
      )
      loss = amp_loss + grad_pen
      self.discriminator_optimizer.zero_grad()
      loss.backward()
      self.discriminator_optimizer.step()
      mean_amp += float(amp_loss.item())
      mean_gp += float(grad_pen.item())
      mean_pol += float(policy_d.mean().item())
      mean_exp += float(expert_d.mean().item())
      n += 1
    if n == 0:
      return {}
    return {
      "Loss/amp": mean_amp / n,
      "Loss/amp_grad": mean_gp / n,
      "Discriminator/policy_pred": mean_pol / n,
      "Discriminator/expert_pred": mean_exp / n,
    }

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["amp"] = {
      "discriminator": self.discriminator.state_dict(),
      "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
      "amp_normalizer": self.amp_normalizer.state_dict(),
    }
    torch.save(payload, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    amp = loaded.get("amp")
    if amp is None:
      print(f"[AMP] checkpoint {path} has no discriminator weights; using random init")
      return infos
    self.discriminator.load_state_dict(amp["discriminator"])
    self.discriminator_optimizer.load_state_dict(amp["discriminator_optimizer"])
    self.amp_normalizer.load_state_dict(amp["amp_normalizer"])
    return infos
