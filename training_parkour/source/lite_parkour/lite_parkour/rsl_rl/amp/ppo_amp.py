# External AMP algorithm for rsl-rl-lib 5.4.1.
# Adapted from lab_dev/rsl_rl@feature/amp (rsl_rl/algorithms/ppo_amp.py).
# Selected via config: algorithm.class_name = "lite_parkour.rsl_rl.amp.ppo_amp:PPOAMP"

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups

from .circular_buffer import CircularBuffer
from .discriminator import AMPDiscriminator, LossType, resolve_amp_config


def _humanoid_joint_mirror(joint_names: tuple[str, ...]) -> tuple[list[int], list[float]]:
    """Left-right mirror permutation + signs for a humanoid joint block.

    Mirrors the same convention as the task-side policy augmentation
    (``tasks.amp.mdp.symmetry.humanoid.joint_mirror``): each joint is
    sourced from its ``left_``/``right_`` partner, and ``_roll_joint`` / ``_yaw_joint``
    values flip sign under the reflection. ``mirrored[k] = signs[k] * data[indices[k]]``.
    """
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    indices: list[int] = []
    signs: list[float] = []
    for name in joint_names:
        if name.startswith("left_"):
            source = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            source = "left_" + name[len("right_") :]
        else:
            source = name
        if source not in name_to_index:
            raise ValueError(f"Discriminator symmetry: no mirror partner for joint '{name}' (expected '{source}').")
        indices.append(name_to_index[source])
        signs.append(-1.0 if ("_roll_joint" in name or "_yaw_joint" in name) else 1.0)
    return indices, signs


def build_disc_mirror(env: VecEnv, group_names: list[str], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a per-step left-right mirror (gather index + sign) for a discriminator obs group set.

    The returned ``(index, sign)`` apply to the last (feature) dim of a disc-obs tensor shaped
    ``(batch, steps, disc_obs_dim)`` as ``mirrored = obs.index_select(-1, index) * sign``. Terms are
    matched by name: angular-velocity terms flip as a pseudovector ``(-x, y, -z)``; joint position /
    velocity terms use the humanoid joint mirror; ``root_local_rot_tan_norm`` (6D tangent+normal)
    flips its y components. Any other term raises -- add a rule here rather than mirror it wrongly.
    """
    unwrapped = getattr(env, "unwrapped", env)
    obs_manager = unwrapped.observation_manager
    action_cfg = unwrapped.cfg.actions.joint_pos
    if getattr(action_cfg, "preserve_order", False):
        joint_names = tuple(action_cfg.joint_names)
    else:
        joint_names = tuple(unwrapped.scene["robot"].joint_names)
    joint_indices, joint_signs = _humanoid_joint_mirror(joint_names)

    index: list[int] = []
    sign: list[float] = []
    for group_name in group_names:
        for term_name in obs_manager.active_terms[group_name]:
            offset = len(index)
            if "ang_vel" in term_name:
                index += [offset + 0, offset + 1, offset + 2]
                sign += [-1.0, 1.0, -1.0]
            elif "joint_pos" in term_name or "joint_vel" in term_name:
                index += [offset + i for i in joint_indices]
                sign += list(joint_signs)
            elif "rot_tan_norm" in term_name:
                index += [offset + i for i in range(6)]
                sign += [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
            else:
                raise ValueError(
                    f"Discriminator symmetry: no mirror rule for observation term '{term_name}'. "
                    "Add one in build_disc_mirror() or disable disc_symmetry_augmentation."
                )
    return (
        torch.tensor(index, dtype=torch.long, device=device),
        torch.tensor(sign, dtype=torch.float32, device=device),
    )


class PPOAMP(PPO):
    """PPO with Adversarial Motion Priors discriminator.

    This class extends the upstream ``PPO`` algorithm (rsl-rl-lib ≥ 5.4) with:
    - An AMP discriminator that scores how closely agent behaviour matches demonstrations.
    - Separate discriminator replay buffers (agent obs and demo obs).
    - A blended reward: ``task_style_lerp * task_reward + (1 - lerp) * style_reward``.
    - An independent discriminator optimizer.

    Selection:
        Set ``algorithm.class_name = "lite_parkour.rsl_rl.amp.ppo_amp:PPOAMP"`` in the runner cfg.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # AMP-specific (set by construct_algorithm before calling __init__)
        disc_obs_buffer: CircularBuffer,
        disc_demo_obs_buffer: CircularBuffer,
        amp_cfg: dict | None = None,
        # Standard PPO params (forwarded to super)
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        if amp_cfg is None:
            raise ValueError("PPOAMP requires 'amp_cfg' in the algorithm config.")
        self.amp_cfg = amp_cfg

        # Map loss-type string to enum
        _loss_map = {"GAN": LossType.GAN, "LSGAN": LossType.LSGAN, "WGAN": LossType.WGAN}
        loss_type_str = amp_cfg.get("loss_type", "LSGAN")
        if loss_type_str not in _loss_map:
            raise ValueError(f"Unknown AMP loss type '{loss_type_str}'. Choose from {list(_loss_map)}.")
        self.loss_type = _loss_map[loss_type_str]

        # Build the discriminator
        self.amp_discriminator = AMPDiscriminator(
            disc_obs_dim=amp_cfg["disc_obs_dim"],
            disc_obs_steps=amp_cfg["disc_obs_steps"],
            obs_groups=amp_cfg["obs_groups"],
            loss_type=self.loss_type,
            device=device,
            distributed=self.is_multi_gpu,
            **amp_cfg.get("amp_discriminator", {}),
        ).to(device)

        # Discriminator optimizer (separate from PPO optimizer)
        disc_params = [
            {
                "name": "disc_trunk",
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": amp_cfg.get("disc_trunk_weight_decay", 1e-4),
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": amp_cfg.get("disc_linear_weight_decay", 1e-1),
            },
        ]
        self.disc_optimizer = optim.Adam(disc_params, lr=amp_cfg.get("disc_learning_rate", 1e-5))
        self.disc_max_grad_norm = amp_cfg.get("disc_max_grad_norm", 0.5)
        self.disc_update_interval = amp_cfg.get("disc_update_interval", 1)

        # AMP replay buffers (pre-built by construct_algorithm)
        self.disc_obs_buffer = disc_obs_buffer
        self.disc_demo_obs_buffer = disc_demo_obs_buffer

        # Left-right mirror tensors for discriminator symmetry augmentation.
        # Populated by construct_algorithm when amp_cfg["disc_symmetry_augmentation"] is set;
        # left as None (a no-op) otherwise.
        self._disc_mirror: tuple[torch.Tensor, torch.Tensor] | None = None
        self._disc_demo_mirror: tuple[torch.Tensor, torch.Tensor] | None = None

        # Logging scratch pads
        self.style_rewards: torch.Tensor | None = None
        self.disc_score: torch.Tensor | None = None
        self.rewards_lerp: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Factory (overrides PPO.construct_algorithm)
    # ------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "PPOAMP":
        """Build PPOAMP from the runner config dict.

        Called by ``OnPolicyRunner.__init__`` after resolving ``class_name``.
        """
        # 1. Resolve class callables
        alg_class: type[PPOAMP] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        # 2. Resolve obs groups (adds "actor"/"critic" defaults)
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # 3. Resolve RND and symmetry extensions (same as PPO)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # 4. Resolve AMP-specific config (fills disc_obs_dim, disc_obs_steps, step_dt)
        cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        # Also pass obs_groups into amp_cfg so the discriminator can look up group names
        cfg["algorithm"]["amp_cfg"]["obs_groups"] = cfg["obs_groups"]

        # 5. Build actor and critic models
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"[PPOAMP] Actor: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"[PPOAMP] Critic: {critic}")

        # 6. Build rollout storage
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        # 7. Build AMP disc replay buffers (size = one rollout = num_steps_per_env)
        disc_obs_buffer = CircularBuffer(
            max_len=cfg["num_steps_per_env"],
            batch_size=env.num_envs,
            device=device,
        )
        disc_demo_obs_buffer = CircularBuffer(
            max_len=cfg["num_steps_per_env"],
            batch_size=env.num_envs,
            device=device,
        )

        # 8. Instantiate PPOAMP (remaining cfg["algorithm"] keys forwarded as kwargs)
        alg: PPOAMP = alg_class(
            actor=actor,
            critic=critic,
            storage=storage,
            disc_obs_buffer=disc_obs_buffer,
            disc_demo_obs_buffer=disc_demo_obs_buffer,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        # Optional: build left-right mirror operators for the discriminator obs groups.
        if alg.amp_cfg.get("disc_symmetry_augmentation", False):
            obs_groups = cfg["obs_groups"]
            disc_obs_dim = alg.amp_cfg["disc_obs_dim"]
            alg._disc_mirror = build_disc_mirror(env, obs_groups["discriminator"], device)
            alg._disc_demo_mirror = build_disc_mirror(env, obs_groups["discriminator_demonstration"], device)
            for label, (mirror_index, _) in (("disc", alg._disc_mirror), ("disc_demo", alg._disc_demo_mirror)):
                if mirror_index.numel() != disc_obs_dim:
                    raise ValueError(
                        f"Discriminator symmetry: {label} mirror width {mirror_index.numel()} "
                        f"!= disc_obs_dim {disc_obs_dim}."
                    )
            print(f"[PPOAMP] Discriminator symmetry augmentation ON (mirror width {disc_obs_dim}).")

        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    # ------------------------------------------------------------------
    # Env-step hook
    # ------------------------------------------------------------------

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Extract disc obs, compute style reward, buffer data, then delegate to PPO."""
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)

        # Substitute terminal (pre-reset) disc obs for done environments
        if "terminal_obs" in extras:
            terminal_disc_obs = self.amp_discriminator.get_disc_obs(
                extras["terminal_obs"], flatten_history_dim=False
            ).to(disc_obs.device)
            done_mask = dones.bool().to(disc_obs.device)
            if torch.any(done_mask):
                disc_obs = disc_obs.clone()
                disc_obs[done_mask] = terminal_disc_obs[done_mask]

        # Compute style reward and blended reward
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(
            disc_obs, dt=self.amp_cfg["step_dt"]
        )
        self.rewards_lerp = self.amp_discriminator.lerp_reward(
            task_reward=rewards, style_reward=self.style_rewards
        )

        # Buffer un-normalised disc obs for the discriminator update
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)

        # Forward blended rewards to PPO (storage + bootstrapping)
        super().process_env_step(obs, self.rewards_lerp, dones, extras)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Run PPO update followed by a discriminator update; return combined loss dict."""
        # --- 1. PPO update (clears rollout storage at the end) ---
        ppo_loss_dict = super().update()

        # --- 2. Discriminator update over AMP replay buffers ---
        disc_loss_dict = self._update_discriminator()

        return {**ppo_loss_dict, **disc_loss_dict}

    def _update_discriminator(self) -> dict[str, float]:
        """One sweep over the disc obs buffers to update the discriminator."""
        mean_disc_loss = 0.0
        mean_disc_grad_penalty = 0.0
        mean_disc_score = 0.0
        mean_disc_demo_score = 0.0
        disc_updates_done = 0

        num_steps = self.storage.num_transitions_per_env  # type: ignore
        disc_obs_gen = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=num_steps,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_gen = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=num_steps,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        for mini_batch_idx, (disc_obs_batch, disc_demo_obs_batch) in enumerate(
            zip(disc_obs_gen, disc_demo_obs_gen)
        ):
            # disc_obs_batch: (mini_batch_size, disc_obs_steps, disc_obs_dim)
            # Left-right symmetry augmentation: append the mirrored batch so the discriminator
            # (and its normalizer) see a left-right symmetric distribution -> the style reward
            # cannot favour one side. Doubles the effective mini-batch.
            if self._disc_mirror is not None:
                agent_index, agent_sign = self._disc_mirror
                demo_index, demo_sign = self._disc_demo_mirror
                disc_obs_batch = torch.cat(
                    [disc_obs_batch, disc_obs_batch.index_select(-1, agent_index) * agent_sign], dim=0
                )
                disc_demo_obs_batch = torch.cat(
                    [disc_demo_obs_batch, disc_demo_obs_batch.index_select(-1, demo_index) * demo_sign], dim=0
                )

            with torch.no_grad():
                normed_agent = self.amp_discriminator.normalize_disc_obs(disc_obs_batch)
                normed_demo = self.amp_discriminator.normalize_disc_obs(disc_demo_obs_batch)

            mb = normed_agent.shape[0]
            agent_flat = normed_agent.reshape(mb, -1)
            demo_flat = normed_demo.reshape(mb, -1)

            disc_score = self.amp_discriminator(agent_flat)
            disc_demo_score = self.amp_discriminator(demo_flat)

            if self.loss_type == LossType.GAN:
                bce = nn.BCEWithLogitsLoss()
                disc_loss = 0.5 * (
                    bce(disc_score, torch.zeros_like(disc_score))
                    + bce(disc_demo_score, torch.ones_like(disc_demo_score))
                )
            elif self.loss_type == LossType.LSGAN:
                mse = nn.MSELoss()
                disc_loss = 0.5 * (
                    mse(disc_score, -torch.ones_like(disc_score))
                    + mse(disc_demo_score, torch.ones_like(disc_demo_score))
                )
            elif self.loss_type == LossType.WGAN:
                disc_loss = -torch.mean(disc_demo_score) + torch.mean(disc_score)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            grad_penalty = self.amp_discriminator.compute_grad_penalty(
                demo_data=demo_flat,
                scale=self.amp_cfg.get("grad_penalty_scale", 40.0),
            )
            total_disc_loss = disc_loss + grad_penalty

            # Only step the discriminator every disc_update_interval mini-batches.
            # This slows the discriminator relative to the policy, preventing early saturation.
            do_disc_update = mini_batch_idx % self.disc_update_interval == 0

            if do_disc_update:
                self.disc_optimizer.zero_grad()
                total_disc_loss.backward()
                if self.is_multi_gpu:
                    self._reduce_discriminator_gradients()
                nn.utils.clip_grad_norm_(
                    self.amp_discriminator.parameters(), self.disc_max_grad_norm
                )
                self.disc_optimizer.step()
                # Update observation normaliser with un-normalised data
                self.amp_discriminator.update_normalization(disc_obs_batch)

            # scores logged every step (saturation tracking); losses only on update steps
            mean_disc_score += disc_score.mean().item()
            mean_disc_demo_score += disc_demo_score.mean().item()
            if do_disc_update:
                mean_disc_loss += disc_loss.item()
                mean_disc_grad_penalty += grad_penalty.item()
                disc_updates_done += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        # disc scores accumulated every step; losses only on actual update steps
        result = {
            "amp/disc_score": mean_disc_score / num_updates,
            "amp/disc_demo_score": mean_disc_demo_score / num_updates,
        }
        if disc_updates_done > 0:
            result["amp/disc_loss"] = mean_disc_loss / disc_updates_done
            result["amp/disc_grad_penalty"] = mean_disc_grad_penalty / disc_updates_done
        else:
            result["amp/disc_loss"] = 0.0
            result["amp/disc_grad_penalty"] = 0.0
        if self.is_multi_gpu:
            metrics = torch.tensor(list(result.values()), device=self.device)
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            result = dict(zip(result, (metrics / self.gpu_world_size).tolist()))
        return result

    # ------------------------------------------------------------------
    # Distributed training
    # ------------------------------------------------------------------

    def broadcast_parameters(self) -> None:
        """Synchronize PPO and AMP parameters/buffers before the first rollout.

        OnPolicyRunner calls this after checkpoint loading, so both fresh and
        resumed runs start with the same discriminator and normalization state.
        """
        super().broadcast_parameters()
        for tensor in self.amp_discriminator.state_dict().values():
            torch.distributed.broadcast(tensor, src=0)

    def _reduce_discriminator_gradients(self) -> None:
        """Average discriminator gradients before clipping and the Adam step.

        Like PPO, all ranks use the same rollout/minibatch sizes and schedule.
        Replay data stays local; only the gradients are communicated.
        """
        grads = [param.grad for param in self.amp_discriminator.parameters() if param.grad is not None]
        flat = torch.cat([grad.flatten() for grad in grads])
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for grad in grads:
            grad.copy_(flat[offset : offset + grad.numel()].view_as(grad))
            offset += grad.numel()

    # ------------------------------------------------------------------
    # Train / eval mode
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        super().train_mode()
        self.amp_discriminator.train()
        self.amp_discriminator.disc_obs_normalizer.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.amp_discriminator.eval()
        self.amp_discriminator.disc_obs_normalizer.eval()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self) -> dict:
        saved = super().save()
        saved["amp_discriminator_state_dict"] = self.amp_discriminator.state_dict()
        saved["amp_discriminator_normalizer_state_dict"] = (
            self.amp_discriminator.disc_obs_normalizer.state_dict()
        )
        saved["amp_discriminator_optimizer_state_dict"] = self.disc_optimizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "amp_discriminator_state_dict" in loaded_dict:
            self.amp_discriminator.load_state_dict(
                loaded_dict["amp_discriminator_state_dict"], strict=strict
            )
        if "amp_discriminator_normalizer_state_dict" in loaded_dict:
            self.amp_discriminator.disc_obs_normalizer.load_state_dict(
                loaded_dict["amp_discriminator_normalizer_state_dict"]
            )
        if "amp_discriminator_optimizer_state_dict" in loaded_dict:
            # Only load if explicitly requested (mirrors PPO optimizer behaviour)
            if load_cfg is None or load_cfg.get("optimizer", True):
                self.disc_optimizer.load_state_dict(
                    loaded_dict["amp_discriminator_optimizer_state_dict"]
                )
        return load_iteration
