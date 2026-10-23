"""AMP discriminator (style reward + expert/policy classification)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class AmpDiscriminator(nn.Module):
  """MLP discriminator over concatenated (s, s_next) AMP observations."""

  def __init__(
    self,
    observation_dim: int,
    hidden_dims: Sequence[int] = (1024, 512, 256),
    reward_coef: float = 0.3,
    task_reward_lerp: float = 0.5,
    style_reward_function: str = "quad_mapping",
  ) -> None:
    super().__init__()
    self.observation_dim = int(observation_dim)
    self.input_dim = self.observation_dim * 2
    self.reward_coef = float(reward_coef)
    self.task_reward_lerp = float(task_reward_lerp)
    self.style_reward_function = style_reward_function

    layers: list[nn.Module] = []
    curr = self.input_dim
    for hidden in hidden_dims:
      layers.append(nn.Linear(curr, hidden))
      layers.append(nn.ReLU())
      curr = hidden
    layers.append(nn.Linear(curr, 1))
    self.architecture = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.architecture(x)

  def compute_grad_pen(
    self,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
    lambda_: float = 10.0,
  ) -> torch.Tensor:
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
    expert_data.requires_grad_(True)
    disc = self.architecture(expert_data)
    ones = torch.ones_like(disc)
    grad = autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=ones,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()

  def compute_wgan_pen(
    self,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
    policy_state: torch.Tensor,
    policy_next_state: torch.Tensor,
    lambda_: float = 10.0,
  ) -> torch.Tensor:
    with torch.no_grad():
      expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
      policy_data = torch.cat([policy_state, policy_next_state], dim=-1)
      lerp = torch.rand_like(expert_data)
      lerped = lerp * expert_data + (1.0 - lerp) * policy_data
    lerped.requires_grad_(True)
    disc = self.architecture(lerped)
    grad = autograd.grad(
      outputs=disc,
      inputs=lerped,
      grad_outputs=torch.ones_like(disc),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return lambda_ * (grad.norm(2, dim=1) - 1.0).pow(2).mean()

  def predict_amp_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    state_normalizer: nn.Module | None = None,
    task_reward_lerp: torch.Tensor | float | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix style reward with task reward. Returns (total, style)."""
    with torch.no_grad():
      was_training = self.training
      self.eval()
      if state_normalizer is not None:
        was_norm_training = state_normalizer.training
        state_normalizer.eval()
        state = state_normalizer(state)
        next_state = state_normalizer(next_state)
        state_normalizer.train(was_norm_training)

      d = self.architecture(torch.cat([state, next_state], dim=-1))
      if self.style_reward_function == "quad_mapping":
        style = self.reward_coef * torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)
      elif self.style_reward_function == "log_mapping":
        style = -torch.log(
          torch.clamp(1 - torch.sigmoid(d), min=1e-4)
        )
      elif self.style_reward_function == "wasserstein_mapping":
        style = self.reward_coef * d
      else:
        raise ValueError(f"Unknown style_reward_function: {self.style_reward_function}")

      lerp = self.task_reward_lerp if task_reward_lerp is None else task_reward_lerp
      if torch.is_tensor(lerp) and lerp.ndim == 1:
        lerp = lerp.unsqueeze(-1)
      total = (1.0 - lerp) * style + lerp * task_reward.unsqueeze(-1)
      self.train(was_training)
    return total.squeeze(-1), style.squeeze(-1)

  def discriminator_loss(
    self,
    policy_state: torch.Tensor,
    policy_next_state: torch.Tensor,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
    loss_function: str,
    grad_pen_coef: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_d = self(torch.cat([policy_state, policy_next_state], dim=-1))
    expert_d = self(torch.cat([expert_state, expert_next_state], dim=-1))

    if loss_function == "MSELoss":
      expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
      policy_loss = F.mse_loss(policy_d, -torch.ones_like(policy_d))
      amp_loss = 0.5 * (expert_loss + policy_loss)
      grad_pen = self.compute_grad_pen(
        expert_state, expert_next_state, lambda_=grad_pen_coef
      )
    elif loss_function == "PairwiseLoss":
      amp_loss = F.softplus(-(expert_d - policy_d)).mean()
      grad_pen = self.compute_wgan_pen(
        expert_state,
        expert_next_state,
        policy_state,
        policy_next_state,
        lambda_=grad_pen_coef,
      )
    elif loss_function == "BCEWithLogitsLoss":
      expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones_like(expert_d))
      policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros_like(policy_d))
      amp_loss = 0.5 * (expert_loss + policy_loss)
      grad_pen = self.compute_grad_pen(
        expert_state, expert_next_state, lambda_=grad_pen_coef
      )
    else:
      raise ValueError(f"Unknown discriminator loss: {loss_function}")

    return amp_loss, grad_pen, policy_d.detach(), expert_d.detach()
