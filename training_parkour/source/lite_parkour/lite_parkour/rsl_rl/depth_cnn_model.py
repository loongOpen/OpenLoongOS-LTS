"""RSL-RL model with a dedicated CNN/MLP encoder for depth-image observations."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import CNNModel
from rsl_rl.modules import HiddenState, MLP


class DepthCNNModel(CNNModel):
    """CNN model that projects every image group to a fixed-size latent.

    Stock RSL-RL ``CNNModel`` concatenates the flattened convolution output
    directly with the 1-D observations. As in InstinctLab, an MLP head first compresses
    it into a compact depth feature, keeping the RSL-RL 5.x model/distribution interface.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        encoder_output_dim: int = 128,
        encoder_activation: str = "relu",
    ) -> None:
        if encoder_output_dim < 1:
            raise ValueError(f"encoder_output_dim must be positive, got {encoder_output_dim}.")
        if not encoder_hidden_dims:
            raise ValueError("encoder_hidden_dims must contain at least one layer.")

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            cnn_cfg=cnn_cfg,
            cnns=cnns,
        )

        self.encoder_output_dim = encoder_output_dim
        self.encoder_heads = nn.ModuleDict(
            {
                obs_group: MLP(
                    input_dim=int(self.cnns[obs_group].output_dim),
                    output_dim=encoder_output_dim,
                    hidden_dims=encoder_hidden_dims,
                    activation=encoder_activation,
                )
                for obs_group in self.obs_groups_2d
            }
        )

        # CNNModel built the policy head for raw flattened CNN features. Rebuild
        # it for the compact encoder latents while keeping its distribution.
        self.cnn_latent_dim = encoder_output_dim * len(self.obs_groups_2d)
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = MLP(self._get_latent_dim(), mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        image_latents = torch.cat(
            [self.encoder_heads[group](self.cnns[group](obs[group])) for group in self.obs_groups_2d], dim=-1
        )
        if not self.obs_groups:
            return image_latents
        proprio_latent = self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1))
        return torch.cat([proprio_latent, image_latents], dim=-1)

    def as_jit(self) -> nn.Module:
        return _TorchDepthCNNModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxDepthCNNModel(self, verbose)


class _TorchDepthCNNModel(nn.Module):
    def __init__(self, model: DepthCNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(copy.deepcopy(model.cnns[group]), copy.deepcopy(model.encoder_heads[group]))
                for group in model.obs_groups_2d
            ]
        )
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        latent_1d = self.obs_normalizer(obs_1d)
        latent_2d = torch.cat([encoder(obs_2d[index]) for index, encoder in enumerate(self.encoders)], dim=-1)
        return self.deterministic_output(self.mlp(torch.cat([latent_1d, latent_2d], dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxDepthCNNModel(nn.Module):
    def __init__(self, model: DepthCNNModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(copy.deepcopy(model.cnns[group]), copy.deepcopy(model.encoder_heads[group]))
                for group in model.obs_groups_2d
            ]
        )
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )
        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        latent_1d = self.obs_normalizer(obs_1d)
        latent_2d = torch.cat([encoder(obs_2d[index]) for index, encoder in enumerate(self.encoders)], dim=-1)
        return self.deterministic_output(self.mlp(torch.cat([latent_1d, latent_2d], dim=-1)))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        images = [torch.zeros(1, channels, *dims) for channels, dims in zip(self.obs_channels_2d, self.obs_dims_2d)]
        return (torch.zeros(1, self.obs_dim_1d), *images)

    @property
    def input_names(self) -> list[str]:
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
