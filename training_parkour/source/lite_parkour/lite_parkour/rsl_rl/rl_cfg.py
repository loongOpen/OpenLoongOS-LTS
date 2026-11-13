from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .amp_cfg import RslRlAmpCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticConv2dCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with convolutional layers."""

    class_name: str = "ActorCriticConv2d"
    """The policy class name. Default is ActorCriticConv2d."""

    conv_layers_params: list[dict] = [
        {"out_channels": 4, "kernel_size": 3, "stride": 2},
        {"out_channels": 8, "kernel_size": 3, "stride": 2},
        {"out_channels": 16, "kernel_size": 3, "stride": 2},
    ]
    """List of convolutional layer parameters for the convolutional network."""

    conv_linear_output_size: int = 16
    """Output size of the linear layer after the convolutional features are flattened."""


@configclass
class RslRlDepthCNNModelCfg(RslRlCNNModelCfg):
    """RSL-RL 5.x model configuration with a compact depth-image encoder."""

    class_name: str = "lite_parkour.rsl_rl.depth_cnn_model:DepthCNNModel"

    encoder_hidden_dims: list[int] = [256, 256]
    """Hidden dimensions of the MLP after the convolutional stack."""

    encoder_output_dim: int = 128
    """Size of the compact image feature concatenated with 1-D observations."""

    encoder_activation: str = "relu"
    """Activation used by the image encoder's MLP head."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMP algorithm.

    The default class_name resolves to ``lite_parkour.rsl_rl.amp.ppo_amp.PPOAMP`` via
    ``rsl_rl.utils.resolve_callable`` — no fork of rsl_rl is required.
    """

    class_name: str = "lite_parkour.rsl_rl.amp.ppo_amp:PPOAMP"
    """The algorithm class name resolved by resolve_callable. Default is the external PPOAMP."""

    amp_cfg: RslRlAmpCfg = RslRlAmpCfg()
    """Configuration for the AMP (Adversarial Motion Priors) in the training."""


#########################
# Runner configurations #
#########################
