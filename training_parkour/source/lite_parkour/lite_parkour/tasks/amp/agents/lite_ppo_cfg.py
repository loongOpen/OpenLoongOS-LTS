from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

from lite_parkour.rsl_rl import RslRlAmpCfg, RslRlDepthCNNModelCfg, RslRlPpoAmpAlgorithmCfg
from lite_parkour.tasks.amp.mdp.symmetry import humanoid as humanoid_symmetry


def _depth_cnn_model(stochastic: bool) -> RslRlDepthCNNModelCfg:
    """MLP over proprioception plus a small CNN encoder for the 8-frame depth stack."""
    return RslRlDepthCNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlDepthCNNModelCfg.GaussianDistributionCfg(init_std=1.0) if stochastic else None,
        cnn_cfg=RslRlDepthCNNModelCfg.CNNCfg(
            output_channels=[4], kernel_size=[3], stride=[1], padding="zeros", activation="relu", max_pool=False
        ),
        encoder_hidden_dims=[256, 256],
        encoder_output_dim=128,
        encoder_activation="relu",
    )


@configclass
class LiteAmpDepthPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "lite_amp_depth"

    # "discriminator" groups are consumed by PPOAMP directly.
    obs_groups = {
        "actor": ["policy", "depth"],
        "critic": ["critic", "depth"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["disc_demo"],
    }

    actor = _depth_cnn_model(stochastic=True)
    critic = _depth_cnn_model(stochastic=False)

    algorithm = RslRlPpoAmpAlgorithmCfg(
        class_name="lite_parkour.rsl_rl.amp.ppo_amp:PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_cfg=RslRlAmpCfg(
            grad_penalty_scale=40.0,
            disc_trunk_weight_decay=1.0e-4,
            disc_linear_weight_decay=1.0e-1,
            disc_learning_rate=1.0e-5,
            disc_max_grad_norm=1.0,
            disc_update_interval=10,
            disc_symmetry_augmentation=False,
            amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                hidden_dims=[1024, 512],
                activation="elu",
                style_reward_scale=2.0,
                task_style_lerp=0.5,
            ),
            loss_type="LSGAN",
        ),
        # Left-right mirror augmentation and mirror loss on the policy.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=humanoid_symmetry.compute_symmetric_states,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
        ),
    )
