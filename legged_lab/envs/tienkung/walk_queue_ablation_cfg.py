# walk_queue_ablation: explicit queue/delay in actor obs, no dynamics predictor (v2a ablation)
# Installed to TienKung-Lab/legged_lab/envs/tienkung/

from isaaclab.utils import configclass

from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.base.base_env_config import RslRlPpoActorCriticCfg


@configclass
class TienKungWalkQueueAblationEnvCfg(TienKungWalkFlatEnvCfg):
    enable_queue_actor_obs: bool = True


@configclass
class TienKungWalkQueueAblationAgentCfg(TienKungWalkAgentCfg):
    experiment_name = "walk_queue_ablation"

    policy = RslRlPpoActorCriticCfg(
        class_name="PredictorFeedbackActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
