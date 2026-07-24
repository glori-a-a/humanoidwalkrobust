# walk_predictor_feedback: main v2 method (Smith-inspired); MVP uses queue obs only until v2b
# Installed to TienKung-Lab/legged_lab/envs/tienkung/

from isaaclab.utils import configclass

from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.base.base_env_config import RslRlPpoActorCriticCfg


@configclass
class TienKungWalkPredictorFeedbackEnvCfg(TienKungWalkFlatEnvCfg):
    enable_queue_actor_obs: bool = True


@configclass
class TienKungWalkPredictorFeedbackAgentCfg(TienKungWalkAgentCfg):
    experiment_name = "walk_predictor_feedback"

    policy = RslRlPpoActorCriticCfg(
        class_name="PredictorFeedbackActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
