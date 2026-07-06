# walk_delay_comp task: same walk env, policy class = ActorCriticDelayPredictor
# Installed to TienKung-Lab/legged_lab/envs/tienkung/

from isaaclab.utils import configclass

from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.base.base_env_config import RslRlPpoActorCriticCfg

TienKungWalkDelayCompEnvCfg = TienKungWalkFlatEnvCfg


@configclass
class TienKungWalkDelayCompAgentCfg(TienKungWalkAgentCfg):
    experiment_name = "walk_delay_comp"

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticDelayPredictor",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
