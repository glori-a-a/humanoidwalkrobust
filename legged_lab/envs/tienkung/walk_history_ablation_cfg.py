# walk_history_ablation task: history=20 at train + eval, plain ActorCritic
# Installed to TienKung-Lab/legged_lab/envs/tienkung/

from isaaclab.utils import configclass

from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)

HISTORY_LEN = 20


@configclass
class TienKungWalkHistoryAblationEnvCfg(TienKungWalkFlatEnvCfg):
    def __post_init__(self):
        self.robot.actor_obs_history_length = HISTORY_LEN
        self.robot.critic_obs_history_length = HISTORY_LEN


@configclass
class TienKungWalkHistoryAblationAgentCfg(TienKungWalkAgentCfg):
    experiment_name = "walk_history_ablation"
