# Policy module for Stage 2: predictor fills newest obs frame, then actor MLP runs.
# Installed to TienKung-Lab/rsl_rl/rsl_rl/modules/

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation


def make_mlp(in_dim, out_dim, hidden_dims, activation):
    layers = []
    width = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(width, h), activation]
        width = h
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class ActorCriticDelayPredictor(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        obs_history_length=10,
        predictor_hidden_dims=None,
        activation="elu",
        **kwargs,
    ):
        if predictor_hidden_dims is None:
            predictor_hidden_dims = [256, 256]

        self.history_len = int(obs_history_length)
        self.frame_dim = num_actor_obs // self.history_len
        if self.frame_dim * self.history_len != num_actor_obs:
            raise ValueError(
                f"num_actor_obs={num_actor_obs} not divisible by history_len={self.history_len}"
            )

        act = resolve_nn_activation(activation)
        print(
            f"[DelayCompPolicy] history={self.history_len}, "
            f"frame_dim={self.frame_dim}, hidden={predictor_hidden_dims}"
        )
        super().__init__(num_actor_obs, num_critic_obs, num_actions, activation=activation, **kwargs)
        self.predictor = make_mlp(num_actor_obs, self.frame_dim, predictor_hidden_dims, act)

    def actor_input(self, observations):
        # replace last history frame with predictor output
        predicted = self.predictor(observations)
        frames = observations.view(-1, self.history_len, self.frame_dim).clone()
        frames[:, -1, :] = predicted
        return frames.reshape(observations.shape[0], -1)

    def update_distribution(self, observations):
        mean = self.actor(self.actor_input(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act_inference(self, observations):
        return self.actor(self.actor_input(observations))
