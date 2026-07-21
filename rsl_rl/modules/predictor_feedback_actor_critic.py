# Shared actor for v2: queue-aware ablation (use_predictor=False) and
# Smith-inspired predictor-feedback main method (use_predictor=True, v2b).
# Installed to TienKung-Lab/rsl_rl/rsl_rl/modules/

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.delay_queue import MAX_DELAY, queue_actor_obs_dim


class PredictorFeedbackActorCritic(ActorCritic):
    """Unified v2 actor-critic. Critic unchanged; actor may add predictor path later."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        use_predictor: bool = False,
        max_delay: int = MAX_DELAY,
        future_encoder_dims=None,
        **kwargs,
    ):
        self.use_predictor = bool(use_predictor)
        self.max_delay = int(max_delay)
        self.num_actions = int(num_actions)
        self.queue_obs_dim = queue_actor_obs_dim(self.num_actions, self.max_delay)
        self.base_actor_obs_dim = int(num_actor_obs) - self.queue_obs_dim

        if self.base_actor_obs_dim <= 0:
            raise ValueError(
                f"num_actor_obs={num_actor_obs} too small for queue dim={self.queue_obs_dim}"
            )

        super().__init__(num_actor_obs, num_critic_obs, num_actions, **kwargs)

        if self.use_predictor:
            if future_encoder_dims is None:
                future_encoder_dims = [128, 64]
            layers = []
            width = 0  # set after state_dim known in v2b
            self._future_encoder_dims = future_encoder_dims
            self.future_encoder = None  # built when dynamics state_dim is wired (v2b)
            self._dynamics_model = None

    @staticmethod
    def warm_start_actor_from_strong(
        actor_critic: "PredictorFeedbackActorCritic",
        checkpoint: dict,
        device: torch.device,
    ) -> int:
        """Copy Strong DR actor weights; zero-init new queue (and future) input columns."""
        if "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
        else:
            sd = checkpoint

        own = actor_critic.state_dict()
        base_dim = actor_critic.base_actor_obs_dim
        loaded = 0

        actor_w_key = "actor.0.weight"
        if actor_w_key in sd and actor_w_key in own:
            old_w = sd[actor_w_key].to(device)
            new_w = own[actor_w_key].clone()
            cols = min(old_w.shape[1], base_dim)
            new_w[:, :cols] = old_w[:, :cols]
            own[actor_w_key] = new_w
            loaded += 1

        for key, val in sd.items():
            if key == actor_w_key:
                continue
            if key.startswith("actor.") and key in own and own[key].shape == val.shape:
                own[key] = val.to(device)
                loaded += 1
            elif key.startswith("critic.") and key in own and own[key].shape == val.shape:
                own[key] = val.to(device)
                loaded += 1
            elif key == "std" and key in own and own[key].shape == val.shape:
                own[key] = val.to(device)
                loaded += 1

        actor_critic.load_state_dict(own, strict=False)
        return loaded

    def set_dynamics_model(self, model: nn.Module):
        """v2b: attach frozen residual dynamics for rollout inside actor_input."""
        self._dynamics_model = model
        if self.use_predictor and self.future_encoder is None:
            raise NotImplementedError("future_encoder wiring is part of v2b (not MVP).")

    def actor_input(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.use_predictor:
            return observations
        raise NotImplementedError(
            "use_predictor=True requires v2b dynamics rollout (not implemented in queue MVP)."
        )
