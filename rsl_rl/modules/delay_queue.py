"""Action-delay queue helpers (Phase 0 canonical rule, shared by env + tests + policy).

Semantics at decision time t, BEFORE appending u_t:
  pre_execution_actions[i] = u_{t-d+i}  for i in 0..d-1  (oldest-first)
  pre_execution_actions[0] == applied action u_{t-d} when d > 0
  u_t is NOT included in the queue
"""
from __future__ import annotations

import torch

MAX_DELAY = 8


def extract_pre_execution(
    ring: torch.Tensor,
    delay: torch.Tensor,
    max_delay: int = MAX_DELAY,
):
    """Extract padded pre-execution action queue from issued-action ring.

    Args:
        ring: [B, L, A] issued actions, newest at index L-1, L >= max_delay.
        delay: [B] long, per-env delay in steps (0 <= d <= max_delay).
        max_delay: fixed queue width.

    Returns:
        pre_exec: [B, max_delay, A], zero-padded.
        mask: [B, max_delay] bool, True where i < d.
    """
    B, L, A = ring.shape
    device = ring.device
    ar = torch.arange(max_delay, device=device)
    mask = ar[None, :] < delay[:, None]
    idx = (L - delay[:, None] + ar[None, :]).clamp(0, L - 1)
    idx_exp = idx[..., None].expand(B, max_delay, A)
    gathered = torch.gather(ring, 1, idx_exp)
    pre_exec = gathered * mask[..., None].to(ring.dtype)
    return pre_exec, mask


def queue_actor_obs_dim(num_actions: int, max_delay: int = MAX_DELAY) -> int:
    """Extra actor dims: flat(pre_exec) + mask + scalar delay_norm."""
    return max_delay * num_actions + max_delay + 1


def build_queue_actor_obs(
    ring: torch.Tensor,
    delay: torch.Tensor,
    max_delay: int = MAX_DELAY,
) -> torch.Tensor:
    """Build queue observation block for actor input (decision-time ring)."""
    pre_exec, mask = extract_pre_execution(ring, delay, max_delay)
    B = ring.shape[0]
    flat_pre = pre_exec.reshape(B, -1)
    mask_f = mask.to(dtype=ring.dtype)
    delay_norm = (delay.to(dtype=ring.dtype) / float(max_delay)).unsqueeze(-1)
    return torch.cat([flat_pre, mask_f, delay_norm], dim=-1)


def append_queue_to_actor_obs(
    actor_obs: torch.Tensor,
    ring: torch.Tensor,
    delay: torch.Tensor,
    max_delay: int = MAX_DELAY,
) -> torch.Tensor:
    """Concatenate base actor obs with queue block."""
    queue_obs = build_queue_actor_obs(ring, delay, max_delay)
    return torch.cat([actor_obs, queue_obs], dim=-1)


# Queue block layout at the end of actor_obs:
#   [flat_pre (max_delay * A) | mask (max_delay) | delay_norm (1)]
CLAMP_MODES = ("full", "clamp_delay", "keep5", "zero_queue")


def clamp_queue_block(
    actor_obs: torch.Tensor,
    num_actions: int,
    mode: str,
    max_delay: int = MAX_DELAY,
    train_max_delay: int = 5,
) -> torch.Tensor:
    """Test-time edit of the queue suffix (does not change the real sim delay).

    Modes:
      full         — no change (true queue + true delay)
      clamp_delay  — delay_norm = min(d, train_max) / max_delay
      keep5        — only first train_max slots kept; later slots zeroed
      zero_queue   — wipe entire queue block (base obs only)
    """
    if mode == "full":
        return actor_obs
    if mode not in CLAMP_MODES:
        raise ValueError(f"unknown clamp mode: {mode}")

    qdim = queue_actor_obs_dim(num_actions, max_delay)
    out = actor_obs.clone()
    q = out[:, -qdim:]
    flat_w = max_delay * num_actions

    if mode == "zero_queue":
        q.zero_()
        return out

    if mode == "clamp_delay":
        # last scalar is delay / max_delay; clamp to train_max / max_delay
        cap = float(train_max_delay) / float(max_delay)
        q[:, -1] = q[:, -1].clamp(max=cap)
        return out

    if mode == "keep5":
        # zero action slots and mask for i >= train_max_delay
        for i in range(train_max_delay, max_delay):
            q[:, i * num_actions : (i + 1) * num_actions] = 0.0
            q[:, flat_w + i] = 0.0
        return out

    return out


class IssuedActionRing:
    """Self-maintained issued-action ring (newest at index L-1)."""

    def __init__(self, batch: int, length: int, action_dim: int, device: torch.device):
        self.length = length
        self.action_dim = action_dim
        self.buf = torch.zeros(batch, length, action_dim, device=device)

    def append(self, actions: torch.Tensor):
        self.buf = torch.roll(self.buf, shifts=-1, dims=1)
        self.buf[:, -1, :] = actions

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() > 0:
            self.buf[env_ids] = 0.0

    @property
    def ring(self) -> torch.Tensor:
        return self.buf
