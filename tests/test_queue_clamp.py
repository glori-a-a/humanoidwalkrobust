"""Unit tests for test-time queue clamp (no Isaac)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsl_rl.modules.delay_queue import (
    MAX_DELAY,
    clamp_queue_block,
    queue_actor_obs_dim,
)


def _fake_obs(batch, num_actions, delay_val=8):
    qdim = queue_actor_obs_dim(num_actions, MAX_DELAY)
    base = torch.randn(batch, 40)
    flat = torch.ones(batch, MAX_DELAY * num_actions)
    mask = torch.ones(batch, MAX_DELAY)
    delay = torch.full((batch, 1), float(delay_val) / MAX_DELAY)
    return torch.cat([base, flat, mask, delay], dim=-1), qdim


def test_full_unchanged():
    obs, _ = _fake_obs(2, 4)
    out = clamp_queue_block(obs, 4, "full")
    assert torch.allclose(out, obs)


def test_clamp_delay():
    obs, qdim = _fake_obs(2, 4, delay_val=8)
    out = clamp_queue_block(obs, 4, "clamp_delay", train_max_delay=5)
    assert torch.allclose(out[:, -1], torch.full((2,), 5.0 / 8.0))
    # base + queue actions/mask untouched except last scalar
    assert torch.allclose(out[:, :-1], obs[:, :-1])


def test_keep5():
    A = 3
    obs, qdim = _fake_obs(1, A, delay_val=8)
    out = clamp_queue_block(obs, A, "keep5", train_max_delay=5)
    q = out[:, -qdim:]
    flat_w = MAX_DELAY * A
    # first 5 action slots stay 1
    assert torch.allclose(q[0, : 5 * A], torch.ones(5 * A))
    # slots 5..7 zeroed
    assert torch.allclose(q[0, 5 * A : flat_w], torch.zeros((MAX_DELAY - 5) * A))
    assert torch.allclose(q[0, flat_w + 5 : flat_w + MAX_DELAY], torch.zeros(MAX_DELAY - 5))


def test_zero_queue():
    obs, qdim = _fake_obs(2, 4)
    base = obs[:, :-qdim].clone()
    out = clamp_queue_block(obs, 4, "zero_queue")
    assert torch.allclose(out[:, :-qdim], base)
    assert torch.allclose(out[:, -qdim:], torch.zeros(2, qdim))


if __name__ == "__main__":
    test_full_unchanged()
    test_clamp_delay()
    test_keep5()
    test_zero_queue()
    print("clamp ablation unit tests passed")
