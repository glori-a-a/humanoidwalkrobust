"""Phase 0 verification (pure tensor, no Isaac) for the action-delay queue.

Goal: pin down the *canonical extraction rule* for ``pre_execution_actions``
before any algorithm module is written, and prove it is free of off-by-one
errors at delay = 0, 1, 3 (fixed) and for per-env mixed delays.

Semantics being verified (see also inspect_action_delay_timeline.py, which
checks the same rule against the live Isaac action buffer):

  At decision time t the policy is about to emit u_t. The issued-action ring
  buffer R holds the previously issued actions, newest at index L-1:

      R[:, L-1] = u_{t-1}, R[:, L-2] = u_{t-2}, ...

  With delay d, the actions that will actually be applied over the next d sim
  steps (before u_t first takes effect at step t+d) are:

      pre_execution_actions = [u_{t-d}, u_{t-d+1}, ..., u_{t-1}]   (exactly d)
      pre_execution_actions[0] == applied action at step t (== u_{t-d})
      u_t is NOT included

  Canonical extraction (oldest-first, padded to max_delay, with mask):

      slot i holds the action applied i steps from now:
          pre_exec[:, i] = R[:, L - d + i]   for i < d, else 0
          mask[:, i]     = (i < d)

Run:  python -m pytest tests/test_delay_queue_order.py -v
      (or)  python tests/test_delay_queue_order.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rsl_rl.modules.delay_queue import MAX_DELAY, extract_pre_execution


class NumberedRing:
    """Rolling issued-action buffer mirroring Isaac's CircularBuffer append order.

    Each decision t appends u_t whose value is the scalar decision index t
    (broadcast over the action dimension), so applied/queue values are readable.
    """

    def __init__(self, batch: int, length: int, action_dim: int):
        self.length = length
        self.action_dim = action_dim
        self.buf = torch.zeros(batch, length, action_dim)

    def append(self, value: float):
        self.buf = torch.roll(self.buf, shifts=-1, dims=1)
        self.buf[:, -1, :] = float(value)

    @property
    def ring(self) -> torch.Tensor:
        return self.buf


def _simulate(batch, action_dim, warmup, delay_vec):
    """Warm the ring, then at 'decision t' extract the queue and return facts."""
    ring = NumberedRing(batch, MAX_DELAY, action_dim)
    # Warm up: append issued actions u_0 .. u_{t-1} before decision t.
    t = warmup
    for k in range(t):
        ring.append(k)  # append u_k
    # Now decide u_t (not yet appended): extract queue for this decision.
    pre_exec, mask = extract_pre_execution(ring.ring, delay_vec, MAX_DELAY)
    return t, pre_exec, mask


def test_fixed_delays_0_1_3():
    batch, action_dim, warmup = 2, 4, 20
    for d in (0, 1, 3):
        delay_vec = torch.full((batch,), d, dtype=torch.long)
        t, pre_exec, mask = _simulate(batch, action_dim, warmup, delay_vec)

        # mask: exactly d active slots, oldest-first.
        assert int(mask.sum(dim=1)[0]) == d, f"d={d}: expected {d} active slots"
        assert bool((mask == (torch.arange(MAX_DELAY) < d)[None]).all())

        for i in range(MAX_DELAY):
            if i < d:
                expected = float(t - d + i)  # u_{t-d+i}
                assert torch.allclose(pre_exec[:, i], torch.full((batch, action_dim), expected)), (
                    f"d={d} slot {i}: got {pre_exec[0, i].tolist()}, want {expected}"
                )
            else:
                assert torch.allclose(pre_exec[:, i], torch.zeros(batch, action_dim)), (
                    f"d={d} slot {i}: padding must be zero"
                )

        if d > 0:
            applied = float(t - d)  # u_{t-d}
            assert torch.allclose(pre_exec[:, 0], torch.full((batch, action_dim), applied)), (
                f"d={d}: pre_exec[0] must equal applied action u_(t-d)={applied}"
            )
            assert torch.allclose(pre_exec[:, d - 1], torch.full((batch, action_dim), float(t - 1))), (
                f"d={d}: pre_exec[d-1] must equal newest issued u_(t-1)={t - 1}"
            )


def test_delay_zero_is_identity_empty_queue():
    batch, action_dim, warmup = 3, 5, 12
    delay_vec = torch.zeros(batch, dtype=torch.long)
    _, pre_exec, mask = _simulate(batch, action_dim, warmup, delay_vec)
    assert bool((~mask).all()), "d=0 must have empty queue mask"
    assert torch.allclose(pre_exec, torch.zeros_like(pre_exec)), "d=0 queue must be all zeros"


def test_mixed_per_env_delays():
    """Uniform-mode: different delay per env in the same batch."""
    action_dim, warmup = 3, 25
    delay_vec = torch.tensor([0, 1, 3, 5], dtype=torch.long)
    batch = delay_vec.numel()
    t, pre_exec, mask = _simulate(batch, action_dim, warmup, delay_vec)
    for b, d in enumerate(delay_vec.tolist()):
        assert int(mask[b].sum()) == d
        if d > 0:
            assert torch.allclose(pre_exec[b, 0], torch.full((action_dim,), float(t - d)))
            assert torch.allclose(pre_exec[b, d - 1], torch.full((action_dim,), float(t - 1)))
        assert torch.allclose(pre_exec[b, d:], torch.zeros(MAX_DELAY - d, action_dim))


def test_shapes_and_dtypes():
    batch, action_dim = 6, 12
    ring = torch.randn(batch, MAX_DELAY, action_dim)
    delay = torch.randint(0, MAX_DELAY + 1, (batch,))
    pre_exec, mask = extract_pre_execution(ring, delay, MAX_DELAY)
    assert pre_exec.shape == (batch, MAX_DELAY, action_dim)
    assert mask.shape == (batch, MAX_DELAY)
    assert mask.dtype == torch.bool


CANONICAL_RULE = (
    "CANONICAL EXTRACTION RULE (Phase 0, verified):\n"
    "  ring R: [B, L, A], newest issued action at index L-1 (u_{t-1}); L >= max_delay.\n"
    "  extract at DECISION TIME t, BEFORE appending u_t.\n"
    "  pre_exec[:, i] = R[:, L - d + i] for i < d else 0   (oldest-first, i=0 == applied u_{t-d})\n"
    "  mask[:, i]     = (i < d)\n"
    "  delay_normalized = d / max_delay   (continuous; no learnable embedding)\n"
    "  u_t is NOT part of the queue; applied_action is used ONLY in the dynamics dataset."
)


if __name__ == "__main__":
    test_fixed_delays_0_1_3()
    test_delay_zero_is_identity_empty_queue()
    test_mixed_per_env_delays()
    test_shapes_and_dtypes()
    print("All Phase 0 queue-order tests passed.\n")
    print(CANONICAL_RULE)
