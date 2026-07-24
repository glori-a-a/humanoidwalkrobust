#!/usr/bin/env python3
"""Validated TorchScript inference wrapper for a locomotion policy.

The wrapper is intentionally independent of ROS 2. A ROS node can assemble the
observation vector, call :meth:`PolicyRuntime.infer`, and publish the normalized
action to ``/policy/action_normalized``. Command limiting remains the
responsibility of the downstream fail-closed safety bridge.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Sequence


def first_tensor(output):
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("policy returned an empty sequence")
        return output[0]
    return output


class PolicyRuntime:
    """Load a policy artifact and reject malformed observations/actions."""

    def __init__(
        self,
        artifact: Path,
        observation_dim: int,
        action_dim: int = 20,
        device: str = "cpu",
    ) -> None:
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("observation/action dimensions must be positive")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for policy inference") from exc

        self.torch = torch
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.model = torch.jit.load(str(artifact), map_location=device)
        self.model.eval()

    def infer(self, observation: Sequence[float]) -> tuple[tuple[float, ...], float]:
        if len(observation) != self.observation_dim:
            raise ValueError(
                f"observation dimension {len(observation)} != {self.observation_dim}"
            )
        if not all(math.isfinite(float(value)) for value in observation):
            raise ValueError("observation contains a non-finite value")

        tensor = self.torch.tensor(
            observation, dtype=self.torch.float32, device=self.device
        ).reshape(1, self.observation_dim)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        start_ns = time.perf_counter_ns()
        with self.torch.inference_mode():
            action = first_tensor(self.model(tensor))
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        if tuple(action.shape) != (1, self.action_dim):
            raise RuntimeError(
                f"policy output shape {tuple(action.shape)} != (1, {self.action_dim})"
            )
        if not self.torch.isfinite(action).all():
            raise RuntimeError("policy produced a non-finite action")

        values = tuple(float(value) for value in action[0].detach().cpu().tolist())
        return values, latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one validated inference from a JSON observation."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--observation-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--observation-json",
        required=True,
        help="JSON list whose length equals --observation-dim",
    )
    args = parser.parse_args()

    observation = json.loads(args.observation_json)
    if not isinstance(observation, list):
        raise SystemExit("--observation-json must decode to a JSON list")
    runtime = PolicyRuntime(
        args.artifact, args.observation_dim, args.action_dim, args.device
    )
    action, latency_ms = runtime.infer(observation)
    print(json.dumps({"action": action, "latency_ms": latency_ms}))


if __name__ == "__main__":
    main()
