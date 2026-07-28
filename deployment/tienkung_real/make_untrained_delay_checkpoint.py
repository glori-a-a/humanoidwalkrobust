#!/usr/bin/env python3
"""Create a format-compatible, explicitly untrained delay-policy checkpoint.

This file is for learning the checkpoint-loading/export pipeline only. It is
randomly initialised, cannot control a robot, and must never be presented as a
trained result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

OBS_SIZE = 750
FRAME_SIZE = 75
ACTION_SIZE = 20


class UntrainedDelayPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(OBS_SIZE, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, FRAME_SIZE),
        )
        self.actor = nn.Sequential(
            nn.Linear(OBS_SIZE, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_SIZE),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        predicted = self.predictor(observations)
        frames = observations.reshape(-1, 10, FRAME_SIZE).clone()
        frames[:, -1, :] = predicted
        return self.actor(frames.reshape(-1, OBS_SIZE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("checkpoints/synthetic_untrained_delay_policy.pt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = UntrainedDelayPolicy().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "iteration": 0,
        "optimizer_state_dict": None,
        "metadata": {
            "synthetic": True,
            "trained": False,
            "purpose": "checkpoint loading/export pipeline study only",
            "observation_size": OBS_SIZE,
            "history_length": 10,
            "frame_size": FRAME_SIZE,
            "action_size": ACTION_SIZE,
            "architecture": "predictor[750-256-256-75] + actor[750-512-256-128-20]",
        },
    }
    torch.save(checkpoint, args.output)

    scripted = torch.jit.script(model)
    scripted_path = args.output.with_name(args.output.stem + "_torchscript.pt")
    scripted.save(str(scripted_path))

    print(f"Saved untrained checkpoint: {args.output}")
    print(f"Saved untrained TorchScript model: {scripted_path}")
    print("WARNING: random weights; not a trained controller; never run on hardware.")


if __name__ == "__main__":
    main()
