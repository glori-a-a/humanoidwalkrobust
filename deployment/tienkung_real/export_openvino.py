#!/usr/bin/env python3
"""Export the combined delay-compensated locomotion policy to OpenVINO IR.

The TorchScript module must already contain both operations performed by
ActorCriticDelayPredictor.act_inference():

1. predict the current 75-value observation frame from the 750-value history;
2. replace the newest frame and run the 20-action actor.

The real-robot C++ code therefore sees one [1, 750] input and one [1, 20]
output.  Keeping the predictor inside the exported graph avoids maintaining a
second, potentially divergent predictor implementation in C++.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

OBS_FRAME_SIZE = 75
OBS_HISTORY_LENGTH = 10
OBSERVATION_SIZE = OBS_FRAME_SIZE * OBS_HISTORY_LENGTH
ACTION_SIZE = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True, help="Combined TorchScript .pt policy")
    parser.add_argument("--output", type=Path, required=True, help="Output .xml path")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def unwrap_output(value: object) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Policy returned unsupported type: {type(value)!r}")
    return value


def main() -> None:
    args = parse_args()
    if args.output.suffix != ".xml":
        raise ValueError("--output must end in .xml")

    try:
        import openvino as ov
    except ImportError as exc:
        raise SystemExit("Install OpenVINO first: pip install openvino") from exc

    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    example = torch.zeros((1, OBSERVATION_SIZE), dtype=torch.float32)

    with torch.inference_mode():
        torch_output = unwrap_output(policy(example))

    expected_output = (1, ACTION_SIZE)
    if tuple(torch_output.shape) != expected_output:
        raise RuntimeError(
            f"Policy contract mismatch: expected output {expected_output}, "
            f"got {tuple(torch_output.shape)}"
        )
    if not torch.isfinite(torch_output).all():
        raise RuntimeError("TorchScript policy produced NaN/Inf on zero input")

    ov_model = ov.convert_model(policy, example_input=example)
    ov_model.reshape({ov_model.input(0): [1, OBSERVATION_SIZE]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(args.output))

    core = ov.Core()
    compiled = core.compile_model(str(args.output), "CPU")
    result = compiled([example.numpy()])[compiled.output(0)]
    if result.shape != expected_output:
        raise RuntimeError(
            f"OpenVINO contract mismatch: expected {expected_output}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise RuntimeError("OpenVINO policy produced NaN/Inf on zero input")

    np.testing.assert_allclose(
        result,
        torch_output.detach().cpu().numpy(),
        atol=args.atol,
        rtol=args.rtol,
    )

    # A second non-zero history catches accidental constant/traced-away inputs.
    probe = torch.linspace(-0.05, 0.05, OBSERVATION_SIZE).reshape(1, -1)
    with torch.inference_mode():
        torch_probe = unwrap_output(policy(probe)).detach().cpu().numpy()
    ov_probe = compiled([probe.numpy()])[compiled.output(0)]
    np.testing.assert_allclose(ov_probe, torch_probe, atol=args.atol, rtol=args.rtol)

    print(f"Exported: {args.output}")
    print(f"Input contract: [1, {OBSERVATION_SIZE}] ({OBS_HISTORY_LENGTH} x {OBS_FRAME_SIZE})")
    print(f"Output contract: [1, {ACTION_SIZE}]")
    print("TorchScript/OpenVINO zero and non-zero numerical tests: PASS")


if __name__ == "__main__":
    main()
