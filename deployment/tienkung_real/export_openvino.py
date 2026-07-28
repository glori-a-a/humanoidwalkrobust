#!/usr/bin/env python3
"""Export a TorchScript locomotion policy to OpenVINO IR.

The official TienKung deployment stack uses OpenVINO for policy inference.
This utility converts an exported TienKung-Lab TorchScript policy and performs
one numerical smoke test before writing the IR files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True, help="TorchScript .pt policy")
    parser.add_argument("--output", type=Path, required=True, help="Output .xml path")
    parser.add_argument(
        "--observation-size",
        type=int,
        required=True,
        help="Flattened policy observation dimension used during training",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.observation_size <= 0:
        raise ValueError("--observation-size must be positive")
    if args.output.suffix != ".xml":
        raise ValueError("--output must end in .xml")

    try:
        import openvino as ov
    except ImportError as exc:
        raise SystemExit("Install OpenVINO first: pip install openvino") from exc

    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    example = torch.zeros((1, args.observation_size), dtype=torch.float32)

    with torch.inference_mode():
        torch_output = policy(example)
    if isinstance(torch_output, (tuple, list)):
        torch_output = torch_output[0]
    if torch_output.ndim != 2 or torch_output.shape[0] != 1:
        raise RuntimeError(f"Unexpected policy output shape: {tuple(torch_output.shape)}")

    ov_model = ov.convert_model(policy, example_input=example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(args.output))

    core = ov.Core()
    compiled = core.compile_model(str(args.output), "CPU")
    result = compiled([example.numpy()])[compiled.output(0)]
    np.testing.assert_allclose(
        result,
        torch_output.detach().cpu().numpy(),
        atol=args.atol,
        rtol=args.rtol,
    )

    print(f"Exported: {args.output}")
    print(f"Input shape: {tuple(example.shape)}")
    print(f"Action size: {result.shape[-1]}")
    print("TorchScript/OpenVINO numerical smoke test: PASS")


if __name__ == "__main__":
    main()
