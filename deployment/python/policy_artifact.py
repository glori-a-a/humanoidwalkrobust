#!/usr/bin/env python3
"""Package and benchmark a scripted locomotion policy for deployment.

The input must be a TorchScript policy whose first output is an action tensor.
This keeps reconstruction of the RSL-RL actor inside the training environment
and makes the exported artifact independently reviewable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def first_tensor(output):
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("policy returned an empty sequence")
        return output[0]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    if args.observation_dim <= 0 or args.action_dim <= 0:
        raise SystemExit("observation/action dimensions must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be non-negative and iterations positive")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to package the policy") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = torch.jit.load(str(args.input), map_location=args.device)
    model.eval()
    example = torch.zeros(
        1, args.observation_dim, dtype=torch.float32, device=args.device
    )

    with torch.inference_mode():
        reference = first_tensor(model(example))
    if tuple(reference.shape) != (1, args.action_dim):
        raise SystemExit(
            f"policy output shape {tuple(reference.shape)} != (1, {args.action_dim})"
        )
    if not torch.isfinite(reference).all():
        raise SystemExit("policy produced a non-finite action")

    frozen = torch.jit.freeze(model)
    torchscript_path = args.output_dir / "policy_torchscript.pt"
    torch.jit.save(frozen, str(torchscript_path))

    reloaded = torch.jit.load(str(torchscript_path), map_location=args.device)
    reloaded.eval()
    with torch.inference_mode():
        candidate = first_tensor(reloaded(example))
    max_abs_error = float((reference - candidate).abs().max().item())
    if max_abs_error > 1e-6:
        raise SystemExit(f"TorchScript parity failed: max abs error={max_abs_error}")

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(args.warmup):
            first_tensor(reloaded(example))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            first_tensor(reloaded(example))
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    artifacts = {
        "torchscript": {
            "path": torchscript_path.name,
            "sha256": sha256_file(torchscript_path),
            "bytes": torchscript_path.stat().st_size,
        }
    }

    if args.export_onnx:
        onnx_path = args.output_dir / "policy.onnx"
        torch.onnx.export(
            frozen,
            example,
            str(onnx_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
        artifacts["onnx"] = {
            "path": onnx_path.name,
            "sha256": sha256_file(onnx_path),
            "bytes": onnx_path.stat().st_size,
            "parity": "not checked unless an ONNX runtime is used separately",
        }

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": args.input.name,
            "sha256": sha256_file(args.input),
        },
        "model": {
            "observation_dim": args.observation_dim,
            "action_dim": args.action_dim,
            "dtype": "float32",
            "torchscript_max_abs_error": max_abs_error,
        },
        "benchmark": {
            "device": args.device,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "mean_ms": statistics.fmean(latencies_ms),
            "p50_ms": percentile(latencies_ms, 0.50),
            "p95_ms": percentile(latencies_ms, 0.95),
            "max_ms": max(latencies_ms),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "artifacts": artifacts,
        "validation_scope": {
            "shape_and_finite_checks": True,
            "torchscript_parity": True,
            "physical_robot": False,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

