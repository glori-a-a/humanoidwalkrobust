import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            result.update(chunk)
    return result.hexdigest()


def get_action(output):
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("policy returned no output")
        return output[0]
    return output


def measure_latency(torch, model, observation, device, warmup, runs):
    with torch.inference_mode():
        for _ in range(warmup):
            get_action(model(observation))

    times = []
    with torch.inference_mode():
        for _ in range(runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            get_action(model(observation))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return times


def percentile(values, ratio):
    values = sorted(values)
    index = round((len(values) - 1) * ratio)
    return values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    if args.observation_dim <= 0 or args.action_dim <= 0:
        raise SystemExit("model dimensions must be positive")
    if args.warmup < 0 or args.runs <= 0:
        raise SystemExit("invalid benchmark settings")

    try:
        import torch
    except ImportError as error:
        raise SystemExit("PyTorch is required") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = torch.jit.load(str(args.input), map_location=args.device)
    model.eval()
    observation = torch.zeros(
        1, args.observation_dim, dtype=torch.float32, device=args.device
    )

    with torch.inference_mode():
        original_action = get_action(model(observation))
    if tuple(original_action.shape) != (1, args.action_dim):
        raise SystemExit("policy output has the wrong shape")
    if not torch.isfinite(original_action).all():
        raise SystemExit("policy output contains non-finite values")

    model = torch.jit.freeze(model)
    model_path = args.output_dir / "policy_torchscript.pt"
    torch.jit.save(model, str(model_path))

    saved_model = torch.jit.load(str(model_path), map_location=args.device)
    saved_model.eval()
    with torch.inference_mode():
        saved_action = get_action(saved_model(observation))
    max_error = float((original_action - saved_action).abs().max().item())
    if max_error > 1e-6:
        raise SystemExit("saved model output does not match the source model")

    times = measure_latency(
        torch,
        saved_model,
        observation,
        args.device,
        args.warmup,
        args.runs,
    )

    artifacts = {
        "torchscript": {
            "path": model_path.name,
            "sha256": file_hash(model_path),
            "bytes": model_path.stat().st_size,
        }
    }

    if args.export_onnx:
        onnx_path = args.output_dir / "policy.onnx"
        torch.onnx.export(
            model,
            observation,
            str(onnx_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
        )
        artifacts["onnx"] = {
            "path": onnx_path.name,
            "sha256": file_hash(onnx_path),
            "bytes": onnx_path.stat().st_size,
        }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": file_hash(args.input),
        "observation_dim": args.observation_dim,
        "action_dim": args.action_dim,
        "device": args.device,
        "mean_latency_ms": statistics.fmean(times),
        "p50_latency_ms": percentile(times, 0.5),
        "p95_latency_ms": percentile(times, 0.95),
        "max_latency_ms": max(times),
        "max_output_error": max_error,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "artifacts": artifacts,
        "physical_robot_tested": False,
    }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
