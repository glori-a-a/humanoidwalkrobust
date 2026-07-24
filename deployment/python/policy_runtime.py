import argparse
import json
import math
from pathlib import Path
import time


def load_policy(path, device):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required") from error

    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return torch, model


def get_action(output):
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("policy returned no output")
        return output[0]
    return output


def run_policy(torch, model, observation, observation_dim, action_dim, device):
    if len(observation) != observation_dim:
        raise ValueError("wrong observation size")
    if not all(math.isfinite(float(value)) for value in observation):
        raise ValueError("observation contains non-finite values")

    tensor = torch.tensor(
        observation, dtype=torch.float32, device=device
    ).reshape(1, observation_dim)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        action = get_action(model(tensor))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    if tuple(action.shape) != (1, action_dim):
        raise RuntimeError("wrong action size")
    if not torch.isfinite(action).all():
        raise RuntimeError("policy returned non-finite values")

    values = action[0].detach().cpu().tolist()
    return values, latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--observation-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--observation-json", required=True)
    args = parser.parse_args()

    observation = json.loads(args.observation_json)
    if not isinstance(observation, list):
        raise SystemExit("observation must be a JSON list")

    torch, model = load_policy(args.artifact, args.device)
    action, latency = run_policy(
        torch,
        model,
        observation,
        args.observation_dim,
        args.action_dim,
        args.device,
    )
    print(json.dumps({"action": action, "latency_ms": latency}))


if __name__ == "__main__":
    main()
