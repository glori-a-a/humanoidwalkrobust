# Stage 1: collect (delayed, teacher) pairs in sim, train ObsPredictor with MSE.
# Needs env patch from apply_tienkung_env_patch.py

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

import legged_lab.utils.cli_args as cli_args  # isort: skip

# --- CLI ---
parser = argparse.ArgumentParser(description="Stage 1 supervised obs predictor.")
parser.add_argument("--task", type=str, default="walk")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--collect_steps", type=int, default=500)
parser.add_argument("--delay_min", type=int, default=0)
parser.add_argument("--delay_max", type=int, default=8)
parser.add_argument("--policy_path", type=str, default=None)
parser.add_argument("--train_epochs", type=int, default=80)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--out_dir", type=str, default="/workspace/results/phase4_stage1")
parser.add_argument("--predictor_hidden", type=str, default="256,256")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa: F401, F403

repo = Path(os.environ.get("HLC_REPO", "/workspace/humanoidlearningcontrol"))
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
from phase4.predictor import ObsPredictor  # noqa: E402


def load_jit(path, device):
    policy = torch.jit.load(path, map_location=device)
    policy.eval()
    return policy


def collect_rollout(env, policy, delay, steps, device):
    # fix delay, step sim, save delayed input + teacher label each step
    env.set_delay(delay)
    obs, _ = env.get_observations()
    delayed_buf, teacher_buf, delay_buf = [], [], []

    with torch.inference_mode():
        for _ in range(steps):
            if policy is not None:
                actions = policy(obs)
            else:
                actions = torch.zeros(env.num_envs, env.num_actions, device=device)
            obs, _, _, extras = env.step(actions)
            row = extras["teacher"]
            delayed_buf.append(row["delayed"].clone())
            teacher_buf.append(row["obs"].clone())
            delay_buf.append(row["delay"].clone())

    return {
        "delayed": torch.cat(delayed_buf, dim=0),
        "teacher": torch.cat(teacher_buf, dim=0),
        "delay": torch.cat(delay_buf, dim=0),
    }


def train_net(model, data, epochs, batch_size, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = data["teacher"].shape[0]
    model.train()

    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        running = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(data["delayed"][idx])
            loss = loss_fn(pred, data["teacher"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            batches += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[Stage1] epoch {ep + 1}/{epochs} mse={running / max(batches, 1):.6f}")


def mse_by_delay(model, data):
    # one MSE number per fixed delay (for thesis plot)
    model.eval()
    rows = []
    with torch.no_grad():
        for d in sorted(data["delay"].unique().tolist()):
            mask = data["delay"] == d
            pred = model(data["delayed"][mask])
            mse = torch.mean((pred - data["teacher"][mask]) ** 2).item()
            rows.append({"delay_steps": int(d), "delay_ms": int(d) * 20, "mse": mse})
            print(f"[Stage1] delay={int(d)} mse={mse:.6f}")
    return rows


def enable_delay_sweep(env_cfg, dmin, dmax):
    env_cfg.domain_rand.action_delay.enable = True
    env_cfg.domain_rand.action_delay.params["min_delay"] = dmin
    env_cfg.domain_rand.action_delay.params["max_delay"] = dmax


def main():
    device = getattr(args_cli, "device", None) or "cuda:0"
    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- build env ---
    task = args_cli.task
    env_cfg, _ = task_registry.get_cfgs(task)
    env_class = task_registry.get_task_class(task)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = args_cli.seed
    enable_delay_sweep(env_cfg, args_cli.delay_min, args_cli.delay_max)
    env = env_class(env_cfg, args_cli.headless)

    policy = load_jit(args_cli.policy_path, device) if args_cli.policy_path else None

    # --- collect ---
    chunks = []
    for d in range(args_cli.delay_min, args_cli.delay_max + 1):
        print(f"[Stage1] collect delay={d} steps={args_cli.collect_steps}")
        chunks.append(collect_rollout(env, policy, d, args_cli.collect_steps, device))

    data = {
        key: torch.cat([c[key] for c in chunks], dim=0).to(device)
        for key in ("delayed", "teacher", "delay")
    }

    # --- train ---
    frame_dim = data["teacher"].shape[-1]
    hidden = [int(x) for x in args_cli.predictor_hidden.split(",") if x.strip()]
    model = ObsPredictor(
        history_dim=data["delayed"].shape[-1],
        frame_dim=frame_dim,
        hidden_dims=hidden,
    ).to(device)

    train_net(model, data, args_cli.train_epochs, args_cli.batch_size, args_cli.lr, device)
    mse_rows = mse_by_delay(model, data)

    # --- save ---
    ckpt = {
        "model_state_dict": model.net.state_dict(),
        "history_dim": data["delayed"].shape[-1],
        "frame_dim": frame_dim,
        "history_len": int(env_cfg.robot.actor_obs_history_length),
        "hidden_dims": hidden,
    }
    ckpt_path = out_dir / "predictor_stage1.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[Stage1] saved {ckpt_path}")

    csv_path = out_dir / "predictor_mse_vs_delay.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["delay_steps", "delay_ms", "mse"])
        writer.writeheader()
        writer.writerows(mse_rows)
    print(f"[Stage1] saved {csv_path}")


if __name__ == "__main__":
    main()
