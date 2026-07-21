"""Record commanded vs actual body velocities (vx, vy, yaw) over one eval episode."""
import argparse
import csv
import os
import sys

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
import legged_lab.utils.cli_args as cli_args

parser = argparse.ArgumentParser(description="Record vx time-series for tracking plots.")
parser.add_argument("--task", type=str, default="walk")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--policy_path", type=str, default=None)
parser.add_argument("--policy_label", type=str, default="policy")
parser.add_argument("--intervention", type=str, default="delay", choices=["nominal", "delay"])
parser.add_argument("--action_delay_steps", type=int, default=0)
parser.add_argument("--lin_vel_x", type=float, default=1.0)
parser.add_argument("--lin_vel_y", type=float, default=0.0)
parser.add_argument("--output", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from legged_lab.envs import *  # noqa: F401,F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner


def configure_eval(env_cfg, lin_vel_x, lin_vel_y):
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (lin_vel_x, lin_vel_x)
    env_cfg.commands.ranges.lin_vel_y = (lin_vel_y, lin_vel_y)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"


def apply_delay(env_cfg, delay_steps):
    d = max(0, int(delay_steps))
    if d == 0:
        env_cfg.domain_rand.action_delay.enable = False
        return
    env_cfg.domain_rand.action_delay.enable = True
    env_cfg.domain_rand.action_delay.params["min_delay"] = d
    env_cfg.domain_rand.action_delay.params["max_delay"] = d


def load_policy(env, agent_cfg, policy_path, checkpoint_name):
    if policy_path:
        path = retrieve_file_path(policy_path) if not os.path.isabs(policy_path) else policy_path
        path = os.path.abspath(path)
        print(f"[INFO] JIT policy: {path}")
        policy = torch.jit.load(path, map_location=env.device)
        policy.eval()
        return policy

    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    if checkpoint_name:
        agent_cfg.load_checkpoint = checkpoint_name
    path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] checkpoint: {path}")
    run_log_dir = os.path.dirname(path)
    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=run_log_dir, device=agent_cfg.device)
    runner.load(path, load_optimizer=False)
    return runner.get_inference_policy(device=env.device)


def record_episode(env, policy):
    # one env, step until episode ends
    obs, _ = env.get_observations()
    step_dt = float(env.step_dt)
    max_steps = int(env.max_episode_length)
    rows = []

    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        cmd = env.command_generator.command[0]
        vel_b = env.robot.data.root_lin_vel_b[0]
        ang_b = env.robot.data.root_ang_vel_b[0]
        rows.append(
            {
                "step": step,
                "time_s": step * step_dt,
                "vx_cmd": cmd[0].item(),
                "vy_cmd": cmd[1].item(),
                "yaw_cmd": cmd[2].item(),
                "vx_actual": vel_b[0].item(),
                "vy_actual": vel_b[1].item(),
                "yaw_actual": ang_b[2].item(),
            }
        )
        if dones[0]:
            break

    fell = len(rows) < max_steps
    for row in rows:
        row["fell"] = int(fell)
        row["episode_steps"] = len(rows)
    return rows, fell


def main():
    if not args_cli.policy_path and args_cli.checkpoint is None and args_cli.load_run is None:
        raise SystemExit("Need --policy_path or --checkpoint/--load_run")

    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = args_cli.seed

    configure_eval(env_cfg, args_cli.lin_vel_x, args_cli.lin_vel_y)
    if args_cli.intervention == "delay":
        apply_delay(env_cfg, args_cli.action_delay_steps)

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    policy = load_policy(env, agent_cfg, args_cli.policy_path, args_cli.checkpoint)
    rows, fell = record_episode(env, policy)

    meta = {
        "policy": args_cli.policy_label,
        "delay_steps": int(args_cli.action_delay_steps),
        "delay_ms": int(args_cli.action_delay_steps) * 20,
        "seed": args_cli.seed,
        "lin_vel_x": args_cli.lin_vel_x,
        "lin_vel_y": args_cli.lin_vel_y,
        "task": args_cli.task,
    }
    for row in rows:
        row.update(meta)

    os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    with open(args_cli.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[INFO] policy={meta['policy']} delay={meta['delay_steps']} "
        f"steps={len(rows)} fell={fell} -> {args_cli.output}"
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
    os._exit(0)
