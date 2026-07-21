# v2b smoke: collect (x_t, u_applied, x_{t+1}), train ResidualDynamics, beat copy baseline.
# Physical state: ang_vel(3) + gravity(3) + joint_pos(A) + joint_vel(A).
# Does NOT train a full predictor-feedback policy — only checks dynamics fit quality.

import argparse
import csv
import os
import sys

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
import legged_lab.utils.cli_args as cli_args

parser = argparse.ArgumentParser(description="v2b frozen dynamics smoke test.")
parser.add_argument("--task", type=str, default="walk")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--collect_steps", type=int, default=200)
parser.add_argument("--delay_min", type=int, default=0)
parser.add_argument("--delay_max", type=int, default=5)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--out_csv", type=str, default="/workspace/results/csv/smoke_dynamics_v2b.csv")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.modules.residual_dynamics import (
    ResidualDynamics,
    copy_baseline_mse,
    train_residual,
)
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner  # noqa: F401


def physical_state(env):
    """Simple flat state used by v2b smoke (no base_lin_vel)."""
    robot = env.robot
    ang = robot.data.root_ang_vel_b
    grav = robot.data.projected_gravity_b
    jp = robot.data.joint_pos - robot.data.default_joint_pos
    jv = robot.data.joint_vel
    return torch.cat([ang, grav, jp, jv], dim=-1)


def applied_action(env):
    # delayed action that actually hit the robot this step
    return env.action.detach().clone()


def main():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = agent_cfg.seed

    env_cfg.domain_rand.action_delay.enable = True
    env_cfg.domain_rand.action_delay.params["min_delay"] = int(args_cli.delay_min)
    env_cfg.domain_rand.action_delay.params["max_delay"] = int(args_cli.delay_max)

    env = task_registry.get_task_class(args_cli.task)(env_cfg, args_cli.headless)

    # optional: drive with a trained policy if --load_run/--checkpoint given
    policy = None
    if args_cli.load_run and args_cli.checkpoint:
        log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
        path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] driving with {path}")
        runner_class = eval(agent_cfg.runner_class_name)
        runner = runner_class(env, agent_cfg.to_dict(), log_dir=os.path.dirname(path), device=agent_cfg.device)
        runner.load(path, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)

    xs, us, xns = [], [], []
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.collect_steps):
            x_t = physical_state(env)
            if policy is not None:
                actions = policy(obs)
            else:
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            obs, _, _, _ = env.step(actions)
            u_app = applied_action(env)
            x_tp1 = physical_state(env)
            xs.append(x_t)
            us.append(u_app)
            xns.append(x_tp1)

    x_t = torch.cat(xs, dim=0)
    u_t = torch.cat(us, dim=0)
    x_tp1 = torch.cat(xns, dim=0)
    print(f"[INFO] collected N={x_t.shape[0]} state_dim={x_t.shape[1]}")

    copy_mse = copy_baseline_mse(x_t, x_tp1)
    model = ResidualDynamics(x_t.shape[1], u_t.shape[1]).to(env.device)
    model_mse = train_residual(model, x_t, u_t, x_tp1, epochs=args_cli.epochs)
    print(f"[RESULT] copy_mse={copy_mse:.6f}  residual_mse={model_mse:.6f}")

    ok = model_mse < copy_mse
    print(f"[RESULT] beats_copy={ok}")

    os.makedirs(os.path.dirname(args_cli.out_csv) or ".", exist_ok=True)
    row = {
        "n_samples": int(x_t.shape[0]),
        "state_dim": int(x_t.shape[1]),
        "copy_mse": copy_mse,
        "residual_mse": model_mse,
        "beats_copy": int(ok),
        "delay_min": int(args_cli.delay_min),
        "delay_max": int(args_cli.delay_max),
    }
    with open(args_cli.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"[INFO] Saved {args_cli.out_csv}")

    # also dump the model for later v2b wiring
    out_pt = args_cli.out_csv.replace(".csv", ".pt")
    torch.save({"state_dict": model.state_dict(), "state_dim": x_t.shape[1], "action_dim": u_t.shape[1]}, out_pt)
    print(f"[INFO] Saved {out_pt}")

    if not ok:
        raise SystemExit("SMOKE FAIL: residual dynamics did not beat copy baseline")
    print("SMOKE PASS: residual dynamics better than copy")
    sys.exit(0)


if __name__ == "__main__":
    main()
