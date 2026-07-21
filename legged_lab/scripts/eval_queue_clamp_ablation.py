# Test-time queue input clamp ablation under fixed delay (default d=8).
# Env delay is always the true value; we only edit what the policy sees.
#
# Modes (see delay_queue.clamp_queue_block):
#   full         — true queue + true delay
#   clamp_delay  — delay_norm clamped to train_max/max_delay (e.g. 5/8)
#   keep5        — only first 5 queue slots kept
#   zero_queue   — wipe queue block (base obs only)

import argparse
import csv
import os
import sys
from collections import defaultdict

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
import legged_lab.utils.cli_args as cli_args

parser = argparse.ArgumentParser(description="Queue clamp ablation at fixed delay.")
parser.add_argument("--task", type=str, default="walk_queue_ablation")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--action_delay_steps", type=int, default=8)
parser.add_argument(
    "--clamp_mode",
    type=str,
    default="full",
    choices=["full", "clamp_delay", "keep5", "zero_queue"],
)
parser.add_argument("--train_max_delay", type=int, default=5)
parser.add_argument("--output", type=str, default="/workspace/results/csv/eval_queue_clamp.csv")
parser.add_argument("--append", action="store_true")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.modules.delay_queue import MAX_DELAY, clamp_queue_block
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner  # noqa: F401


def make_plane_delay_env(env_cfg, delay_steps: int):
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.env_spacing = 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    d = int(delay_steps)
    env_cfg.domain_rand.action_delay.enable = d > 0
    if d > 0:
        env_cfg.domain_rand.action_delay.params["min_delay"] = d
        env_cfg.domain_rand.action_delay.params["max_delay"] = d


class ClampPolicy:
    """Wrap inference policy; edit queue suffix before forward."""

    def __init__(self, policy, num_actions, mode, train_max_delay):
        self.policy = policy
        self.num_actions = num_actions
        self.mode = mode
        self.train_max_delay = train_max_delay

    def __call__(self, obs):
        obs = clamp_queue_block(
            obs,
            num_actions=self.num_actions,
            mode=self.mode,
            max_delay=MAX_DELAY,
            train_max_delay=self.train_max_delay,
        )
        return self.policy(obs)


def load_rsl_policy(env, agent_cfg):
    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {path}")
    run_dir = os.path.dirname(path)
    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=run_dir, device=agent_cfg.device)
    runner.load(path, load_optimizer=False)
    return runner.get_inference_policy(device=env.device), path


def run_eval(env, policy, n_episodes):
    max_steps = int(env.max_episode_length)
    done_n = 0
    lengths, returns = [], []
    falls = 0
    reward_sum = defaultdict(float)
    reward_cnt = defaultdict(int)

    obs, _ = env.get_observations()
    ep_ret = torch.zeros(env.num_envs, device=env.device)
    ep_len = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    while done_n < n_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

        ep_ret += rewards
        ep_len += 1
        for i in torch.where(dones)[0].tolist():
            done_n += 1
            L = int(ep_len[i].item())
            lengths.append(L)
            returns.append(float(ep_ret[i].item()))
            if L < max_steps:
                falls += 1
            ep_ret[i] = 0.0
            ep_len[i] = 0

        if "log" in extras:
            for k, v in extras["log"].items():
                if torch.is_tensor(v):
                    reward_sum[k] += v.mean().item()
                    reward_cnt[k] += 1

    track = reward_sum.get("Episode_Reward/track_lin_vel_xy_exp", float("nan"))
    if reward_cnt.get("Episode_Reward/track_lin_vel_xy_exp", 0):
        track /= reward_cnt["Episode_Reward/track_lin_vel_xy_exp"]

    return {
        "episodes": done_n,
        "success_rate": 1.0 - falls / max(done_n, 1),
        "fall_count": falls,
        "mean_episode_length": sum(lengths) / max(len(lengths), 1),
        "mean_episode_return": sum(returns) / max(len(returns), 1),
        "track_lin_vel_xy_exp": track,
    }


def main():
    if args_cli.load_run is None or args_cli.checkpoint is None:
        raise SystemExit("Need --load_run and --checkpoint for queue ablation ckpt.")

    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = agent_cfg.seed
    make_plane_delay_env(env_cfg, args_cli.action_delay_steps)

    env = task_registry.get_task_class(args_cli.task)(env_cfg, args_cli.headless)
    base_policy, ckpt_path = load_rsl_policy(env, agent_cfg)
    policy = ClampPolicy(
        base_policy,
        num_actions=env.num_actions,
        mode=args_cli.clamp_mode,
        train_max_delay=args_cli.train_max_delay,
    )

    print(
        f"[EVAL] clamp={args_cli.clamp_mode} true_delay={args_cli.action_delay_steps} "
        f"train_max={args_cli.train_max_delay}"
    )
    stats = run_eval(env, policy, args_cli.episodes)
    print(
        f"  success={stats['success_rate']:.2%} len={stats['mean_episode_length']:.1f} "
        f"return={stats['mean_episode_return']:.2f}"
    )

    row = {
        "policy_source": ckpt_path,
        "clamp_mode": args_cli.clamp_mode,
        "true_delay_steps": int(args_cli.action_delay_steps),
        "train_max_delay": int(args_cli.train_max_delay),
        "eval_seed": agent_cfg.seed,
        **stats,
    }
    os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
    write_header = not (args_cli.append and os.path.isfile(args_cli.output))
    with open(args_cli.output, "a" if args_cli.append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[INFO] Saved: {args_cli.output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
