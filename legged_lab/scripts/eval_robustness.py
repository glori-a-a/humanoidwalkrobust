"""Evaluate a frozen walk policy under nominal or single-parameter perturbations (Phase 2/3)."""
import argparse
import csv
import os
import sys
from collections import defaultdict
import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
import legged_lab.utils.cli_args as cli_args

parser = argparse.ArgumentParser(description="Robustness evaluation for TienKung walk (policy frozen).")
parser.add_argument("--task", type=str, default="walk")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=20, help="Completed episodes to collect.")
parser.add_argument(
    "--policy_path",
    type=str,
    default=None,
    help="TorchScript policy (e.g. Exported_policy/walk.pt) for script validation.",
)
parser.add_argument("--intervention", type=str, default="nominal", choices=["nominal", "mass", "friction", "delay"])
parser.add_argument("--mass_delta", type=float, default=0.0, help="Pelvis mass offset (kg) when intervention=mass.")
parser.add_argument(
    "--static_friction",
    type=float,
    default=0.8,
    help="Fixed static friction when intervention=friction.",
)
parser.add_argument(
    "--dynamic_friction",
    type=float,
    default=0.6,
    help="Fixed dynamic friction when intervention=friction.",
)
parser.add_argument(
    "--action_delay_steps",
    type=int,
    default=0,
    help="Fixed control delay in sim steps when intervention=delay (fixed mode).",
)
parser.add_argument(
    "--delay_mode",
    type=str,
    default="fixed",
    choices=["fixed", "uniform"],
    help="fixed: min=max=delay_steps; uniform: delay ~ U(0, delay_steps) per env at init.",
)
parser.add_argument("--output", type=str, default="/workspace/results/eval_row.csv")
parser.add_argument("--append", action="store_true")
parser.add_argument("--lin_vel_x", type=float, default=1.0)
parser.add_argument("--lin_vel_y", type=float, default=0.0)
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


def configure_play_baseline(env_cfg, lin_vel_x: float = 1.0, lin_vel_y: float = 0.0):
    """Match legged_lab/scripts/play.py evaluation conditions."""
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


def apply_intervention(env_cfg, args):
    configure_play_baseline(env_cfg, args.lin_vel_x, args.lin_vel_y)

    if args.intervention == "nominal":
        # play.py leaves domain-rand events at training defaults; no extra overrides.
        return

    if args.intervention == "mass":
        env_cfg.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (
            args.mass_delta,
            args.mass_delta,
        )
        env_cfg.domain_rand.events.add_base_mass.params["operation"] = "add"
        env_cfg.domain_rand.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        env_cfg.domain_rand.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        return

    if args.intervention == "friction":
        env_cfg.domain_rand.events.physics_material.params["static_friction_range"] = (
            args.static_friction,
            args.static_friction,
        )
        env_cfg.domain_rand.events.physics_material.params["dynamic_friction_range"] = (
            args.dynamic_friction,
            args.dynamic_friction,
        )
        env_cfg.domain_rand.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        env_cfg.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        return

    if args.intervention == "delay":
        d = max(0, int(args.action_delay_steps))
        if d == 0 and args.delay_mode == "fixed":
            env_cfg.domain_rand.action_delay.enable = False
            return
        env_cfg.domain_rand.action_delay.enable = True
        if args.delay_mode == "fixed":
            env_cfg.domain_rand.action_delay.params["min_delay"] = d
            env_cfg.domain_rand.action_delay.params["max_delay"] = d
        else:
            env_cfg.domain_rand.action_delay.params["min_delay"] = 0
            env_cfg.domain_rand.action_delay.params["max_delay"] = d
        return

    raise ValueError(f"Unknown intervention: {args.intervention}")


def intervention_label(args):
    if args.intervention == "nominal":
        return "0"
    if args.intervention == "mass":
        return f"{args.mass_delta:+.1f}kg"
    if args.intervention == "friction":
        return f"mu_s={args.static_friction:.2f},mu_d={args.dynamic_friction:.2f}"
    if args.intervention == "delay":
        suffix = "fixed" if args.delay_mode == "fixed" else "uniform0"
        return f"{args.action_delay_steps}steps_{suffix}"
    return "unknown"


def load_policy(env, agent_cfg, log_dir, policy_path, checkpoint_name):
    if policy_path:
        path = retrieve_file_path(policy_path) if not os.path.isabs(policy_path) else policy_path
        path = os.path.abspath(path)
        print(f"[INFO] Loading TorchScript policy: {path}")
        policy = torch.jit.load(path, map_location=env.device)
        policy.eval()
        return policy, None

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    if checkpoint_name:
        agent_cfg.load_checkpoint = checkpoint_name
    path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading RSL-RL checkpoint: {path}")
    # Match play.py: runner needs the run directory (normalizer / AMP artifacts).
    run_log_dir = os.path.dirname(path)
    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=run_log_dir, device=agent_cfg.device)
    runner.load(path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    return policy, runner


def evaluate(env, policy, target_episodes: int, intervention: str, intervention_value: str, policy_source: str, eval_seed: int):
    num_envs = env.num_envs
    max_episode_steps = int(env.max_episode_length)

    completed = 0
    episode_lengths = []
    episode_returns = []
    fall_count = 0
    reward_sums = defaultdict(float)
    reward_counts = defaultdict(int)

    obs, _ = env.get_observations()
    ep_return = torch.zeros(num_envs, device=env.device)
    ep_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)

    while completed < target_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

        ep_return += rewards
        ep_steps += 1

        for idx in torch.where(dones)[0].tolist():
            completed += 1
            length = ep_steps[idx].item()
            episode_lengths.append(length)
            episode_returns.append(ep_return[idx].item())
            if length < max_episode_steps:
                fall_count += 1
            ep_return[idx] = 0.0
            ep_steps[idx] = 0

        if "log" in extras:
            for key, value in extras["log"].items():
                if torch.is_tensor(value):
                    reward_sums[key] += value.mean().item()
                    reward_counts[key] += 1

    def avg_term(name):
        return reward_sums[name] / reward_counts[name] if reward_counts[name] else float("nan")

    row = {
        "policy_source": policy_source,
        "intervention": intervention,
        "intervention_value": intervention_value,
        "delay_steps": int(args_cli.action_delay_steps) if args_cli.intervention == "delay" else 0,
        "delay_mode": args_cli.delay_mode if args_cli.intervention == "delay" else "n/a",
        "lin_vel_x": float(args_cli.lin_vel_x),
        "lin_vel_y": float(args_cli.lin_vel_y),
        "eval_seed": eval_seed,
        "episodes": completed,
        "success_rate": 1.0 - fall_count / max(completed, 1),
        "fall_count": fall_count,
        "mean_episode_length": sum(episode_lengths) / max(len(episode_lengths), 1),
        "mean_episode_return": sum(episode_returns) / max(len(episode_returns), 1),
        "track_lin_vel_xy_exp": avg_term("Episode_Reward/track_lin_vel_xy_exp"),
        "termination_penalty": avg_term("Episode_Reward/termination_penalty"),
    }
    return row


def main():
    if not args_cli.policy_path and args_cli.checkpoint is None and args_cli.load_run is None:
        raise SystemExit("Provide --policy_path (JIT) or --checkpoint / --load_run (trained baseline).")

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = agent_cfg.seed

    apply_intervention(env_cfg, args_cli)
    label = intervention_label(args_cli)

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_dir = os.path.join("logs", agent_cfg.experiment_name)
    policy_source = args_cli.policy_path or args_cli.checkpoint or f"logs/{agent_cfg.experiment_name}"
    policy, _runner = load_policy(
        env,
        agent_cfg,
        log_dir,
        args_cli.policy_path,
        args_cli.checkpoint,
    )

    print(f"\n[EVAL] intervention={args_cli.intervention} value={label}")
    row = evaluate(env, policy, args_cli.episodes, args_cli.intervention, label, policy_source, agent_cfg.seed)
    print(
        f"  episodes={row['episodes']} success={row['success_rate']:.2%} "
        f"len={row['mean_episode_length']:.1f} return={row['mean_episode_return']:.2f}"
    )

    os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
    write_header = not (args_cli.append and os.path.isfile(args_cli.output))
    with open(args_cli.output, "a" if args_cli.append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[INFO] Saved: {args_cli.output}")
    # Isaac headless shutdown can hang; exit cleanly after one eval point.
    sys.exit(0)


if __name__ == "__main__":
    main()
