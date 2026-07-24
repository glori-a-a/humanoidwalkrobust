# Fine-tune v2 policy (queue ablation or predictor-feedback) from Strong DR checkpoint.

import argparse
import os
from datetime import datetime

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train v2 queue / predictor-feedback policy.")
parser.add_argument("--task", type=str, default="walk_queue_ablation")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_predictor", action="store_true", help="Main method (v2b); default is queue ablation.")
parser.add_argument("--init_policy_path", type=str, default=None, help="Strong DR checkpoint for actor warm-start.")
parser.add_argument("--delay_min", type=int, default=0, help="Training delay DR min (inclusive).")
parser.add_argument("--delay_max", type=int, default=5, help="Training delay DR max (inclusive).")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if args_cli.task and "sensor" in args_cli.task:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.modules.predictor_feedback_actor_critic import PredictorFeedbackActorCritic
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def enable_delay_dr(env_cfg, dmin=0, dmax=5):
    env_cfg.domain_rand.action_delay.enable = True
    env_cfg.domain_rand.action_delay.params["min_delay"] = dmin
    env_cfg.domain_rand.action_delay.params["max_delay"] = dmax


def train():
    task = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(task)
    env_class = task_registry.get_task_class(task)

    if not getattr(env_cfg, "enable_queue_actor_obs", False):
        raise SystemExit(f"Task {task} must set enable_queue_actor_obs=True (run env_patch_queue_obs.py).")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.scene.seed = agent_cfg.seed
    dmin, dmax = int(args_cli.delay_min), int(args_cli.delay_max)
    enable_delay_dr(env_cfg, dmin, dmax)

    mode = "predictor-feedback" if args_cli.use_predictor else "queue-ablation"
    print(f"[INFO] v2 train mode={mode}, delay DR U[{dmin},{dmax}], seed={agent_cfg.seed}")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)

    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)

    train_cfg = agent_cfg.to_dict()
    train_cfg["policy"]["obs_history_length"] = int(env_cfg.robot.actor_obs_history_length)
    train_cfg["policy"]["use_predictor"] = bool(args_cli.use_predictor)

    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming training from: {resume_path}")
        runner.load(resume_path)
    elif args_cli.init_policy_path:
        init_path = args_cli.init_policy_path
        ckpt = torch.load(init_path, map_location=agent_cfg.device)
        n = PredictorFeedbackActorCritic.warm_start_actor_from_strong(
            runner.alg.policy, ckpt, agent_cfg.device
        )
        print(f"[INFO] warm-start {n} tensors from {init_path}")
    else:
        print("[WARN] no init_policy_path; training v2 actor from scratch")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()
