# Ablation C: longer obs history (20) + delay DR, no predictor.

import argparse
import os
import sys
from datetime import datetime

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="History ablation (no predictor).")
parser.add_argument("--task", type=str, default="walk_history_ablation")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def apply_ablation_cfg(env_cfg):
    env_cfg.robot.actor_obs_history_length = 20
    env_cfg.robot.critic_obs_history_length = 20
    env_cfg.domain_rand.action_delay.enable = True
    env_cfg.domain_rand.action_delay.params["min_delay"] = 0
    env_cfg.domain_rand.action_delay.params["max_delay"] = 5
    print(f"[INFO] ablation history={env_cfg.robot.actor_obs_history_length}, delay DR ON")


def train():
    task = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(task)
    env_class = task_registry.get_task_class(task)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    apply_ablation_cfg(env_cfg)

    env = env_class(env_cfg, args_cli.headless)

    log_root = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)

    runner_class = eval(agent_cfg.runner_class_name)
    train_cfg = agent_cfg.to_dict()
    train_cfg["policy"]["obs_history_length"] = int(env_cfg.robot.actor_obs_history_length)
    runner = runner_class(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        runner.load(resume_path)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()
    sys.exit(0)
