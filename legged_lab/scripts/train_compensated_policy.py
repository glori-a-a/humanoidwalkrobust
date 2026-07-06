# Stage 2: load Stage-1 predictor, fine-tune walk policy with delay DR (same idea as strong baseline).

import argparse
import os
import sys
from datetime import datetime

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

import legged_lab.utils.cli_args as cli_args  # isort: skip

# --- CLI ---
parser = argparse.ArgumentParser(description="Stage 2 predictor + PPO fine-tune.")
parser.add_argument("--task", type=str, default="walk_delay_comp")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--predictor_path", type=str, required=True)
parser.add_argument("--init_policy_path", type=str, default=None)
parser.add_argument("--freeze_predictor", action="store_true")

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
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner  # noqa: F401 — for eval(runner_class_name)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def enable_delay_dr(env_cfg, dmin=0, dmax=5):
    env_cfg.domain_rand.action_delay.enable = True
    env_cfg.domain_rand.action_delay.params["min_delay"] = dmin
    env_cfg.domain_rand.action_delay.params["max_delay"] = dmax


def load_predictor(actor_critic, path, freeze):
    device = next(actor_critic.parameters()).device
    ckpt = torch.load(path, map_location=device)
    actor_critic.predictor.load_state_dict(ckpt["model_state_dict"])
    print(f"[Stage2] predictor from {path}")
    if freeze:
        for p in actor_critic.predictor.parameters():
            p.requires_grad = False
        print("[Stage2] predictor frozen")


def warm_start_policy(actor_critic, path):
    # copy actor/critic weights from strong/nominal ckpt when shapes match
    sd = torch.load(path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    own = actor_critic.state_dict()
    loaded = 0
    for key, val in sd.items():
        if key.startswith("predictor."):
            continue
        if key in own and own[key].shape == val.shape:
            own[key] = val
            loaded += 1
    actor_critic.load_state_dict(own, strict=False)
    print(f"[Stage2] warm-start {loaded} tensors from {path}")


def train():
    task = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(task)
    env_class = task_registry.get_task_class(task)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    enable_delay_dr(env_cfg)

    print(
        f"[INFO] Stage 2 delay DR U[0,5], freeze_predictor={args_cli.freeze_predictor}"
    )

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
    train_cfg["policy"]["predictor_hidden_dims"] = [256, 256]

    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

    load_predictor(runner.alg.policy, args_cli.predictor_path, args_cli.freeze_predictor)

    if args_cli.init_policy_path:
        warm_start_policy(runner.alg.policy, args_cli.init_policy_path)
    elif agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] resume {resume_path}")
        runner.load(resume_path)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()
    sys.exit(0)
