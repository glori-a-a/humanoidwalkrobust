"""Record Isaac Sim walk playback video (headless + cameras, optional action delay)."""
import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
import legged_lab.utils.cli_args as cli_args
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner  # noqa: F401

parser = argparse.ArgumentParser(description="Record TienKung policy playback video.")
parser.add_argument("--task", type=str, default="walk")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--policy_path", type=str, default=None, help="TorchScript JIT (nominal/strong).")
parser.add_argument("--action_delay_steps", type=int, default=0)
parser.add_argument("--delay_mode", type=str, default="fixed", choices=["fixed", "uniform"])
parser.add_argument("--lin_vel_x", type=float, default=1.0)
parser.add_argument("--lin_vel_y", type=float, default=0.0)
parser.add_argument("--video_path", type=str, default="/workspace/videos/walk_demo.mp4")
parser.add_argument("--video_length", type=int, default=400, help="Frames after warmup (~13s @ 30fps).")
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from legged_lab.envs import *  # noqa: F401,F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def configure_play_baseline(env_cfg, lin_vel_x: float, lin_vel_y: float):
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


def apply_delay(env_cfg, delay_steps: int, delay_mode: str):
    d = max(0, int(delay_steps))
    if d == 0 and delay_mode == "fixed":
        env_cfg.domain_rand.action_delay.enable = False
        return
    env_cfg.domain_rand.action_delay.enable = True
    if delay_mode == "fixed":
        env_cfg.domain_rand.action_delay.params["min_delay"] = d
        env_cfg.domain_rand.action_delay.params["max_delay"] = d
    else:
        env_cfg.domain_rand.action_delay.params["min_delay"] = 0
        env_cfg.domain_rand.action_delay.params["max_delay"] = d


def load_policy(env, agent_cfg, policy_path, checkpoint_name):
    if policy_path:
        path = retrieve_file_path(policy_path) if not os.path.isabs(policy_path) else policy_path
        path = os.path.abspath(path)
        print(f"[INFO] Loading TorchScript policy: {path}")
        policy = torch.jit.load(path, map_location=env.device)
        policy.eval()
        return policy

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    if checkpoint_name:
        agent_cfg.load_checkpoint = checkpoint_name
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading RSL-RL checkpoint: {resume_path}")
    runner_class = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    return runner.get_inference_policy(device=env.device)


def main():
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    configure_play_baseline(env_cfg, args_cli.lin_vel_x, args_cli.lin_vel_y)
    apply_delay(env_cfg, args_cli.action_delay_steps, args_cli.delay_mode)

    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if getattr(args_cli, "seed", None) is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)
    policy = load_policy(env, agent_cfg, args_cli.policy_path, args_cli.checkpoint)

    import omni.replicator.core as rep
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=np.array([4.0, 4.0, 2.5]), target=np.array([0.0, 0.0, 1.0]))
    render_product = rep.create.render_product("/OmniverseKit_Persp", (args_cli.width, args_cli.height))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator.attach([render_product])

    obs, _ = env.get_observations()
    frames = []
    warmup = 10
    max_steps = args_cli.video_length + warmup

    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        env.sim.render()
        rgb_data = rgb_annotator.get_data()
        if rgb_data.size == 0:
            continue
        frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)[:, :, :3]
        if step >= warmup:
            frames.append(frame.copy())
        if dones.any() and step >= warmup + 30:
            break

    os.makedirs(os.path.dirname(args_cli.video_path) or ".", exist_ok=True)
    imageio.mimwrite(args_cli.video_path, frames, fps=args_cli.fps)
    print(f"[INFO] Saved video: {args_cli.video_path} ({len(frames)} frames, delay={args_cli.action_delay_steps})")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    sys.exit(0)
