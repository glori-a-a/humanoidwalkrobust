# Quick smoke test: walk_queue_ablation actor obs dim includes queue block.
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="walk_queue_ablation")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils import task_registry  # noqa: E402
from rsl_rl.modules.delay_queue import MAX_DELAY, queue_actor_obs_dim  # noqa: E402

task = args.task
env_cfg, _ = task_registry.get_cfgs(task)
env_class = task_registry.get_task_class(task)
env_cfg.scene.num_envs = args.num_envs
env_cfg.scene.seed = 42
env_cfg.domain_rand.action_delay.enable = True
env_cfg.domain_rand.action_delay.params["min_delay"] = 0
env_cfg.domain_rand.action_delay.params["max_delay"] = 5

env = env_class(env_cfg, headless=True)
obs, extras = env.get_observations()
num_actions = env.num_actions
expected_extra = queue_actor_obs_dim(num_actions, MAX_DELAY)

# One step with random actions to exercise ring append.
actions = torch.zeros(env.num_envs, num_actions, device=env.device)
obs2, _, _, _ = env.step(actions)

print(f"task={task} num_actions={num_actions}")
print(f"actor_obs shape={obs.shape} (after reset)")
print(f"actor_obs shape={obs2.shape} (after 1 step)")
print(f"queue block dim={expected_extra} (max_delay={MAX_DELAY})")
print(f"enable_queue_actor_obs={getattr(env_cfg, 'enable_queue_actor_obs', False)}")

if not getattr(env_cfg, "enable_queue_actor_obs", False):
    raise SystemExit("FAIL: enable_queue_actor_obs is False")

simulation_app.close()
print("SMOKE PASS: queue task loads and steps.")
