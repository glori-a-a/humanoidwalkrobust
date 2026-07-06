#!/usr/bin/env python3
"""Patch tienkung_env.py so Stage 1 can read teacher obs from sim."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# TEACHER_OBS_PATCH"


def patch(content: str) -> str:
    if MARKER in content:
        return content
    if "# PHASE4_TEACHER" in content:
        raise RuntimeError(
            "Old env patch detected. Restore tienkung_env.py.bak_phase4 (or git checkout) then re-run."
        )

    if "def compute_current_observations(self):" not in content:
        raise RuntimeError("compute_current_observations not found")

    # allow teacher obs to use fresh actions instead of delayed buffer
    content = content.replace(
        "def compute_current_observations(self):",
        "def compute_current_observations(self, action_override: torch.Tensor | None = None):",
        1,
    )
    content = content.replace(
        "        action = self.action_buffer._circular_buffer.buffer[:, -1, :]\n",
        "        action = (\n"
        "            action_override\n"
        "            if action_override is not None\n"
        "            else self.action_buffer._circular_buffer.buffer[:, -1, :]\n"
        "        )\n",
        1,
    )

    teacher_block = f'''
    {MARKER}
    def init_teacher_buffers(self):
        hist = self.cfg.robot.actor_obs_history_length
        self.raw_action_buffer = CircularBuffer(
            max_len=hist, batch_size=self.num_envs, device=self.device
        )

    def teacher_obs(self, raw_actions: torch.Tensor) -> torch.Tensor:
        obs, _ = self.compute_current_observations(action_override=raw_actions)
        return obs

    def set_delay(self, delay: int, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        delay = int(max(0, delay))
        lags = torch.full((len(env_ids),), delay, dtype=torch.int, device=self.device)
        self.action_buffer.set_time_lag(lags, env_ids)

    def read_delay(self) -> torch.Tensor:
        for name in ("time_lags", "time_lag"):
            if hasattr(self.action_buffer, name):
                val = getattr(self.action_buffer, name)
                if isinstance(val, torch.Tensor):
                    return val.to(device=self.device, dtype=torch.long)
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def publish_teacher(self, raw_actions: torch.Tensor, delayed_obs: torch.Tensor) -> None:
        self.extras["teacher"] = {{
            "obs": self.teacher_obs(raw_actions),
            "delayed": delayed_obs,
            "delay": self.read_delay(),
        }}
'''

    content = content.replace(
        "        self.init_obs_buffer()\n",
        "        self.init_obs_buffer()\n        self.init_teacher_buffers()\n",
        1,
    )

    anchor = content.find("    def compute_observations(self):")
    if anchor < 0:
        raise RuntimeError("compute_observations not found")
    content = content[:anchor] + teacher_block + content[anchor:]

    content = content.replace(
        "        self.action_buffer.reset(env_ids)\n",
        "        self.action_buffer.reset(env_ids)\n        self.raw_action_buffer.reset(env_ids)\n",
        1,
    )

    old_step = """    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)"""
    new_step = """    def step(self, actions: torch.Tensor):
        self.raw_action_buffer.append(actions)
        delayed_actions = self.action_buffer.compute(actions)"""
    if old_step not in content:
        raise RuntimeError("step() anchor not found")
    content = content.replace(old_step, new_step, 1)

    content = content.replace(
        "        actor_obs, critic_obs = self.compute_observations()\n"
        "        self.extras[\"observations\"] = {\"critic\": critic_obs}\n\n"
        "        return actor_obs, reward_buf, self.reset_buf, self.extras\n",
        "        actor_obs, critic_obs = self.compute_observations()\n"
        "        self.extras[\"observations\"] = {\"critic\": critic_obs}\n"
        "        self.publish_teacher(raw_actions=actions, delayed_obs=actor_obs)\n\n"
        "        return actor_obs, reward_buf, self.reset_buf, self.extras\n",
        1,
    )

    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tienkung", default="/workspace/TienKung-Lab")
    args = parser.parse_args()

    path = Path(args.tienkung) / "legged_lab/envs/tienkung/tienkung_env.py"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    original = path.read_text(encoding="utf-8")
    patched = patch(original)
    if patched == original:
        print(f"Already patched: {path}")
        return

    backup = path.with_suffix(".py.bak_teacher")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
