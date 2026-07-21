#!/usr/bin/env python3
"""Patch tienkung_env.py to append queue actor observations (v2 queue interface).

Run AFTER the teacher-obs patch (env_patch.py). Idempotent via QUEUE_OBS_PATCH marker.

When cfg.enable_queue_actor_obs is True, actor obs becomes:
  [original_obs_history, flat(pre_execution_actions), queue_mask, delay_normalized]

Critic obs is unchanged. Uses our IssuedActionRing, not Isaac private buffer indices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# QUEUE_OBS_PATCH"
TEACHER_MARKER = "# TEACHER_OBS_PATCH"


def patch(content: str) -> str:
    if MARKER in content:
        return content
    if TEACHER_MARKER not in content:
        raise RuntimeError("Run env_patch.py (teacher patch) before queue-obs patch.")

    queue_block = f'''
    {MARKER}
    def init_queue_obs_buffers(self):
        from rsl_rl.modules.delay_queue import IssuedActionRing, MAX_DELAY

        self._issued_ring = IssuedActionRing(
            self.num_envs, MAX_DELAY, self.num_actions, self.device
        )

    def _append_queue_actor_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
        from rsl_rl.modules.delay_queue import append_queue_to_actor_obs

        delay = self.read_delay()
        return append_queue_to_actor_obs(actor_obs, self._issued_ring.ring, delay)
'''

    anchor = content.find("    def compute_observations(self):")
    if anchor < 0:
        raise RuntimeError("compute_observations not found")
    content = content[:anchor] + queue_block + content[anchor:]

    content = content.replace(
        "        self.init_teacher_buffers()\n",
        "        self.init_teacher_buffers()\n        self.init_queue_obs_buffers()\n",
        1,
    )

    # TienKung-Lab: append queue block after clip, before return (history-buffer path).
    old_clip_return = """        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs"""
    new_clip_return = """        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        if getattr(self.cfg, "enable_queue_actor_obs", False):
            actor_obs = self._append_queue_actor_obs(actor_obs)

        return actor_obs, critic_obs"""

    if old_clip_return in content:
        content = content.replace(old_clip_return, new_clip_return, 1)
    elif "enable_queue_actor_obs" not in content:
        old = "        return actor_obs, critic_obs\n"
        new = (
            "        if getattr(self.cfg, \"enable_queue_actor_obs\", False):\n"
            "            actor_obs = self._append_queue_actor_obs(actor_obs)\n"
            "        return actor_obs, critic_obs\n"
        )
        if content.count(old) != 1:
            raise RuntimeError(
                "Could not find compute_observations return anchor; "
                "update env_patch_queue_obs.py for this tienkung_env.py version."
            )
        content = content.replace(old, new, 1)

    old_step = """    def step(self, actions: torch.Tensor):
        self.raw_action_buffer.append(actions)
        delayed_actions = self.action_buffer.compute(actions)"""
    new_step = """    def step(self, actions: torch.Tensor):
        self.raw_action_buffer.append(actions)
        if getattr(self.cfg, "enable_queue_actor_obs", False):
            self._issued_ring.append(actions)
        delayed_actions = self.action_buffer.compute(actions)"""
    if old_step not in content:
        raise RuntimeError("step() anchor not found for queue-obs patch")
    content = content.replace(old_step, new_step, 1)

    content = content.replace(
        "        self.raw_action_buffer.reset(env_ids)\n",
        "        self.raw_action_buffer.reset(env_ids)\n"
        "        if getattr(self.cfg, \"enable_queue_actor_obs\", False):\n"
        "            self._issued_ring.reset(env_ids)\n",
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
        print(f"Already patched (queue obs): {path}")
        return

    path.write_text(patched, encoding="utf-8")
    print(f"Queue-obs patched: {path}")


if __name__ == "__main__":
    main()
