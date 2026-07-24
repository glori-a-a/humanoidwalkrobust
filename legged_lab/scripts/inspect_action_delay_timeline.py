"""Phase 0: verify action-delay timing in the live TienKung / Isaac Lab env.

Confirms, with numbered actions, that our issued-action ring buffer and the
canonical ``pre_execution_actions`` extraction agree with what the Isaac action
buffer actually applies, at delay = 0, 1, 3. No policy is loaded; every step
sets a *tiny numbered offset* (tag * SCALE) so the robot mostly holds its
default pose (few resets) while applied / queue values stay decodable.

Guarantees (per design review):
  1. The DelayBuffer.compute wrapper calls the real compute exactly once and
     only caches its return value; it never re-invokes compute.
  2. On per-env auto-reset (done), the matching rows of our self-maintained
     issued-action ring are zeroed so it mirrors Isaac's buffer reset.
  3. pre_execution_actions is built at DECISION TIME t, BEFORE appending u_t.

Assertions are only applied to envs whose window has been reset-free for >= d
steps (post-reset the delay buffer is not yet warm, so applied != issued - d).

Checks (eligible envs, after warmup):
  1. decode(applied) == issued - d           (d>0)   / == issued (d==0)
  2. our ring[-d]     == applied
  3. pre_exec[0]      == applied              (d>0)
  4. pre_exec[i]      == u_{t-d+i}            (oldest-first, no off-by-one)
  5. queue length (mask.sum) == d, padding zero
  6. d == 0 -> empty queue, applied == issued (identity)

Run on the GPU pod (Isaac env active):
  python legged_lab/scripts/inspect_action_delay_timeline.py \
      --task walk --num_envs 16 --seed 42 --delays 0,1,3 --headless
"""
import argparse

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
import legged_lab.utils.cli_args as cli_args

MAX_DELAY = 8
SCALE = 1e-3  # tiny per-step offset so numbered actions ~ hold default pose

parser = argparse.ArgumentParser(description="Phase 0 action-delay timeline inspection.")
parser.add_argument("--task", type=str, default="walk")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--delays", type=str, default="0,1,3")
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--measure", type=int, default=16)
parser.add_argument("--min_samples", type=int, default=8, help="Required eligible asserts per delay.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa: F401,F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def extract_pre_execution(ring: torch.Tensor, delay: torch.Tensor, max_delay: int):
    """Canonical rule (mirrors tests/test_delay_queue_order.py)."""
    B, L, A = ring.shape
    ar = torch.arange(max_delay, device=ring.device)
    mask = ar[None, :] < delay[:, None]
    idx = (L - delay[:, None] + ar[None, :]).clamp(0, L - 1)
    idx_exp = idx[..., None].expand(B, max_delay, A)
    gathered = torch.gather(ring, 1, idx_exp)
    pre_exec = gathered * mask[..., None].to(ring.dtype)
    return pre_exec, mask


class NumberedRing:
    """Our own issued-action ring, newest at index L-1 (Isaac append order)."""

    def __init__(self, batch, length, action_dim, device):
        self.buf = torch.zeros(batch, length, action_dim, device=device)

    def append(self, actions: torch.Tensor):
        self.buf = torch.roll(self.buf, shifts=-1, dims=1)
        self.buf[:, -1, :] = actions

    def reset(self, env_ids: torch.Tensor):
        if env_ids.numel() > 0:
            self.buf[env_ids] = 0.0

    @property
    def ring(self):
        return self.buf


def decode_tag(x: torch.Tensor) -> torch.Tensor:
    """Recover the integer step tag from a tiny numbered offset."""
    return torch.round(x / SCALE)


def configure_eval(env_cfg):
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"
    env_cfg.domain_rand.action_delay.enable = True
    env_cfg.domain_rand.action_delay.params["min_delay"] = 0
    env_cfg.domain_rand.action_delay.params["max_delay"] = MAX_DELAY


def dones_to_ids(dones, device):
    if dones is None:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.where(dones.reshape(-1).bool())[0]


def main():
    delays = [int(x) for x in args_cli.delays.split(",") if x.strip() != ""]

    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    configure_eval(env_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.seed = args_cli.seed
    env_cfg.scene.seed = args_cli.seed

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    device = env.device
    B = env.num_envs
    A = env.num_actions

    # Guarantee 1: wrap compute so it runs the real compute exactly once per step
    # and only caches the returned (applied) tensor.
    captured = {}
    orig_compute = env.action_buffer.compute

    def wrapped_compute(a):
        out = orig_compute(a)          # single real call
        captured["applied"] = out.detach().clone()
        return out

    env.action_buffer.compute = wrapped_compute

    # Mirror the proven eval loop: prime observations once, no explicit env.reset().
    print("env ready; priming observations...", flush=True)
    env.get_observations()
    print("observations primed.", flush=True)

    all_ok = True
    for d in delays:
        print(f"\n================  delay = {d}  ================", flush=True)
        env.set_delay(d)
        ring = NumberedRing(B, MAX_DELAY, A, device)
        # after a delay change, treat all envs as freshly warming (ineligible until warm)
        steps_since_reset = torch.zeros(B, dtype=torch.long, device=device)
        step = 0

        def do_step(assert_phase):
            nonlocal all_ok, step
            t = step
            delay_vec = torch.full((B,), d, dtype=torch.long, device=device)

            # Guarantee 3: extract queue at decision time, BEFORE appending u_t.
            ring_snapshot = ring.ring.clone()
            pre_exec, mask = extract_pre_execution(ring_snapshot, delay_vec, MAX_DELAY)
            eligible = steps_since_reset >= d  # reset-free window long enough

            actions = torch.full((B, A), float(t) * SCALE, device=device)
            ring.append(actions)
            steps_since_reset.add_(1)
            with torch.inference_mode():
                out = env.step(actions)
            dones = out[2]
            applied = captured["applied"]

            n_assert = 0
            if assert_phase:
                elig_ids = torch.where(eligible)[0]
                for b in elig_ids.tolist():
                    app_tag = int(decode_tag(applied[b, 0]).item())
                    exp_app = t if d == 0 else t - d
                    if app_tag != exp_app:
                        print(f"  FAIL[env{b}] applied tag {app_tag} != {exp_app}"); all_ok = False
                    if int(mask[b].sum().item()) != d:
                        print(f"  FAIL[env{b}] qlen {int(mask[b].sum())} != {d}"); all_ok = False
                    if d > 0:
                        ring_d_tag = int(decode_tag(ring_snapshot[b, -d, 0]).item())
                        if ring_d_tag != app_tag:
                            print(f"  FAIL[env{b}] ring[-d] at decision {ring_d_tag} != applied {app_tag}"); all_ok = False
                        pre0_tag = int(decode_tag(pre_exec[b, 0, 0]).item())
                        if pre0_tag != app_tag:
                            print(f"  FAIL[env{b}] pre_exec[0] {pre0_tag} != applied {app_tag}"); all_ok = False
                        for i in range(d):
                            tag_i = int(decode_tag(pre_exec[b, i, 0]).item())
                            if tag_i != t - d + i:
                                print(f"  FAIL[env{b}] pre_exec[{i}] {tag_i} != {t - d + i}"); all_ok = False
                        if bool((pre_exec[b, d:].abs() > 1e-9).any()):
                            print(f"  FAIL[env{b}] padding not zero"); all_ok = False
                    n_assert += 1

            # Guarantee 2: zero our ring rows for envs that just auto-reset.
            reset_ids = dones_to_ids(dones, device)
            if reset_ids.numel() > 0:
                ring.reset(reset_ids)
                steps_since_reset[reset_ids] = 0
            step += 1
            if assert_phase or step % 5 == 0:
                print(f"    step={t} issued={t} applied={int(decode_tag(applied[0,0]).item())} "
                      f"eligible={int(eligible.sum())}", flush=True)
            return n_assert

        for wi in range(args_cli.warmup):
            do_step(assert_phase=False)
            if (wi + 1) % 10 == 0:
                print(f"  warmup {wi + 1}/{args_cli.warmup}", flush=True)

        total_asserts = 0
        for mi in range(args_cli.measure):
            total_asserts += do_step(assert_phase=True)
            print(f"  measure {mi + 1}/{args_cli.measure} (asserts so far {total_asserts})", flush=True)

        print(f"  eligible assertions for delay {d}: {total_asserts}")
        if total_asserts < args_cli.min_samples:
            print(f"  FAIL: only {total_asserts} eligible samples (< {args_cli.min_samples}); "
                  f"increase --num_envs / --measure or reduce resets.")
            all_ok = False

        # cross-check our ring vs env.raw_action_buffer newest entry
        try:
            env_raw = env.raw_action_buffer._circular_buffer.buffer[:, -1, :]
            same = torch.allclose(env_raw, ring.ring[:, -1, :], atol=1e-6)
            print(f"  raw_action_buffer newest matches our ring: {bool(same)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (skipped raw_action_buffer cross-check: {exc})")

    print("\n" + "=" * 46)
    if all_ok:
        print("PHASE 0 PASS: timing verified for delays", delays)
        print("CANONICAL RULE: pre_exec[:, i] = ring[:, L - d + i] for i < d else 0;")
        print(f"                mask[:, i] = (i < d); delay_norm = d / {MAX_DELAY};")
        print("                pre_exec[0] == applied == u_(t-d); u_t excluded;")
        print("                extraction at decision time BEFORE appending u_t.")
    else:
        print("PHASE 0 FAIL: see FAIL lines above (off-by-one / indexing / sample count).")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
