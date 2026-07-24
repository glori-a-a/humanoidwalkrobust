from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deployment" / "python"))

from runtime_guard import (  # noqa: E402
    GuardConfig,
    JointMapper,
    JointSpec,
    Mode,
    RobotState,
    SafetyGuard,
)


def joints():
    return [
        JointSpec("left", -1.0, 1.0, 0.0),
        JointSpec("right", -0.5, 0.5, 0.1),
    ]


def valid_state(timestamp=1.0):
    return RobotState(
        timestamp_s=timestamp,
        joint_names=["right", "left"],
        position=[0.1, 0.0],
        velocity=[0.0, 0.0],
        roll_rad=0.0,
        pitch_rad=0.0,
    )


class JointMapperTests(unittest.TestCase):
    def test_reorders_by_name(self):
        mapper = JointMapper(["left", "right"])
        self.assertEqual(
            mapper.reorder(["right", "left"], [20.0, 10.0]), (10.0, 20.0)
        )

    def test_rejects_missing_joint(self):
        mapper = JointMapper(["left", "right"])
        with self.assertRaisesRegex(ValueError, "missing policy joints"):
            mapper.indices(["left"])


class SafetyGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = SafetyGuard(
            joints(),
            GuardConfig(
                action_scale=0.5,
                max_target_velocity=1.0,
                state_timeout_s=0.1,
                max_abs_roll_pitch_rad=0.8,
            ),
        )

    def test_requires_state_before_enable(self):
        with self.assertRaisesRegex(RuntimeError, "valid robot state"):
            self.guard.enable(1.0)

    def test_rate_limits_and_clips_policy_action(self):
        self.guard.update_state(valid_state())
        self.guard.enable(1.0)
        out = self.guard.command([4.0, -4.0], 1.02)
        self.assertEqual(out.mode, Mode.ACTIVE)
        self.assertAlmostEqual(out.target_position[0], 0.02)
        self.assertAlmostEqual(out.target_position[1], 0.08)

    def test_stale_state_faults_closed(self):
        self.guard.update_state(valid_state())
        self.guard.enable(1.0)
        out = self.guard.command([0.0, 0.0], 1.2)
        self.assertEqual(out.mode, Mode.FAULT)
        self.assertIn("stale", out.reason)

    def test_non_finite_action_faults_closed(self):
        self.guard.update_state(valid_state())
        self.guard.enable(1.0)
        out = self.guard.command([math.nan, 0.0], 1.01)
        self.assertEqual(out.mode, Mode.FAULT)
        self.assertIn("non-finite", out.reason)

    def test_wrong_action_dimension_faults_closed(self):
        self.guard.update_state(valid_state())
        self.guard.enable(1.0)
        out = self.guard.command([0.0], 1.01)
        self.assertEqual(out.mode, Mode.FAULT)
        self.assertIn("dimension", out.reason)

    def test_excess_attitude_faults_on_state_update(self):
        state = valid_state()
        tilted = RobotState(
            timestamp_s=state.timestamp_s,
            joint_names=state.joint_names,
            position=state.position,
            velocity=state.velocity,
            roll_rad=0.9,
            pitch_rad=0.0,
        )
        self.guard.update_state(tilted)
        self.assertEqual(self.guard.mode, Mode.FAULT)


if __name__ == "__main__":
    unittest.main()

