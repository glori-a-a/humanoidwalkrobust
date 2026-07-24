import math
from pathlib import Path
import sys
import unittest


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "deployment" / "python"))

from runtime_guard import enable
from runtime_guard import make_command
from runtime_guard import make_guard
from runtime_guard import reorder_values
from runtime_guard import update_state


def joints():
    return [
        {"name": "left", "lower": -1.0, "upper": 1.0, "default": 0.0},
        {"name": "right", "lower": -0.5, "upper": 0.5, "default": 0.1},
    ]


def new_guard():
    return make_guard(
        joints(),
        {
            "action_scale": 0.5,
            "max_target_velocity": 1.0,
            "state_timeout": 0.1,
            "max_tilt": 0.8,
        },
    )


def add_state(guard, timestamp=1.0, roll=0.0, pitch=0.0):
    return update_state(
        guard,
        timestamp,
        ["right", "left"],
        [0.1, 0.0],
        [0.0, 0.0],
        roll,
        pitch,
    )


class RuntimeGuardTests(unittest.TestCase):
    def test_joint_order(self):
        values = reorder_values(["left", "right"], ["right", "left"], [2.0, 1.0])
        self.assertEqual(values, [1.0, 2.0])

    def test_missing_joint(self):
        with self.assertRaises(ValueError):
            reorder_values(["left", "right"], ["left"], [1.0])

    def test_enable_needs_state(self):
        guard = new_guard()
        self.assertFalse(enable(guard, 1.0))

    def test_action_limit(self):
        guard = new_guard()
        add_state(guard)
        self.assertTrue(enable(guard, 1.0))
        mode, target, reason = make_command(guard, [4.0, -4.0], 1.02)
        self.assertEqual(mode, "active")
        self.assertEqual(reason, "ok")
        self.assertAlmostEqual(target[0], 0.02)
        self.assertAlmostEqual(target[1], 0.08)

    def test_stale_state(self):
        guard = new_guard()
        add_state(guard)
        enable(guard, 1.0)
        mode, _, reason = make_command(guard, [0.0, 0.0], 1.2)
        self.assertEqual(mode, "fault")
        self.assertIn("stale", reason)

    def test_nan_action(self):
        guard = new_guard()
        add_state(guard)
        enable(guard, 1.0)
        mode, _, _ = make_command(guard, [math.nan, 0.0], 1.01)
        self.assertEqual(mode, "fault")

    def test_wrong_action_size(self):
        guard = new_guard()
        add_state(guard)
        enable(guard, 1.0)
        mode, _, _ = make_command(guard, [0.0], 1.01)
        self.assertEqual(mode, "fault")

    def test_tilt_limit(self):
        guard = new_guard()
        self.assertFalse(add_state(guard, roll=0.9))
        self.assertEqual(guard["mode"], "fault")


if __name__ == "__main__":
    unittest.main()
