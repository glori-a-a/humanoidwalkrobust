"""Runtime joint mapping, state machine and command safety for deployment.

This module is simulator/hardware agnostic and uses only the Python standard
library so its safety behaviour can be unit-tested in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Sequence


class Mode(str, Enum):
    STARTUP = "startup"
    STANDBY = "standby"
    ACTIVE = "active"
    FAULT = "fault"


@dataclass(frozen=True)
class JointSpec:
    name: str
    lower: float
    upper: float
    default: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("joint name must not be empty")
        if not all(math.isfinite(v) for v in (self.lower, self.upper, self.default)):
            raise ValueError(f"{self.name}: limits/default must be finite")
        if self.lower >= self.upper:
            raise ValueError(f"{self.name}: lower must be smaller than upper")
        if not self.lower <= self.default <= self.upper:
            raise ValueError(f"{self.name}: default position is outside limits")


@dataclass(frozen=True)
class GuardConfig:
    action_scale: float = 0.25
    max_target_velocity: float = 1.0
    state_timeout_s: float = 0.1
    max_abs_roll_pitch_rad: float = 0.8

    def __post_init__(self) -> None:
        values = (
            self.action_scale,
            self.max_target_velocity,
            self.state_timeout_s,
            self.max_abs_roll_pitch_rad,
        )
        if not all(math.isfinite(v) and v > 0.0 for v in values):
            raise ValueError("all guard parameters must be finite and positive")


@dataclass(frozen=True)
class RobotState:
    timestamp_s: float
    joint_names: Sequence[str]
    position: Sequence[float]
    velocity: Sequence[float]
    roll_rad: float
    pitch_rad: float


@dataclass(frozen=True)
class GuardOutput:
    mode: Mode
    target_position: tuple[float, ...]
    reason: str


class JointMapper:
    """Map unordered ROS JointState arrays into the policy's declared order."""

    def __init__(self, policy_joint_names: Sequence[str]) -> None:
        if not policy_joint_names:
            raise ValueError("policy joint list must not be empty")
        if len(set(policy_joint_names)) != len(policy_joint_names):
            raise ValueError("policy joint list contains duplicates")
        self.policy_joint_names = tuple(policy_joint_names)

    def indices(self, incoming_names: Sequence[str]) -> tuple[int, ...]:
        if len(set(incoming_names)) != len(incoming_names):
            raise ValueError("incoming JointState contains duplicate names")
        lookup = {name: i for i, name in enumerate(incoming_names)}
        missing = [name for name in self.policy_joint_names if name not in lookup]
        if missing:
            raise ValueError(f"missing policy joints: {missing}")
        return tuple(lookup[name] for name in self.policy_joint_names)

    def reorder(
        self, incoming_names: Sequence[str], values: Sequence[float]
    ) -> tuple[float, ...]:
        if len(incoming_names) != len(values):
            raise ValueError("joint names and value arrays have different lengths")
        return tuple(values[i] for i in self.indices(incoming_names))


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


class SafetyGuard:
    """Fail-closed guard around normalized policy actions."""

    def __init__(self, joints: Sequence[JointSpec], config: GuardConfig) -> None:
        if not joints:
            raise ValueError("at least one joint specification is required")
        self.joints = tuple(joints)
        self.config = config
        self.mapper = JointMapper([joint.name for joint in joints])
        self.mode = Mode.STARTUP
        self.fault_reason = ""
        self._state: RobotState | None = None
        self._ordered_position: tuple[float, ...] | None = None
        self._last_target = tuple(joint.default for joint in joints)
        self._last_command_time_s: float | None = None

    def update_state(self, state: RobotState) -> None:
        values = (
            state.timestamp_s,
            state.roll_rad,
            state.pitch_rad,
            *state.position,
            *state.velocity,
        )
        if not _finite(values):
            self._fault("non-finite robot state")
            return
        try:
            ordered_q = self.mapper.reorder(state.joint_names, state.position)
            self.mapper.reorder(state.joint_names, state.velocity)
        except ValueError as exc:
            self._fault(str(exc))
            return
        if any(
            q < spec.lower or q > spec.upper
            for q, spec in zip(ordered_q, self.joints)
        ):
            self._fault("measured joint position outside configured limits")
            return
        if (
            abs(state.roll_rad) > self.config.max_abs_roll_pitch_rad
            or abs(state.pitch_rad) > self.config.max_abs_roll_pitch_rad
        ):
            self._fault("body attitude exceeds configured safety envelope")
            return
        self._state = state
        self._ordered_position = ordered_q
        if self.mode is Mode.STARTUP:
            self.mode = Mode.STANDBY

    def enable(self, now_s: float) -> None:
        self._require_finite_scalar(now_s, "enable timestamp")
        if self.mode is Mode.FAULT:
            raise RuntimeError(f"cannot enable while faulted: {self.fault_reason}")
        if self._state is None or self._ordered_position is None:
            raise RuntimeError("cannot enable before a valid robot state")
        if now_s - self._state.timestamp_s > self.config.state_timeout_s:
            raise RuntimeError("cannot enable with stale robot state")
        self._last_target = self._ordered_position
        self._last_command_time_s = now_s
        self.mode = Mode.ACTIVE

    def standby(self) -> None:
        if self.mode is not Mode.FAULT:
            self.mode = Mode.STANDBY

    def reset_fault(self) -> None:
        self.mode = Mode.STARTUP
        self.fault_reason = ""
        self._state = None
        self._ordered_position = None
        self._last_command_time_s = None
        self._last_target = tuple(joint.default for joint in self.joints)

    def command(
        self, normalized_action: Sequence[float], now_s: float
    ) -> GuardOutput:
        self._require_finite_scalar(now_s, "command timestamp")
        if self.mode is not Mode.ACTIVE:
            return GuardOutput(self.mode, self._last_target, "guard is not active")
        if self._state is None:
            self._fault("robot state unavailable")
            return GuardOutput(self.mode, self._last_target, self.fault_reason)
        age = now_s - self._state.timestamp_s
        if age < 0.0 or age > self.config.state_timeout_s:
            self._fault(f"robot state stale ({age:.6f} s)")
            return GuardOutput(self.mode, self._last_target, self.fault_reason)
        if len(normalized_action) != len(self.joints):
            self._fault(
                f"action dimension {len(normalized_action)} != {len(self.joints)}"
            )
            return GuardOutput(self.mode, self._last_target, self.fault_reason)
        if not _finite(normalized_action):
            self._fault("non-finite policy action")
            return GuardOutput(self.mode, self._last_target, self.fault_reason)

        previous_time = self._last_command_time_s
        dt = 0.0 if previous_time is None else now_s - previous_time
        if dt <= 0.0:
            self._fault("non-positive command timestep")
            return GuardOutput(self.mode, self._last_target, self.fault_reason)

        max_delta = self.config.max_target_velocity * dt
        targets: list[float] = []
        for action, spec, previous in zip(
            normalized_action, self.joints, self._last_target
        ):
            bounded_action = _clip(float(action), -1.0, 1.0)
            desired = spec.default + self.config.action_scale * bounded_action
            desired = _clip(desired, spec.lower, spec.upper)
            rate_limited = _clip(desired, previous - max_delta, previous + max_delta)
            targets.append(_clip(rate_limited, spec.lower, spec.upper))

        self._last_target = tuple(targets)
        self._last_command_time_s = now_s
        return GuardOutput(Mode.ACTIVE, self._last_target, "ok")

    def _fault(self, reason: str) -> None:
        self.mode = Mode.FAULT
        self.fault_reason = reason

    @staticmethod
    def _require_finite_scalar(value: float, label: str) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")

