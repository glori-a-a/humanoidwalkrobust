import math


def make_guard(joints, config=None):
    if not joints:
        raise ValueError("joints must not be empty")

    names = [joint["name"] for joint in joints]
    if len(names) != len(set(names)):
        raise ValueError("joint names must be unique")

    for joint in joints:
        values = [joint["lower"], joint["upper"], joint["default"]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint values must be finite")
        if joint["lower"] >= joint["upper"]:
            raise ValueError("invalid joint limits")
        if not joint["lower"] <= joint["default"] <= joint["upper"]:
            raise ValueError("default position is outside joint limits")

    settings = {
        "action_scale": 0.25,
        "max_target_velocity": 1.0,
        "state_timeout": 0.1,
        "max_tilt": 0.8,
    }
    if config:
        settings.update(config)
    if not all(math.isfinite(value) and value > 0 for value in settings.values()):
        raise ValueError("guard settings must be positive")

    return {
        "joints": joints,
        "config": settings,
        "mode": "startup",
        "fault": "",
        "state_time": None,
        "position": None,
        "last_target": [joint["default"] for joint in joints],
        "last_command_time": None,
    }


def reorder_values(expected_names, incoming_names, values):
    if len(incoming_names) != len(values):
        raise ValueError("joint names and values have different lengths")
    if len(incoming_names) != len(set(incoming_names)):
        raise ValueError("incoming joint names contain duplicates")

    lookup = dict(zip(incoming_names, values))
    missing = [name for name in expected_names if name not in lookup]
    if missing:
        raise ValueError(f"missing joints: {missing}")
    return [lookup[name] for name in expected_names]


def set_fault(guard, reason):
    guard["mode"] = "fault"
    guard["fault"] = reason


def update_state(guard, timestamp, names, position, velocity, roll, pitch):
    values = [timestamp, roll, pitch, *position, *velocity]
    if not all(math.isfinite(value) for value in values):
        set_fault(guard, "non-finite robot state")
        return False

    expected = [joint["name"] for joint in guard["joints"]]
    try:
        ordered_position = reorder_values(expected, names, position)
        reorder_values(expected, names, velocity)
    except ValueError as error:
        set_fault(guard, str(error))
        return False

    for value, joint in zip(ordered_position, guard["joints"]):
        if value < joint["lower"] or value > joint["upper"]:
            set_fault(guard, "joint position outside limits")
            return False

    if abs(roll) > guard["config"]["max_tilt"]:
        set_fault(guard, "body tilt outside limit")
        return False
    if abs(pitch) > guard["config"]["max_tilt"]:
        set_fault(guard, "body tilt outside limit")
        return False

    guard["state_time"] = timestamp
    guard["position"] = ordered_position
    if guard["mode"] == "startup":
        guard["mode"] = "standby"
    return True


def enable(guard, now):
    if guard["mode"] == "fault":
        return False
    if guard["position"] is None:
        return False
    age = now - guard["state_time"]
    if age < 0 or age > guard["config"]["state_timeout"]:
        return False

    guard["last_target"] = list(guard["position"])
    guard["last_command_time"] = now
    guard["mode"] = "active"
    return True


def standby(guard):
    if guard["mode"] != "fault":
        guard["mode"] = "standby"


def reset_guard(guard):
    guard["mode"] = "startup"
    guard["fault"] = ""
    guard["state_time"] = None
    guard["position"] = None
    guard["last_target"] = [joint["default"] for joint in guard["joints"]]
    guard["last_command_time"] = None


def clip(value, lower, upper):
    return min(max(value, lower), upper)


def make_command(guard, action, now):
    if guard["mode"] != "active":
        return guard["mode"], list(guard["last_target"]), "guard is not active"

    age = now - guard["state_time"]
    if age < 0 or age > guard["config"]["state_timeout"]:
        set_fault(guard, "robot state is stale")
        return guard["mode"], list(guard["last_target"]), guard["fault"]

    if len(action) != len(guard["joints"]):
        set_fault(guard, "wrong action size")
        return guard["mode"], list(guard["last_target"]), guard["fault"]
    if not all(math.isfinite(value) for value in action):
        set_fault(guard, "non-finite policy action")
        return guard["mode"], list(guard["last_target"]), guard["fault"]

    dt = now - guard["last_command_time"]
    if dt <= 0:
        set_fault(guard, "invalid command time")
        return guard["mode"], list(guard["last_target"]), guard["fault"]

    max_change = guard["config"]["max_target_velocity"] * dt
    targets = []
    for value, joint, previous in zip(
        action, guard["joints"], guard["last_target"]
    ):
        value = clip(value, -1.0, 1.0)
        target = joint["default"] + guard["config"]["action_scale"] * value
        target = clip(target, joint["lower"], joint["upper"])
        target = clip(target, previous - max_change, previous + max_change)
        targets.append(clip(target, joint["lower"], joint["upper"]))

    guard["last_target"] = targets
    guard["last_command_time"] = now
    return "active", targets, "ok"
