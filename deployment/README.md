# Deployment-readiness layer

This directory turns the trained TienKung locomotion policy into a reviewable
deployment package. It is deliberately split into:

1. code that can be verified without robot access;
2. simulator evidence already present in this repository; and
3. hardware-originated evidence that must only be added after a real run.

Nothing in this directory claims that the controller has already run on a
physical TienKung robot. See [EVIDENCE_STATUS.md](EVIDENCE_STATUS.md) for the
current boundary.

## Architecture

```text
Isaac Lab checkpoint / scripted policy
        |
        v
policy_artifact.py ----> TorchScript / ONNX + SHA-256 manifest
        |
        v
ROS 2 policy output (/policy/action_normalized)
        |
        v
C++ safety bridge
  - joint-name mapping
  - finite/shape checks
  - position and rate limits
  - stale-state watchdog
  - explicit enable / fault state
        |
        v
official TienKung hardware adapter (not included here)
        |
        v
physical robot + hardware telemetry
```

The C++ bridge intentionally refuses activation while
`hardware_mapping_verified=false`. The final adapter to the official
`bodyctrl_msgs`/`Deploy_Tienkung` interface must be reviewed with the robot
owner before a hardware run.

## Policy artifact

Starting from a scripted policy accepted by the existing evaluation scripts:

```bash
python deployment/python/policy_artifact.py \
  --input /path/to/policy.pt \
  --output-dir deployment/artifacts/policy \
  --observation-dim <actor_observation_dimension> \
  --action-dim 20
```

The command:

- checks input/output shapes and finite outputs;
- re-saves a frozen TorchScript artifact;
- optionally exports ONNX;
- checks TorchScript parity;
- measures warm and steady-state inference latency; and
- writes a manifest containing hashes, dimensions and runtime metadata.

Generated policy binaries remain ignored by git. Commit the small JSON manifest
only after it was generated from the actual selected checkpoint.

The validated runtime wrapper is
[`python/policy_runtime.py`](python/policy_runtime.py). It rejects observations
or policy outputs with the wrong shape or non-finite values and reports
per-call inference latency:

```bash
python deployment/python/policy_runtime.py \
  --artifact deployment/artifacts/policy/policy_torchscript.pt \
  --observation-dim <actor_observation_dimension> \
  --action-dim 20 \
  --observation-json '[0.0, ...]'
```

An observation-builder node still has to reproduce the exact training-time
normalisation and feature order. The runtime wrapper does not guess that
contract.

## ROS 2 package

The generic bridge is in
[`ros2/tienkung_policy_bridge`](ros2/tienkung_policy_bridge). It uses standard
ROS 2 messages so its safety logic can be reviewed independently of proprietary
or machine-local message packages.

```bash
colcon build --packages-select tienkung_policy_bridge
source install/setup.bash
ros2 launch tienkung_policy_bridge policy_bridge.launch.py
```

Before any real run:

1. replace every `UNVERIFIED_action_*` entry with the exact hardware joint
   names and order from the robot owner's `bodyIdMap.h`;
2. fill limits from the matching URDF and hardware documentation;
3. set `hardware_mapping_verified=true` only after a two-person review;
4. connect the safe target topic to the official TienKung adapter;
5. test in simulation/data replay first; and
6. use an on-site operator, fall protection and a physical emergency stop.

## Local verification

```bash
python -m unittest discover -s deployment/tests -p 'test_*.py' -v

g++ -std=c++17 -Wall -Wextra -pedantic \
  deployment/ros2/tienkung_policy_bridge/test/test_safety_guard.cpp \
  -Ideployment/ros2/tienkung_policy_bridge/include \
  -o /tmp/test_safety_guard
/tmp/test_safety_guard
```

The same checks run in GitHub Actions.

## Evidence index

- [Current validation status](EVIDENCE_STATUS.md)
- [Failure and debugging record](evidence/FAILURE_LOG.md)
- [Hardware evidence contract](evidence/hardware_validation/README.md)
- [Collaborator attestation template](evidence/COLLABORATOR_ATTESTATION_TEMPLATE.md)
