# Evidence status

Last reviewed: 2026-07-24

This is the single source of truth for claims made from this repository.

| Claim | Status | Verifiable evidence | What it supports |
|---|---|---|---|
| Isaac Lab training and evaluation | Completed | `legged_lab/`, `rsl_rl/`, `results/csv/`, `results/figures/` | Simulation-based robot learning and quantitative evaluation |
| Action-delay modelling | Completed | `rsl_rl/modules/delay_queue.py`, environment patches and delay-sweep CSVs | Explicit latency modelling rather than a presentation-only demo |
| Queue ordering and clamp safety tests | Completed | `tests/test_delay_queue_order.py`, `tests/test_queue_clamp.py` | Off-by-one and queue-ablation behaviour are executable and reviewable |
| Simulator video | Completed | `assets/demo_delay_walk.gif` | Visual Isaac Lab execution only; not physical hardware |
| Policy packaging and validated inference tooling | Implemented; trained artifact manifest pending | `deployment/python/policy_artifact.py`, `deployment/python/policy_runtime.py` | Training-to-runtime conversion plus runtime shape/finite checks |
| Runtime joint mapping and safety guard | Implemented and unit-tested generically | `deployment/python/runtime_guard.py`, ROS 2 C++ package and tests | Deployment engineering, watchdog and command-safety design |
| Exact TienKung hardware joint mapping | Pending hardware-owner review | `deployment/ros2/tienkung_policy_bridge/config/tienkung_policy_bridge.yaml` remains locked with `hardware_mapping_verified=false` | No claim of a verified hardware mapping yet |
| MuJoCo Sim-to-Sim validation | Not evidenced in the current repository | Required artifacts listed below | Must not be claimed until logs/video/config are added |
| Physical TienKung run | Not completed / no hardware access | `deployment/evidence/hardware_validation/` contains the evidence contract only | Must not be described as physical deployment |
| Physical rosbag, joint/IMU telemetry and synchronized video | Not available | Evidence contract only | Must originate from a real robot session |
| External collaborator or supervisor confirmation | Pending | Attestation template only | Must be signed or sent by the named person |

## Required promotion rule

A row may move to **Completed** only when:

1. the underlying raw artifact exists;
2. its SHA-256 is recorded;
3. the command/configuration needed to interpret it is included;
4. the README states who produced it and on which platform; and
5. a reviewer can distinguish simulation, replay and hardware data.

Writing documentation or creating an empty evidence directory does not complete
the corresponding claim.
