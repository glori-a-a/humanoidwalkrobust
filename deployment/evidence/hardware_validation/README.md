# Physical-hardware evidence contract

This directory is intentionally documentation-only until a real TienKung test
session occurs. Do not add generated or simulator data under a hardware label.

A hardware session is reviewable only when it contains:

```text
hardware_validation/<YYYY-MM-DD_session-id>/
├── metadata.yaml
├── run_notes.md
├── sha256.txt
├── video/
│   └── synchronized_run.mp4
├── telemetry/
│   ├── joint_state.csv
│   ├── imu.csv
│   ├── policy_action.csv
│   └── safe_target.csv
└── rosbag2/
    ├── metadata.yaml
    └── <bag database files>
```

## Required metadata

- robot model, revision and serial identifier with sensitive digits redacted;
- controller computer and inference runtime;
- repository commit and policy artifact SHA-256;
- exact joint-map/configuration hash;
- policy and low-level control frequencies;
- time source and synchronization method;
- on-site operator and remote software owner;
- fall protection, speed/torque limits and emergency-stop procedure;
- number of trials, success criterion and excluded trials;
- whether the run was autonomous, teleoperated or replayed.

## Minimum telemetry columns

Every row must contain a monotonic or synchronized timestamp. At minimum:

- joint position and velocity in the verified policy order;
- IMU orientation/angular velocity;
- command velocity;
- normalized policy action;
- safety-limited target;
- controller mode and fault code;
- inference latency and end-to-end command age where available.

## Integrity

Generate `sha256.txt` from the original files before editing or transcoding the
video. Large rosbag/video files may be stored in a release or external research
archive, but this repository must retain their hashes and stable links.

## Claim boundary

The existence of this checklist is not evidence of a physical run. Update
`deployment/EVIDENCE_STATUS.md` only after the artifacts above exist and have
been reviewed.

