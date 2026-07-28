# TienKung real-robot deployment

This directory connects the delay-robust policy in this repository to the **official TienKung ROS 2 deployment stack** rather than a mock robot API.

## Upstream interface

Pinned upstream: `Open-X-Humanoid/Deploy_Tienkung@1f2b8b8071398d65bc2b085570b2520449b633e9`

The official stack consists of:

- `rl_control_new`: ROS 2 control node and hardware topic bridge;
- `x_humanoid_rl_sdk`: STOP/ZERO/MLP finite-state machine, robot interface and OpenVINO inference.

The real node subscribes to `/leg/status`, `/arm/status`, `/imu/status` and `/sbus_data`. It publishes motor commands through `/leg/cmd_ctrl` and `/arm/cmd_ctrl`. The low-level loop runs at `dt=0.0025 s` (400 Hz), with the operator-controlled `STOP -> ZERO -> MLP` transition.

## Delay-compensated policy integration

The official `StateMLP` already constructs the same observation contract used by this project:

```text
one frame = 75 values
history   = 10 frames
input     = [1, 750]
actions   = [1, 20]
```

The 75-value frame contains angular velocity, projected gravity, velocity command, 20 joint-position errors, 20 joint velocities, previous 20 actions and six gait-phase values. The official controller shifts the ten-frame history before each policy update.

`ActorCriticDelayPredictor.act_inference()` predicts the newest 75-value frame, replaces the final history frame, and then executes the actor. Both operations are exported as **one OpenVINO graph**. The C++ controller therefore feeds raw delayed history into one model; it does not maintain a second predictor implementation.

`patch_official_delay_policy.py` patches the genuine upstream `FSMStateImpl.cpp` to:

- name the `75 x 10 -> 20` contract explicitly;
- reject OpenVINO models that are not `[1,750] -> [1,20]`;
- use the named frame-size constant in the history buffer;
- fail immediately if the runtime action tensor changes size.

## 1. Install and patch the official stack

```bash
bash deployment/tienkung_real/install_official_stack.sh ~/tklab_ws
```

Required platform: Ubuntu 22.04, ROS 2 Humble, C++17, OpenVINO, Eigen3, yaml-cpp and the robot-side `bodyctrl_msgs` package.

## 2. Export the trained combined policy

Export a TorchScript module whose forward path is the same as `ActorCriticDelayPredictor.act_inference()`:

```bash
python deployment/tienkung_real/export_openvino.py \
  --policy /path/to/exported/delay_compensated_policy.pt \
  --output delay_compensated_policy.xml
```

The exporter enforces `[1,750] -> [1,20]` and compares TorchScript with OpenVINO on both zero and non-zero histories. Do not continue if either numerical test fails.

Copy the generated `.xml` and `.bin` files into:

```text
~/tklab_ws/src/Deploy_Tienkung/rl_control_new/config/policy/
```

## 3. Install robot configuration

```bash
cp deployment/tienkung_real/tg22_delay_comp_config.yaml \
  ~/tklab_ws/src/Deploy_Tienkung/rl_control_new/config/tg22_config.yaml
```

The included file preserves the official 20-motor configuration, 400 Hz period, current scales and PD gains. Zero offsets, IMU roll offset and gains still require confirmation on the specific physical robot.

## 4. Build and launch

```bash
cd ~/tklab_ws
colcon build --packages-select x_humanoid_rl_sdk rl_control_new
source install/setup.bash
ros2 launch rl_control_new rl.launch.py
```

For a wired Xbox controller:

```bash
ros2 run joy joy_node --ros-args --remap joy:=sbus_data
```

Official mapping: `X` enters ZERO, `A` enters MLP, and `Y` enters STOP.

## Remaining physical checks

Before enabling MLP on hardware, compare the exported training configuration against the robot:

1. all 20 joint names and both reordering tables;
2. radians/radians-per-second units and encoder signs;
3. IMU Euler convention and projected-gravity direction;
4. command scales, default pose and action scale;
5. policy update ratio relative to the 400 Hz motor loop;
6. robot-specific zero offsets, PD gains and emergency-stop operation.

## Validation status

- official ROS 2 hardware interface: integrated and pinned;
- official `StateMLP` observation/history path: matched to the 750-dimensional policy input;
- embedded predictor + actor OpenVINO export: implemented;
- C++ tensor-contract guards: implemented;
- TG22 deployment configuration: provided;
- physical TienKung execution: requires robot access and operator validation.

No hardware result or success rate is claimed until the policy has physically run and logs have been collected.
