# TienKung real-robot deployment

This directory connects the delay-robust policy in this repository to the **official TienKung ROS 2 deployment stack** rather than a mock robot API.

## Upstream interface

Pinned upstream: `Open-X-Humanoid/Deploy_Tienkung@1f2b8b8071398d65bc2b085570b2520449b633e9`

The official stack consists of:

- `rl_control_new`: ROS 2 control node and hardware topic bridge;
- `x_humanoid_rl_sdk`: STOP/ZERO/MLP finite-state machine, robot interface and OpenVINO inference.

The real node subscribes to:

- `/leg/status` (`bodyctrl_msgs/msg/MotorStatusMsg`)
- `/arm/status` (`bodyctrl_msgs/msg/MotorStatusMsg`)
- `/imu/status` (`bodyctrl_msgs/msg/Imu`)
- `/sbus_data` (`sensor_msgs/msg/Joy`)

It publishes:

- `/leg/cmd_ctrl` (`bodyctrl_msgs/msg/CmdMotorCtrl`)
- `/arm/cmd_ctrl` (`bodyctrl_msgs/msg/CmdMotorCtrl`)
- `/waist/cmd_pos` (`bodyctrl_msgs/msg/CmdSetMotorPosition`)

The control loop runs with `dt=0.0025 s` (400 Hz). The official FSM transitions through `STOP -> ZERO -> MLP`; the operator must retain a working emergency-stop path.

## 1. Install the pinned official stack

```bash
bash deployment/tienkung_real/install_official_stack.sh ~/tklab_ws
```

Required platform:

- Ubuntu 22.04
- ROS 2 Humble
- C++17
- OpenVINO
- Eigen3 and yaml-cpp
- the robot-side `bodyctrl_msgs` package

## 2. Export the trained policy

TienKung-Lab exports a TorchScript policy from `play.py`. Convert it to the OpenVINO IR consumed by the official deployment stack:

```bash
python deployment/tienkung_real/export_openvino.py \
  --policy /path/to/exported/policy.pt \
  --observation-size <TRAINING_OBSERVATION_DIM> \
  --output delay_compensated_policy.xml
```

The command creates `delay_compensated_policy.xml` and `delay_compensated_policy.bin`, then compares OpenVINO output with TorchScript output on the same zero observation. Do not proceed when this numerical smoke test fails.

Copy both files into:

```text
~/tklab_ws/src/Deploy_Tienkung/rl_control_new/config/policy/
```

## 3. Install the robot configuration

```bash
cp deployment/tienkung_real/tg22_delay_comp_config.yaml \
  ~/tklab_ws/src/Deploy_Tienkung/rl_control_new/config/tg22_config.yaml
```

The included configuration preserves the official TG22 motor count, action count, 400 Hz period, current scales and PD gains. **Zero offsets, IMU roll offset and gains must be checked on the specific physical robot by the platform operator.**

## 4. Build and launch

```bash
cd ~/tklab_ws
colcon build --packages-select x_humanoid_rl_sdk rl_control_new
source install/setup.bash
ros2 launch rl_control_new rl.launch.py
```

For a wired Xbox controller, the official stack uses:

```bash
ros2 run joy joy_node --ros-args --remap joy:=sbus_data
```

Official Xbox mapping:

- `X`: ZERO
- `A`: MLP policy control
- `Y`: STOP

## Policy contract that must be verified

A model file being loadable is not enough. Before physical activation, verify all of the following against the training configuration and the C++ MLP state implementation:

1. exact joint order for all 20 actions;
2. observation order and history layout;
3. radians and radians-per-second units;
4. projected-gravity / IMU Euler convention;
5. command scaling and clipping;
6. default joint pose and action scale;
7. policy update rate versus the 400 Hz motor loop;
8. whether the delay compensator needs recurrent/history state not represented by a single flat input;
9. OpenVINO input and output tensor names/shapes;
10. STOP behaviour, ZERO interpolation and emergency stop.

The delay-compensated policy cannot be safely substituted for the official policy until its observation builder is matched exactly. That adaptation belongs in the official `x_humanoid_rl_sdk` MLP state, not in a fabricated Python hardware wrapper.

## Validation status

- Official ROS 2 hardware interface: integrated by pinned upstream dependency
- OpenVINO conversion and numerical smoke test: implemented here
- TG22 deployment configuration: provided here
- Physical TienKung execution: requires robot access and operator validation

No physical-run result, success rate or hardware video should be claimed until the policy has actually been enabled on the robot and logs have been collected.
