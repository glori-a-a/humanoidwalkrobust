#!/usr/bin/env bash
set -euo pipefail

# Official deployment stack used by TienKung-Lab for real-robot validation.
# Pinning the commit prevents silent interface drift between training and deployment.
DEPLOY_REPO="https://github.com/Open-X-Humanoid/Deploy_Tienkung.git"
DEPLOY_COMMIT="1f2b8b8071398d65bc2b085570b2520449b633e9"
WORKSPACE="${1:-$HOME/tklab_ws}"
SRC_DIR="$WORKSPACE/src"
TARGET="$SRC_DIR/Deploy_Tienkung"

mkdir -p "$SRC_DIR"

if [[ ! -d "$TARGET/.git" ]]; then
  git clone "$DEPLOY_REPO" "$TARGET"
fi

git -C "$TARGET" fetch --all --tags
git -C "$TARGET" checkout --detach "$DEPLOY_COMMIT"

cat <<EOF
Pinned official deployment stack at:
  $TARGET
  commit: $DEPLOY_COMMIT

Expected ROS 2 packages:
  - rl_control_new
  - x_humanoid_rl_sdk

Next:
  1. Install ROS 2 Humble, OpenVINO, Eigen3, yaml-cpp and bodyctrl_msgs.
  2. Copy the exported OpenVINO model to:
       $TARGET/rl_control_new/config/policy/
  3. Copy deployment/tienkung_real/tg22_delay_comp_config.yaml to:
       $TARGET/rl_control_new/config/tg22_config.yaml
  4. Build from $WORKSPACE with colcon.
EOF
