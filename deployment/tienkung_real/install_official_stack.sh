#!/usr/bin/env bash
set -euo pipefail

# Official deployment stack used by TienKung-Lab for real-robot validation.
# Pinning the commit prevents silent interface drift between training and deployment.
DEPLOY_REPO="https://github.com/Open-X-Humanoid/Deploy_Tienkung.git"
DEPLOY_COMMIT="1f2b8b8071398d65bc2b085570b2520449b633e9"
WORKSPACE="${1:-$HOME/tklab_ws}"
SRC_DIR="$WORKSPACE/src"
TARGET="$SRC_DIR/Deploy_Tienkung"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SRC_DIR"

if [[ ! -d "$TARGET/.git" ]]; then
  git clone "$DEPLOY_REPO" "$TARGET"
fi

git -C "$TARGET" fetch --all --tags
git -C "$TARGET" checkout --detach "$DEPLOY_COMMIT"

# Apply the repository-specific policy contract to the genuine upstream StateMLP.
python3 "$SCRIPT_DIR/patch_official_delay_policy.py" --deploy-root "$TARGET" --check
python3 "$SCRIPT_DIR/patch_official_delay_policy.py" --deploy-root "$TARGET"

cat <<EOF
Pinned and patched official deployment stack at:
  $TARGET
  commit: $DEPLOY_COMMIT

Policy contract:
  raw hardware history: [1, 750] = 10 frames x 75 values
  exported graph: embedded observation predictor + actor
  motor action output: [1, 20]

Expected ROS 2 packages:
  - rl_control_new
  - x_humanoid_rl_sdk

Next:
  1. Install ROS 2 Humble, OpenVINO, Eigen3, yaml-cpp and bodyctrl_msgs.
  2. Export the combined delay-compensated policy to OpenVINO.
  3. Copy the .xml/.bin model to:
       $TARGET/rl_control_new/config/policy/
  4. Copy deployment/tienkung_real/tg22_delay_comp_config.yaml to:
       $TARGET/rl_control_new/config/tg22_config.yaml
  5. Build from $WORKSPACE with colcon.
EOF
