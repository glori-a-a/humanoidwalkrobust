#!/usr/bin/env bash
set -euo pipefail

# Downloads the pretrained baseline policies published by the official
# Open-X-Humanoid/TienKung-Lab repository. These are baseline walk/run policies,
# NOT the delay-compensated policy developed in this repository.
UPSTREAM_COMMIT="main"
BASE_URL="https://raw.githubusercontent.com/Open-X-Humanoid/TienKung-Lab/${UPSTREAM_COMMIT}/Exported_policy"
OUT_DIR="${1:-checkpoints/official_tienkung}"

mkdir -p "$OUT_DIR"

curl --fail --location --retry 3 \
  "$BASE_URL/walk.pt" \
  --output "$OUT_DIR/walk_official_pretrained.pt"

curl --fail --location --retry 3 \
  "$BASE_URL/run.pt" \
  --output "$OUT_DIR/run_official_pretrained.pt"

cat > "$OUT_DIR/SOURCE.md" <<EOF
# Checkpoint provenance

- Source repository: Open-X-Humanoid/TienKung-Lab
- Source directory: Exported_policy/
- Revision requested: ${UPSTREAM_COMMIT}
- Files: walk.pt, run.pt
- Purpose: official pretrained baseline policies for play/sim2sim
- These files are not delay-compensated checkpoints from this dissertation.
EOF

printf 'Downloaded official baseline checkpoints to %s\n' "$OUT_DIR"
printf 'Do not rename or describe them as delay-compensated policies.\n'
