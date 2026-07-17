#!/usr/bin/env bash
# ============================================================================
# Verify the LeRobot dataset used for Pi0 fine-tuning (Phase 5 equivalent).
#
# Run this BEFORE run_train_pi0.sh. Checks parquet layout, episode count, and
# readable MP4 frames for:
#   ~/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2
#
# Usage:
#   ./scripts/imitation_learning/run_verify_pi0_dataset.sh
# ============================================================================

set -euo pipefail

REPO_ID="trossen-admin/mobile_ai_right_pick_place_20260714_v2"
ROOT="$HOME/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2"
REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

echo "============================================================"
echo "  Verifying Pi0 dataset at $ROOT"
echo "============================================================"

~/lerobot_trossen/.venv/bin/python "$REPO_DIR/scripts/imitation_learning/validation/verify_dataset.py" \
  --root "$ROOT" \
  --repo_id "$REPO_ID" \
  --skip-lerobot

echo ""
echo "Dataset verify complete. Next: ./scripts/imitation_learning/run_train_pi0.sh"
