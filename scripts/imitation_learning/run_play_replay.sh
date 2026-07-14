#!/usr/bin/env bash
# ============================================================================
# Open-loop replay of a single recorded episode in simulation.
#
# Useful for sanity-checking a collected dataset before training: the episode
# actions are replayed in the joint-position lift environment so you can see
# whether the motion looks correct and the cube is actually picked up.
#
# Usage:
#   ./run_play_replay.sh [DATASET_ROOT] [EPISODE_INDEX]
#
# Defaults:
#   DATASET_ROOT   ~/lerobot_trossen/datasets/mobile_ai_right_pick_place_session1
#   EPISODE_INDEX  0
# ============================================================================

set -euo pipefail

DATASET_ROOT="${1:-$HOME/lerobot_trossen/datasets/mobile_ai_right_pick_place_session1}"
EPISODE="${2:-0}"

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
SCRIPT="$REPO_DIR/scripts/imitation_learning/evaluation/play_replay.py"

if [ ! -d "$DATASET_ROOT" ]; then
  echo "[FAIL] Dataset root not found: $DATASET_ROOT" >&2
  exit 1
fi

echo "============================================================"
echo "  Replay episode $EPISODE from $DATASET_ROOT"
echo "============================================================"

~/lerobot_trossen/.venv/bin/python "$SCRIPT" \
  --root "$DATASET_ROOT" \
  --episode "$EPISODE" \
  --fps 30 \
  --headless
