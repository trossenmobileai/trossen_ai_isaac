#!/usr/bin/env bash
# ============================================================================
# Phase 7 — Evaluate a trained ACT policy in simulation.
#
# Launches Isaac Sim with the joint-position lift environment, starts the ACT
# policy sidecar in the LeRobot venv, and runs closed-loop rollouts.
#
# Usage:
#   ./run_play_act.sh [CHECKPOINT_DIR] [NUM_EPISODES]
#
# Defaults:
#   CHECKPOINT_DIR  ~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/last/pretrained_model
#   NUM_EPISODES    10
#
# Results are written to:
#   ~/trossen_ai_isaac/outputs/eval/rollout_summary.json
# ============================================================================

set -euo pipefail

CHECKPOINT_DIR="${1:-$HOME/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/last/pretrained_model}"
NUM_EPISODES="${2:-10}"

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
SCRIPT="$REPO_DIR/scripts/imitation_learning/evaluation/play_act.py"
OUTPUT_DIR="$HOME/trossen_ai_isaac/outputs/eval"
SIDECAR_PYTHON="$HOME/lerobot_trossen/.venv/bin/python"

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "[FAIL] Checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  Phase 7: ACT policy rollout"
echo "  Checkpoint : $CHECKPOINT_DIR"
echo "  Episodes   : $NUM_EPISODES"
echo "  Output     : $OUTPUT_DIR/rollout_summary.json"
echo "============================================================"

python "$SCRIPT" \
  --policy.path "$CHECKPOINT_DIR" \
  --num_episodes "$NUM_EPISODES" \
  --fps 30 \
  --sidecar-python "$SIDECAR_PYTHON" \
  --output_dir "$OUTPUT_DIR" \
  --headless

echo ""
echo "Rollout complete.  Summary at:"
echo "  $OUTPUT_DIR/rollout_summary.json"
