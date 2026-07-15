#!/usr/bin/env bash
# ============================================================================
# Phase 7 — Evaluate a trained ACT policy in simulation.
#
# Launches Isaac Sim with the joint-position lift environment, starts the ACT
# policy sidecar in the lerobot_train conda env (same env used for training),
# and runs closed-loop rollouts.
#
# Usage:
#   ./run_play_act.sh [CHECKPOINT_DIR] [NUM_EPISODES] [FPS] [--visual]
#
# Defaults:
#   CHECKPOINT_DIR  ~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/last/pretrained_model
#   NUM_EPISODES    10
#   FPS             60  (match training/recording rate)
#
# Pass --visual anywhere to open the Isaac Sim GUI instead of headless mode.
#
# Results are written to:
#   ~/trossen_ai_isaac/outputs/eval/rollout_summary.json
#
# Episode control (see metrics.py):
#   - Success: cube clears on-table band (z > 0.845 m), then released on table (cube_is_placed)
#   - Early stop: +60 steps after success, or ~400 steps if no pick / ~400 steps after lift if no place
#   - Terminal line includes stop_reason: success | no_pick | no_place | env_done
# ============================================================================

set -euo pipefail

_resolve_sidecar_python() {
  if command -v conda >/dev/null 2>&1; then
    local py
    py="$(conda run -n lerobot_train which python 2>/dev/null || true)"
    if [ -n "$py" ] && [ -x "$py" ]; then
      echo "$py"
      return
    fi
  fi
  echo "$HOME/lerobot_trossen/.venv/bin/python"
}

CHECKPOINT_DIR="${1:-$HOME/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/last/pretrained_model}"
NUM_EPISODES="${2:-10}"
shift 2 2>/dev/null || true

FPS=60
HEADLESS="--headless"
for arg in "$@"; do
  case "$arg" in
    --visual) HEADLESS="" ;;
    [0-9]*) FPS="$arg" ;;
    *)
      echo "[FAIL] Unknown argument: $arg (expected FPS number or --visual)" >&2
      exit 1
      ;;
  esac
done

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
OUTPUT_DIR="$HOME/trossen_ai_isaac/outputs/eval"
SIDECAR_PYTHON="$(_resolve_sidecar_python)"

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "[FAIL] Checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

if [ ! -x "$SIDECAR_PYTHON" ]; then
  echo "[FAIL] Sidecar Python not found or not executable: $SIDECAR_PYTHON" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  Phase 7: ACT policy rollout"
echo "  Checkpoint : $CHECKPOINT_DIR"
echo "  Episodes   : $NUM_EPISODES"
echo "  FPS        : $FPS"
echo "  Mode       : $(if [ -n "$HEADLESS" ]; then echo headless; else echo visual; fi)"
echo "  Sidecar py : $SIDECAR_PYTHON"
echo "  Output     : $OUTPUT_DIR/rollout_summary.json"
echo "============================================================"

cd "$REPO_DIR"

# isaaclab.sh calls `tabs`; TERM=dumb (common in CI/automation) makes that fail.
if [ "${TERM:-}" = "dumb" ] || [ -z "${TERM:-}" ]; then
  export TERM=xterm
fi

PLAY_ARGS=(
  --policy.path "$CHECKPOINT_DIR"
  --num_episodes "$NUM_EPISODES"
  --fps "$FPS"
  --sidecar-python "$SIDECAR_PYTHON"
  --output_dir "$OUTPUT_DIR"
)
if [ -n "$HEADLESS" ]; then
  PLAY_ARGS+=("$HEADLESS")
fi

if ! ~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/evaluation/play_act.py "${PLAY_ARGS[@]}"; then
  echo "[FAIL] Rollout failed." >&2
  exit 1
fi

echo ""
echo "Rollout complete.  Summary at:"
echo "  $OUTPUT_DIR/rollout_summary.json"
