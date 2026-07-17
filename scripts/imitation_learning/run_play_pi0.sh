#!/usr/bin/env bash
# ============================================================================
# Phase 7 — Evaluate a trained Pi0 policy in simulation.
#
# Uses the shared Isaac eval path (play_act.py → act_rollout.py → metrics.py).
# Same contract as run_play_act.sh; only default checkpoint and output dir differ.
# Sidecar loads policy type from the checkpoint (pi0) via LeRobot config.
#
# Usage:
#   ./run_play_pi0.sh [CHECKPOINT_DIR] [NUM_EPISODES] [FPS] [--visual]
#
# Defaults:
#   CHECKPOINT_DIR  ~/trossen_ai_isaac/outputs/train/pi0_mobile_ai_right_v2/checkpoints/last/pretrained_model
#   NUM_EPISODES    10
#   FPS             60  (match training/recording rate)
#
# Pass --visual anywhere to open the Isaac Sim GUI instead of headless mode.
#
# Results are written to:
#   ~/trossen_ai_isaac/outputs/eval/pi0/rollout_summary.json
#
# EVAL CONTRACT (see lift/mdp/metrics.py):
#   - Success: clear lift (z > LIFT_CLEAR_Z), then released on table (cube_is_placed)
#   - Early stop: IDLE_STEPS=200 → no_progress; MAX_APPROACH_STEPS=1000 → no_pick;
#     MAX_STEPS_AFTER_LIFT=500 → no_place; POST_SUCCESS_STEPS=60 → success
#   - Play env timeout 90 s; warm-up hold before policy (not in episode metrics)
#   - Summary includes success_rate_by_color
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

CHECKPOINT_DIR="${1:-$HOME/trossen_ai_isaac/outputs/train/pi0_mobile_ai_right_v2/checkpoints/last/pretrained_model}"
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
OUTPUT_DIR="$HOME/trossen_ai_isaac/outputs/eval/pi0"
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
echo "  Phase 7: Pi0 policy rollout"
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
