#!/usr/bin/env bash
# ============================================================================
# Open-loop replay of a single recorded episode in simulation.
#
# Useful for sanity-checking a collected dataset before training: the episode
# actions are replayed in the joint-position lift environment so you can see
# whether the motion looks correct and the cube is actually picked up.
#
# Usage:
#   ./run_play_replay.sh [DATASET_ROOT] [EPISODE_INDEX] [FPS] [--visual]
#
# Defaults:
#   DATASET_ROOT   ~/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2
#   EPISODE_INDEX  0
#   FPS            60
# ============================================================================

set -euo pipefail

DATASET_ROOT="${1:-$HOME/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2}"
EPISODE="${2:-0}"
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

if [ ! -d "$DATASET_ROOT" ]; then
  echo "[FAIL] Dataset root not found: $DATASET_ROOT" >&2
  exit 1
fi

echo "============================================================"
echo "  Replay episode $EPISODE from $DATASET_ROOT (fps=$FPS)"
echo "============================================================"

cd "$REPO_DIR"

if [ "${TERM:-}" = "dumb" ] || [ -z "${TERM:-}" ]; then
  export TERM=xterm
fi

REPLAY_ARGS=(
  --root "$DATASET_ROOT"
  --episode "$EPISODE"
  --fps "$FPS"
)
if [ -n "$HEADLESS" ]; then
  REPLAY_ARGS+=("$HEADLESS")
fi

~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/evaluation/play_replay.py "${REPLAY_ARGS[@]}"
