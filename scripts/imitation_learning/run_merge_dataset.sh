#!/usr/bin/env bash
# ============================================================================
# Merge per-session dataset shards into one unified LeRobot dataset.
#
# Each shard is produced by a single run of run_collect_dataset.sh. After
# collecting shards across multiple sessions, run this script once to combine
# them all into a single valid LeRobot v3 dataset.
#
# Usage:
#   ./scripts/imitation_learning/run_merge_dataset.sh          # uses defaults below
#   ./scripts/imitation_learning/run_merge_dataset.sh --verify # also runs verify_dataset
#
# Requirements:
#   - All shards must share the same --record_arm mode and --fps (the merge
#     step raises an error if they differ).
#   - Run after at least 2 shard sessions are complete.
#
# Output:
#   $ROOT_BASE/merged/   <- final merged dataset
# ============================================================================

set -euo pipefail

# ---- Configure the same base variables as run_collect_dataset.sh ----
REPO_BASE="trossen-admin/mobile_ai_right_pick_place_20260714_v2"
ROOT_BASE="$HOME/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2"
MERGED_REPO_ID="$REPO_BASE"
MERGED_ROOT="$ROOT_BASE/merged"
SHARDS_DIR="$ROOT_BASE/shards"

REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
VENV_PYTHON="$HOME/lerobot_trossen/.venv/bin/python"
MERGE_SCRIPT="$REPO_DIR/scripts/imitation_learning/recording/merge_datasets.py"
VERIFY_SCRIPT="$REPO_DIR/scripts/imitation_learning/validation/verify_dataset.py"

RUN_VERIFY=false
if [[ "${1:-}" == "--verify" ]]; then
    RUN_VERIFY=true
fi

echo "============================================================"
echo "  Mobile AI dataset merge"
echo "  Shards dir : $SHARDS_DIR"
echo "  Output     : $MERGED_ROOT"
echo "  repo_id    : $MERGED_REPO_ID"
echo "============================================================"
echo ""

if [[ ! -d "$SHARDS_DIR" ]]; then
    echo "ERROR: Shards directory not found: $SHARDS_DIR"
    echo "       Run run_collect_dataset.sh at least once first."
    exit 1
fi

SHARD_COUNT=$(find "$SHARDS_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
if [[ "$SHARD_COUNT" -lt 1 ]]; then
    echo "ERROR: No shard subdirectories found in $SHARDS_DIR"
    exit 1
fi

echo "Found $SHARD_COUNT shard(s)."
echo ""

OVERWRITE_FLAG=""
if [[ -d "$MERGED_ROOT" ]]; then
    echo "WARNING: Merged output already exists at $MERGED_ROOT"
    read -rp "  Overwrite? [y/N] " answer
    if [[ "${answer,,}" == "y" ]]; then
        OVERWRITE_FLAG="--overwrite"
    else
        echo "Aborted."
        exit 1
    fi
fi

"$VENV_PYTHON" "$MERGE_SCRIPT" \
    --shards_dir "$SHARDS_DIR" \
    --repo_id "$MERGED_REPO_ID" \
    --root "$MERGED_ROOT" \
    $OVERWRITE_FLAG

if $RUN_VERIFY; then
    echo ""
    echo "============================================================"
    echo "  Running dataset verification on merged output..."
    echo "============================================================"
    "$VENV_PYTHON" "$VERIFY_SCRIPT" \
        --root "$MERGED_ROOT" \
        --repo_id "$MERGED_REPO_ID"
fi

echo ""
echo "Merge complete. Dataset at: $MERGED_ROOT"
