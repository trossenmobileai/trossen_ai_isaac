#!/usr/bin/env bash
# ============================================================================
# Phase 5 + 6 — Verify dataset and train ACT policy.
#
# Run this script AFTER the recording session has finished (run_collect_50.sh
# has completed and printed "Finalized dataset").
#
# The script will:
#   1. Verify the dataset (parquet, episode count, readable MP4 frames)
#   2. Train an ACT policy for 10 000 steps on the GPU
#
# Checkpoints are saved to:
#   ~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/
#
# To resume training from a checkpoint:
#   conda run -n lerobot_train lerobot-train \
#     --config_path ~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/last/pretrained_model/train_config.json \
#     --resume=true
# ============================================================================

set -euo pipefail

REPO_ID="trossen-admin/mobile_ai_right_pick_place_20260714_v2"
ROOT="$HOME/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2"
OUTPUT_DIR="$HOME/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2"
REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
STEPS=10000

# ---------------------------------------------------------------------------
# Phase 5 — Verify dataset
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Phase 5: Verifying dataset at $ROOT"
echo "============================================================"

~/lerobot_trossen/.venv/bin/python "$REPO_DIR/scripts/imitation_learning/validation/verify_dataset.py" \
  --root "$ROOT" \
  --repo_id "$REPO_ID" \
  --skip-lerobot

echo ""

# ---------------------------------------------------------------------------
# Phase 6 — Train ACT policy
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Phase 6: Training ACT policy ($STEPS steps)"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"
echo ""

# Source .bashrc to pick up HF_DATASETS_CACHE before launching conda
source "$HOME/.bashrc" 2>/dev/null || true
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/trossen_ai_isaac/.hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"

conda run -n lerobot_train \
  env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  lerobot-train \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$ROOT" \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --output_dir="$OUTPUT_DIR" \
    --job_name=act_mobile_ai_right_v2 \
    --policy.device=cuda \
    --steps="$STEPS" \
    --save_freq=1000 \
    --log_freq=100 \
    --num_workers=4 \
    --save_checkpoint=true \
    --wandb.enable=false \
    --policy.push_to_hub=false

echo ""
echo "Training complete. Checkpoints at:"
echo "  $OUTPUT_DIR/checkpoints/"
