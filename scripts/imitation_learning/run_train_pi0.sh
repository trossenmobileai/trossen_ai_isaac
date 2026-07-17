#!/usr/bin/env bash
# ============================================================================
# Fine-tune a Pi0 policy on the right-arm pick-place dataset (interactive).
#
# Prerequisites:
#   1. Dataset verified: ./scripts/imitation_learning/run_verify_pi0_dataset.sh
#   2. conda env: lerobot_train (LeRobot with Pi0 / [pi] extras)
#   3. Network on first run (downloads lerobot/pi0_base from Hugging Face)
#
# Uses `conda run --no-capture-output` so the LeRobot progress bar streams to
# the terminal. Do not background this unless you do not need live progress.
#
# Checkpoints are saved to:
#   ~/trossen_ai_isaac/outputs/train/pi0_mobile_ai_right_v2/checkpoints/
#
# To resume training from a checkpoint:
#   conda run --no-capture-output -n lerobot_train lerobot-train \
#     --config_path ~/trossen_ai_isaac/outputs/train/pi0_mobile_ai_right_v2/checkpoints/last/pretrained_model/train_config.json \
#     --resume=true
#
# After training, evaluate with:
#   ./scripts/imitation_learning/run_play_pi0.sh \
#     ~/trossen_ai_isaac/outputs/train/pi0_mobile_ai_right_v2/checkpoints/last/pretrained_model \
#     10 60
# ============================================================================

set -euo pipefail

REPO_ID="trossen-admin/mobile_ai_right_pick_place_20260714_v2"
ROOT="$HOME/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2"
OUTPUT_DIR="$HOME/trossen_ai_isaac/outputs/train/pi0_mobile_ai_right_v2"
STEPS=10000

echo "============================================================"
echo "  Training Pi0 policy ($STEPS steps)"
echo "  Dataset : $ROOT"
echo "  Output  : $OUTPUT_DIR"
echo "  Progress bar: live (conda --no-capture-output)"
echo "============================================================"
echo ""

# Source .bashrc to pick up HF_DATASETS_CACHE before launching conda
source "$HOME/.bashrc" 2>/dev/null || true
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/trossen_ai_isaac/.hf_datasets_cache}"
mkdir -p "$HF_DATASETS_CACHE"

conda run --no-capture-output -n lerobot_train \
  env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  lerobot-train \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$ROOT" \
    --dataset.video_backend=pyav \
    --policy.type=pi0 \
    --policy.pretrained_path=lerobot/pi0_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.train_expert_only=true \
    --policy.device=cuda \
    --batch_size=8 \
    --steps="$STEPS" \
    --save_freq=1000 \
    --log_freq=100 \
    --num_workers=4 \
    --save_checkpoint=true \
    --output_dir="$OUTPUT_DIR" \
    --job_name=pi0_mobile_ai_right_v2 \
    --wandb.enable=false \
    --policy.push_to_hub=false

echo ""
echo "Training complete. Checkpoints at:"
echo "  $OUTPUT_DIR/checkpoints/"
echo "Evaluate with:"
echo "  ./scripts/imitation_learning/run_play_pi0.sh \\"
echo "    $OUTPUT_DIR/checkpoints/last/pretrained_model \\"
echo "    10 60"
