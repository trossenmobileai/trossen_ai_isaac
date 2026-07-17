# IL Workflow Cheat Sheet

Step-by-step commands for Mobile AI imitation learning on this fork: collect demos → verify → train → evaluate.

> **Paths are examples** from this workstation (`~/trossen_ai_isaac`, `~/IsaacLab`, `~/lerobot_trossen/...`). Replace with your locations. Always `cd` into your `trossen_ai_isaac` clone first.

**Production fact:** the reporting dataset `mobile_ai_right_pick_place_20260714_v2` is a [LeRobot Dataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) collected with **VR**, **`--record_arm right`** only. Keyboard/gamepad recording scripts exist for smoke tests but were not used for production demos. Details: [Epic 3 §4.7](EPIC3_SIMULATION_TRAINING_PIPELINE.md#47-imitation-learning-recording-pipeline), [Epic 4 §5.4](EPIC4_VR_INTEGRATION.md#54-vr-recording-procedure).

---

## 0. Prerequisites

```bash
cd ~/trossen_ai_isaac
~/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
# Confirm: Isaac-Reach-MobileAI-IK-Abs-Play-v0, Isaac-Reach-MobileAI-Record-Play-v0,
#          Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0
```

---

## 1. Collect demos — VR (production)

**Stack:** ALVR launched → SteamVR from ALVR → Quest 3 connected → in Isaac Sim set Output Plugin = OpenXR → Start AR. Full setup: [Epic 4](EPIC4_VR_INTEGRATION.md).

**Why right arm only:** pick-and-place needs one arm; focusing on the active hand often makes VR lose tracking of the other arm (it drifts).

```bash
cd ~/trossen_ai_isaac
./scripts/imitation_learning/run_collect_dataset.sh          # optional session label as $1
# after all shards:
./scripts/imitation_learning/run_merge_dataset.sh --verify
```

Or one shot:

```bash
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm_vr.py \
  --repo_id YOUR_USERNAME/dataset_name \
  --root ~/lerobot_trossen/datasets/dataset_name \
  --fps 60 \
  --record_arm right
```

**Workstation keys:** **U** engage · **N** start/save episode · **M** discard · **B** re-anchor · **J** reset.

---

## 2. Collect demos — keyboard / gamepad (alternate)

```bash
cd ~/trossen_ai_isaac
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 \
  --repo_id YOUR_USERNAME/dataset_name \
  --root ~/lerobot_trossen/datasets/dataset_name \
  --fps 60 \
  --enable_cameras \
  --record_arm right
```

Use `--teleop_device gamepad` for gamepad. **Keys:** **N** toggle episode · **M** discard · **J** reset · **TAB** switch arm (when recording `both`).

Smoke (no human demos):

```bash
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/smoke/smoke_record_env.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 --enable_cameras
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/smoke/smoke_record_dataset.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 --enable_cameras --overwrite
```

---

## 3. Verify dataset

```bash
cd ~/trossen_ai_isaac
~/lerobot_trossen/.venv/bin/python scripts/imitation_learning/validation/verify_dataset.py \
  --root ~/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2 \
  --repo_id trossen-admin/mobile_ai_right_pick_place_20260714_v2
```

---

## 4. Train

**ACT 10k (wrapper):**

```bash
cd ~/trossen_ai_isaac
./scripts/imitation_learning/run_verify_and_train.sh
# → outputs/train/act_mobile_ai_right_v2/
```

**ACT 100k (reporting model):** see [Epic 3 §5.10](EPIC3_SIMULATION_TRAINING_PIPELINE.md#510-act-training-smoke-test-and-full-training) (`--steps=100000`, separate `output_dir`).

**Pi0 10k:**

```bash
./scripts/imitation_learning/run_verify_pi0_dataset.sh
./scripts/imitation_learning/run_train_pi0.sh
# → outputs/train/pi0_mobile_ai_right_v2/
```

Short connectivity smoke (not production train):

```bash
python scripts/imitation_learning/training/smoke_train_act.py \
  --root ~/lerobot_trossen/datasets/DATASET \
  --repo_id USER/DATASET
```

---

## 5. Evaluate (closed-loop)

**ACT reporting eval (30 episodes @ 60 FPS):**

```bash
cd ~/trossen_ai_isaac
./scripts/imitation_learning/run_play_act.sh \
  ~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2_100k/checkpoints/last/pretrained_model \
  30 60
```

Results summary: **[ACT Evaluation Report](ACT_EVAL_REPORT_100K.md)** (56.7% on the 100k run). Architecture: [Epic 3 §4.10](EPIC3_SIMULATION_TRAINING_PIPELINE.md#410-sim-act--pi0-evaluation).

Optional open-loop replay:

```bash
./scripts/imitation_learning/run_play_replay.sh \
  ~/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2 0
```

Pi0 sim eval: [Epic 3 §5.13](EPIC3_SIMULATION_TRAINING_PIPELINE.md#513-evaluate-pi0-policy-closed-loop-rollout) (deferred).

---

## Quick map

| Stage | Script / doc |
|-------|----------------|
| VR collect | `run_collect_dataset.sh` / Epic 4 §5.4 |
| KB collect | `record_dual_arm.py` / Epic 3 §5.8 |
| Verify | `verify_dataset.py` / Epic 3 §5.9 |
| Train ACT | `run_verify_and_train.sh` / Epic 3 §5.10 |
| Train Pi0 | `run_train_pi0.sh` / Epic 3 §5.12 |
| Eval ACT | `run_play_act.sh` / [ACT Evaluation Report](ACT_EVAL_REPORT_100K.md) |
