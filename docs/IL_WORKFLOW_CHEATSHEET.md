# IL Workflow Cheat Sheet

Step-by-step commands for Mobile AI imitation learning on this fork: collect demos → verify → train → evaluate.

> **Paths are examples** from this workstation (`~/trossen_ai_isaac`, `~/IsaacLab`, `~/lerobot_trossen/...`). Replace with your locations. Always `cd` into your `trossen_ai_isaac` clone first.

**Production fact (canonical):** reporting dataset `mobile_ai_right_pick_place_20260714_v2` — [LeRobot Dataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3), collected with **VR**, **`--record_arm right`** only (~50 episodes / ~30.5k frames @ 60 FPS). Keyboard/gamepad recording is smoke tooling only. Design: [Recording](epic3/04-recording-lerobot.md) · [VR recording](epic4/05-vr-recording.md). Reporting eval: [ACT Evaluation Report](ACT_EVAL_REPORT_100K.md) (56.7%).

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

**One-time host setup:** [Workstation config — Part A](epic4/03-workstation-config.md#part-a--one-time-setup). **Full session walkthrough (Wi-Fi → Trust → SteamVR → hands → Start AR):** [Part B](epic4/03-workstation-config.md#part-b--per-session-startup). **Design:** [VR recording](epic4/05-vr-recording.md) · [docs index — Epic 4](README.md#epic-4--vr-integration).

**Session startup** (summary — see Part B for dual-operator detail + screenshots):

- [ ] Same Wi-Fi; ALVR on PC + Quest; **Trust** device
- [ ] **Launch SteamVR from ALVR**; both hands tracked; **Toggle Dashboard** off
- [ ] Launch teleop / collect script → Isaac **Output Plugin = OpenXR** → **Start AR**
- [ ] POV wrong? Remove headset briefly, put it back
- [ ] Workstation: **U** / **N** after warm-up ([Controls](#controls-quick-reference))

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

**Practice teleop (no dataset):**

```bash
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_vr.py \
  --task Isaac-Reach-MobileAI-IK-Abs-Play-v0
```

Keys: [Controls quick reference](#controls-quick-reference) · full detail [VR teleoperation](epic4/04-vr-teleoperation.md) / [VR recording](epic4/05-vr-recording.md).

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

Use `--teleop_device gamepad` for gamepad. Keys: [Controls quick reference](#controls-quick-reference) · full tables [Teleoperation](epic3/03-teleoperation.md).

Smoke (no human demos):

```bash
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/smoke/smoke_record_env.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 --enable_cameras
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/smoke/smoke_record_dataset.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 --enable_cameras --overwrite
```

---

## Controls quick reference

Canonical detail: [Teleoperation](epic3/03-teleoperation.md) · [VR teleoperation](epic4/04-vr-teleoperation.md) · [VR recording](epic4/05-vr-recording.md) · [Recording](epic3/04-recording-lerobot.md).

### VR teleop only (`teleop_dual_arm_vr.py`)

| Key / input | Action |
|-------------|--------|
| **N** | Engage teleop |
| **M** | Pause (re-anchors on resume) |
| **B** | Re-anchor |
| **J** | Reset environment |
| **TAB** | Switch active arm (single-arm mode; not with `--dual_arm`) |
| **Pinch** | Toggle gripper (headset) |

### VR recording (`record_dual_arm_vr.py` / `run_collect_dataset.sh`)

| Key / input | Action |
|-------------|--------|
| **U** | Engage teleop without recording |
| **I** | Pause teleop |
| **N** | Start episode / save-and-reset |
| **M** | Discard episode buffer |
| **B** | Re-anchor |
| **J** | Reset (discards in-progress episode) |
| **TAB** | Switch arm if single-arm and not locked by `--record_arm left/right` |
| **Pinch** | Toggle gripper (headset) |

### Keyboard teleop / recording (`teleop_dual_arm_switch.py` / `record_dual_arm.py`)

| Key | Action |
|-----|--------|
| Motion (**W/S A/D Q/E Z/X T/G C/V**), **L** | EE motion / clear deltas — [full table](epic3/03-teleoperation.md) |
| **TAB** / **K** / **J** | Switch arm / gripper / reset |
| **N** / **M** | Episode toggle / discard (**recording only**) |

### Gamepad (`--teleop_device gamepad`)

| Button | Action |
|--------|--------|
| Sticks / D-pad | EE motion — [full table](epic3/03-teleoperation.md) |
| **Y** / **A** / **B** | Switch arm / gripper / reset |
| **X** | Episode toggle (**recording only**) |
| Keyboard **M** | Discard episode (no gamepad discard binding) |

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

**ACT 100k (reporting model):** same ACT recipe with `--steps=100000` and a separate `output_dir` / job name — hyperparameters in [Training](epic3/05-training.md).

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

Results summary: **[ACT Evaluation Report](ACT_EVAL_REPORT_100K.md)**. Architecture: [Evaluation](epic3/06-evaluation.md).

Optional open-loop replay:

```bash
./scripts/imitation_learning/run_play_replay.sh \
  ~/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2 0
```

Pi0 sim eval: [Evaluation](epic3/06-evaluation.md) (deferred — Inductor / timeout).

---

## Quick map

| Stage | Script / doc |
|-------|----------------|
| VR collect | `run_collect_dataset.sh` / [VR recording](epic4/05-vr-recording.md) |
| KB collect (smoke) | `record_dual_arm.py` / this cheat sheet §2 |
| Keys (all devices) | [Controls quick reference](#controls-quick-reference) |
| Verify | `verify_dataset.py` / this cheat sheet §3 |
| Train ACT / Pi0 | wrappers above / [Training](epic3/05-training.md) |
| Eval ACT | `run_play_act.sh` / [Evaluation](epic3/06-evaluation.md) · [ACT report](ACT_EVAL_REPORT_100K.md) |
