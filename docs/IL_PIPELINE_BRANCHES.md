# IL Pipeline Branch Status

**Current branch:** `main` — IL recording pipeline, scene integration, and VR recording are merged (PR #3, PR #2).

**Epic documentation:** [Mobile AI docs index](README.md) · [Epic 3 — Simulation Training Pipeline](EPIC3_SIMULATION_TRAINING_PIPELINE.md) · [Epic 4 — VR Integration](EPIC4_VR_INTEGRATION.md)

## Repository layout (scripts vs package)

| Path | Role |
|------|------|
| `scripts/teleoperation/` | Runnable teleop CLIs (`isaaclab.sh -p`) |
| `scripts/imitation_learning/` | Recording, validation, training, **ACT evaluation** | `isaaclab.sh -p` or plain Python |
| `scripts/demos/` | Standalone Isaac Sim demos (`isaacsim/python.sh`) |
| `scripts/lib/` | Shared helpers for demos (`controller.py`, `leader_arm.py`) |
| `source/trossen_ai_isaac/trossen_ai_isaac/teleop/` | Teleop **library** (loops, session, VR) |
| `source/trossen_ai_isaac/trossen_ai_isaac/recording/` | LeRobot writer, frame capture, smokes |
| `source/trossen_ai_isaac/trossen_ai_isaac/validation/` | Offline dataset checks |
| `source/trossen_ai_isaac/trossen_ai_isaac/training/` | Training smoke helpers |
| `source/trossen_ai_isaac/trossen_ai_isaac/evaluation/` | ACT rollout, policy sidecar |

## Branch history

| Branch | Phase | Status |
|--------|-------|--------|
| `feat/il-record-env` | 1 — IL recording environment | **Merged** (into phase 2); archive |
| `feat/il-record-phase2` | 2 — LeRobot dataset writer | **Merged** (PR #3); archive |
| `feat/sim-environment` | 4 — table + cube + randomization | **Merged** (PR #3); archive |
| `feat/il-pipeline-integration` | 1+2+4 scene integration | **Merged to `main`** (PR #3) |
| `feat/vr-handtracking-teleop` | VR teleoperation | **Merged**; archive |
| `feat/vr-recording-integration` | VR + LeRobot recording | **Merged to `main`** (PR #2) |
| `feat/sim-training` | *(misnamed)* | **Deprecated — do not merge** |

## Deprecated: `feat/sim-training`

This branch is **superseded by `feat/il-record-phase2`**. Do not merge it.

| Issue | `feat/sim-training` | `feat/il-record-phase2` |
|-------|---------------------|-------------------------|
| Output format | Robomimic **HDF5** | **LeRobot** parquet + MP4 |
| Env completeness | `record_env_cfg.py` references missing `record_joint_pos_14`; no gym registration | Full env + `Isaac-Reach-MobileAI-Record-Play-v0` |
| Camera prims | `*_optical_frame` (broken) | USD `Camera_*` prims (validated) |
| Architecture | 576-line monolithic `teleop_dual_arm_switch.py` | Modular `teleop/` + `recording/` packages |
| Recording bug | **Double `env.step()`** per frame | Single step + `recorder.on_step()` |
| Training | None (despite branch name) | N/A — training smoke in `lerobot_train` conda env |

## Remaining work

| Area | Work | Notes |
|------|------|-------|
| Full ACT training in-repo wrapper | Production `lerobot-train` helper script | [`run_verify_and_train.sh`](../scripts/imitation_learning/run_verify_and_train.sh) exists; smoke in-repo |
| Sim-to-real | Deploy trained policies on physical Mobile AI | Parallel track; see [Trossen real-robot eval docs](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot_plugin/train_and_evaluate.html) |
| Large-scale VR collection | Headset-on-workstation validation and bulk VR demos | Entrypoint exists; hardware validation pending |
| Eval dataset recording in sim | Save `eval_*` LeRobot datasets during rollout | Metrics-only eval implemented; optional future work |

## Evaluation quick reference

Sim equivalent of real-robot `lerobot-record --policy.path=<checkpoint>`:

```bash
# Preflight: verify checkpoint loads
conda run -n lerobot_train python -c "
from lerobot.policies.act.modeling_act import ACTPolicy
ACTPolicy.from_pretrained('~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/last/pretrained_model')
print('OK')
"

# Open-loop replay (sanity check before policy rollout)
./scripts/imitation_learning/run_play_replay.sh \
  ~/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2 0

# Closed-loop ACT rollout (writes ~/trossen_ai_isaac/outputs/eval/rollout_summary.json)
# Per-episode: success, lifted, returned, on_table, stop_reason (success|no_pick|no_place|env_done), steps
./scripts/imitation_learning/run_play_act.sh

# Visual mode: add --visual anywhere
./scripts/imitation_learning/run_play_act.sh \
  ~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/last/pretrained_model \
  1 --visual
```

See [Epic 3 §4.8](EPIC3_SIMULATION_TRAINING_PIPELINE.md#48-sim-act-evaluation) for architecture and troubleshooting.

## Recording quick reference

```bash
# Smoke test (Isaac Sim required)
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/smoke/smoke_record_env.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 --enable_cameras

# Automated one-episode dataset smoke
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/smoke/smoke_record_dataset.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 --enable_cameras --overwrite

# Record demonstrations (keyboard / gamepad)
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 \
  --repo_id USER/dataset_name --root ~/lerobot_trossen/datasets/dataset_name \
  --fps 60 --enable_cameras

# Record demonstrations (VR hand tracking; requires headset + ALVR/SteamVR)
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm_vr.py \
  --repo_id USER/dataset_name --root ~/lerobot_trossen/datasets/dataset_name \
  --fps 60

# Verify dataset (LeRobot venv, not system python3)
~/lerobot_trossen/.venv/bin/python scripts/imitation_learning/validation/verify_dataset.py \
  --root ~/lerobot_trossen/datasets/dataset_name --repo_id USER/dataset_name

# Training smoke (train: lerobot_train conda; verify uses installed package)
python scripts/imitation_learning/training/smoke_train_act.py \
  --root ~/lerobot_trossen/datasets/dataset_name --repo_id USER/dataset_name
```
