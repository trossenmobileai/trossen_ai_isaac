# IL Pipeline Branch Status

Integration branch: **`feat/il-pipeline-integration`** (Phases 1–2 recording + Phase 4 scene).

## Repository layout (scripts vs package)

| Path | Role |
|------|------|
| `scripts/teleoperation/` | Runnable teleop CLIs (`isaaclab.sh -p`) |
| `scripts/imitation_learning/` | Recording, dataset QA, training smoke |
| `scripts/demos/` | Standalone Isaac Sim demos (`isaacsim/python.sh`) |
| `scripts/lib/` | Shared helpers for demos (`controller.py`, `leader_arm.py`) |
| `source/trossen_ai_isaac/trossen_ai_isaac/teleop/` | Teleop **library** (loops, session, VR) |
| `source/trossen_ai_isaac/trossen_ai_isaac/recording/` | LeRobot writer, frame capture, smokes |
| `source/trossen_ai_isaac/trossen_ai_isaac/validation/` | Offline dataset checks |
| `source/trossen_ai_isaac/trossen_ai_isaac/training/` | Training smoke helpers |

## Active branches

| Branch | Phase | Status |
|--------|-------|--------|
| `feat/il-record-env` | 1 — IL recording environment | **Merged into phase2**; archive |
| `feat/il-record-phase2` | 2 — LeRobot dataset writer | **Complete**; baseline for integration |
| `feat/il-pipeline-integration` | 1+2+4 scene | **Integration target** → PR to `main` |
| `feat/sim-environment` | 4 — table + cube + randomization | **Merged into integration** |
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
| Training | None (despite branch name) | N/A — training is Phase 4 in `lerobot_trossen` |

## Not started

| Phase | Work | Suggested branch |
|-------|------|------------------|
| 3 | VR teleop + LeRobot recording (`teleop_dual_arm_vr.py`) | `feat/il-record-vr` off integration |
| 4 training | `lerobot-train` ACT on sim datasets | `lerobot_train` conda env — smoke: `smoke_train_act.py` |

## Recording quick reference

```bash
# Smoke test (Isaac Sim required)
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/smoke/smoke_record_env.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 --enable_cameras

# Automated one-episode dataset smoke
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/smoke/smoke_record_dataset.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 --enable_cameras --overwrite

# Record demonstrations
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 \
  --repo_id USER/dataset_name --root ~/lerobot_trossen/datasets/dataset_name \
  --fps 60 --enable_cameras

# Verify dataset (LeRobot venv, not system python3)
~/lerobot_trossen/.venv/bin/python scripts/imitation_learning/validation/verify_dataset.py \
  --root ~/lerobot_trossen/datasets/dataset_name --repo_id USER/dataset_name

# Training smoke (train: lerobot_train conda; verify uses installed package)
python scripts/imitation_learning/training/smoke_train_act.py \
  --root ~/lerobot_trossen/datasets/dataset_name --repo_id USER/dataset_name
```
