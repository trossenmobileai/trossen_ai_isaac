# IL Pipeline Branch Status

Integration branch: **`feat/il-pipeline-integration`** (Phases 1–2 recording + Phase 4 scene).

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
| 4 training | `lerobot-train` ACT on sim datasets | `lerobot_trossen` repo |

## Recording quick reference

```bash
# Smoke test (Isaac Sim required)
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/smoke_record_env.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 --enable_cameras

# Record demonstrations
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/record_dual_arm.py \
  --task Isaac-Reach-MobileAI-Record-Play-v0 \
  --repo_id USER/dataset_name --root ~/lerobot_trossen/datasets/dataset_name \
  --fps 60 --enable_cameras

# Verify dataset (LeRobot venv, not system python3)
~/lerobot_trossen/.venv/bin/python scripts/teleoperation/verify_recorded_dataset.py \
  --root ~/lerobot_trossen/datasets/dataset_name --repo_id USER/dataset_name
```
