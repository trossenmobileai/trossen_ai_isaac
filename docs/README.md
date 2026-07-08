# Trossen Mobile AI — Simulation & VR Documentation

Documentation for the **Trossen Mobile AI** imitation-learning project, covering two development epics:

| Epic | Document | What it covers |
|------|----------|----------------|
| **Epic 3** | [Simulation Training Pipeline](EPIC3_SIMULATION_TRAINING_PIPELINE.md) | Isaac Lab tasks, dual-arm teleoperation, LeRobot data collection (keyboard/gamepad and VR), training smoke |
| **Epic 4** | [VR Integration](EPIC4_VR_INTEGRATION.md) | Quest 3 + ALVR + OpenXR teleoperation and VR dataset recording on the same task framework |

## Who this is for

New team members, stakeholders, and contributors who need to understand what was built, why, and how to run it — without prior Isaac Sim experience.

Both epic reports follow the same technical-report structure (Goal, Overview, Background, Implementation, Operational Procedures, Findings and Limitations, Troubleshooting, Current Status). Content is updated as the project progresses.

## Reading order

1. **This page** — orientation (2 minutes)
2. **[Epic 3](EPIC3_SIMULATION_TRAINING_PIPELINE.md)** — read first; the full simulation and training story
3. **[Epic 4](EPIC4_VR_INTEGRATION.md)** — read second; VR builds on Epic 3's teleoperation layer

## Related docs in this repo

| Doc | Purpose |
|-----|---------|
| [IL Pipeline Branches](IL_PIPELINE_BRANCHES.md) | Branch status, folder glossary, recording quick reference |
| [Repository README](../README.md) | Install, scripts, API reference |

**Branch:** `main` (IL pipeline and VR recording merged via PR #3 and PR #2)

Upstream baseline: [Trossen AI Isaac tutorial](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_ai_isaac.html)

## Environment

| Component | Version / location |
|-----------|-------------------|
| OS | Ubuntu 22.04 |
| Isaac Sim | 5.1.0 (`~/isaacsim/`) |
| Isaac Lab | 2.3.0 (`~/IsaacLab/`) |
| Extension | `trossen_ai_isaac` (`~/trossen_ai_isaac/`) |
| LeRobot (recording verify) | `~/lerobot_trossen/.venv` |
| LeRobot (training smoke) | `lerobot_train` conda env |
| VR headset | Meta Quest 3 |
| VR stack | ALVR + SteamVR (OpenXR runtime) |

## Quick verification checklist

Use this on a configured workstation to confirm the pipeline is working (from the repo root):

- [ ] Mobile AI tasks appear in env list:
  ```bash
  cd ~/trossen_ai_isaac
  ~/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py | grep MobileAI
  ```
- [ ] Robot model loads in standalone Isaac Sim:
  ```bash
  ~/isaacsim/python.sh scripts/demos/robot_bringup.py mobile_ai
  ```
- [ ] Keyboard or gamepad teleoperation runs:
  ```bash
  ~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_switch.py \
    --task Isaac-Reach-MobileAI-IK-Abs-Play-v0 --teleop_device keyboard
  ```
- [ ] (Optional, after reading Epic 4) VR teleoperation session smoke:
  ```bash
  ~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_vr.py \
    --task Isaac-Reach-MobileAI-IK-Abs-Play-v0
  ```
- [ ] (Optional, with headset) VR LeRobot recording smoke:
  ```bash
  ~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm_vr.py \
    --repo_id USER/vr_dataset_name --root ~/lerobot_trossen/datasets/vr_dataset_name
  ```

## Superseded documentation

The earlier `sim-vr-onboarding` notes are **out of date** and kept for historical reference only. Use the epic guides in this `docs/` folder instead.
