# Trossen Mobile AI — Simulation & VR Documentation

Documentation for the **Trossen Mobile AI** imitation-learning project:

| Doc | What it covers |
|-----|----------------|
| **[Epic 3](EPIC3_SIMULATION_TRAINING_PIPELINE.md)** | Isaac Lab tasks, keyboard/gamepad teleop, LeRobot pipeline, ACT / Pi0 training, eval architecture |
| **[Epic 4](EPIC4_VR_INTEGRATION.md)** | Quest 3 + ALVR + OpenXR teleoperation and **production VR dataset recording** |
| **[IL Workflow Cheat Sheet](IL_WORKFLOW_CHEATSHEET.md)** | Step-by-step collect (VR + keyboard) → verify → train → eval |
| **[ACT Evaluation Report](ACT_EVAL_REPORT_100K.md)** | Closed-loop ACT 100k / 30-episode results (56.7%) |

## Who this is for

New team members, stakeholders, and contributors who need to understand what was built, why, and how to run it — without prior Isaac Sim experience.

Epic reports follow a technical-report structure (Goal, Overview, Background, Implementation, Operational Procedures, Findings, Troubleshooting, Status). The cheat sheet and eval report are shorter operational/result docs.

## Reading order

1. **This page** — orientation (2 minutes)
2. **[IL Workflow Cheat Sheet](IL_WORKFLOW_CHEATSHEET.md)** — if you only need commands
3. **[Epic 3](EPIC3_SIMULATION_TRAINING_PIPELINE.md)** — full simulation and training story
4. **[Epic 4](EPIC4_VR_INTEGRATION.md)** — VR stack (production collection path)
5. **[ACT Evaluation Report](ACT_EVAL_REPORT_100K.md)** — reporting metrics

## Related docs in this repo

| Doc | Purpose |
|-----|---------|
| [Repository README](../README.md) | Clone/setup onboarding, repo map, upstream demos + IL overview |

**Branch:** `main` (all Mobile AI IL and VR work lives here)

Upstream baseline: [Trossen AI Isaac tutorial](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_ai_isaac.html)

## Environment

| Component | Version / location (examples — adjust to your machine) |
|-----------|-------------------|
| OS | Ubuntu 22.04 |
| Isaac Sim | 5.1.0 (`~/isaacsim/`) |
| Isaac Lab | 2.3.0 (`~/IsaacLab/`) |
| Extension | `trossen_ai_isaac` (`~/trossen_ai_isaac/`) |
| LeRobot (recording verify) | `~/lerobot_trossen/.venv` |
| LeRobot (training smoke) | `lerobot_train` conda env |
| VR headset | Meta Quest 3 |
| VR stack | ALVR + SteamVR (OpenXR runtime) |

**Mobile AI task names:** Gym IDs still contain **Reach** / **Lift** from early development; the intended task is **pick and place** (table + cube). Details: [Epic 3 §4.3](EPIC3_SIMULATION_TRAINING_PIPELINE.md#43-custom-reach-task-environment).

**Reporting dataset:** `mobile_ai_right_pick_place_20260714_v2` — collected with **VR**, `--record_arm right` only ([cheat sheet](IL_WORKFLOW_CHEATSHEET.md), [Epic 4 §5.4](EPIC4_VR_INTEGRATION.md#54-vr-recording-procedure)).

## Quick verification checklist

> Paths like `~/trossen_ai_isaac`, `~/IsaacLab`, `~/isaacsim`, and `~/lerobot_trossen/...` are **examples from this workstation**. Replace them with the real locations on your machine before running commands. Do not copy-paste blindly — especially `--root` and dataset paths.

Always start from the clone directory:

```bash
cd ~/trossen_ai_isaac   # required — use YOUR clone path if different
```

Then confirm the pipeline is working:

- [ ] Mobile AI tasks appear in env list — run the full list and **scan the output manually** (do not pipe through `grep`):
  ```bash
  cd ~/trossen_ai_isaac
  ~/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
  ```
  You should see these Mobile AI tasks among the WXAI entries:
  - `Isaac-Reach-MobileAI-IK-Abs-Play-v0` (teleop)
  - `Isaac-Reach-MobileAI-Record-Play-v0` (recording)
  - `Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0` (closed-loop eval)
- [ ] Robot model loads in standalone Isaac Sim:
  ```bash
  cd ~/trossen_ai_isaac
  ~/isaacsim/python.sh scripts/demos/robot_bringup.py mobile_ai
  ```
- [ ] Keyboard or gamepad teleoperation runs:
  ```bash
  cd ~/trossen_ai_isaac
  ~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_switch.py \
    --task Isaac-Reach-MobileAI-IK-Abs-Play-v0 --teleop_device keyboard
  ```
- [ ] (Optional, after reading Epic 4) VR teleoperation session smoke:
  ```bash
  cd ~/trossen_ai_isaac
  ~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_vr.py \
    --task Isaac-Reach-MobileAI-IK-Abs-Play-v0
  ```
- [ ] (Production path, with headset) VR LeRobot recording — prefer `run_collect_dataset.sh` with `--record_arm right`. Full flags: [Epic 3 §5.5](EPIC3_SIMULATION_TRAINING_PIPELINE.md#55-vr-recording--production-demonstrations), [Epic 4 §5.4](EPIC4_VR_INTEGRATION.md#54-vr-recording-procedure)
  ```bash
  cd ~/trossen_ai_isaac
  ~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm_vr.py \
    --repo_id USER/vr_dataset_name --root ~/lerobot_trossen/datasets/vr_dataset_name \
    --record_arm right
  ```
