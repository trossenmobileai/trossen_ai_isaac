# Trossen AI Isaac Sim

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-orange.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) [![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.3.0-orange.svg)](https://isaac-sim.github.io/IsaacLab) [![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html) [![Linux](https://img.shields.io/badge/platform-Ubuntu_22.04-lightgrey.svg)](https://releases.ubuntu.com/22.04/)

## Overview

This repository integrates **NVIDIA Isaac Sim** and **Isaac Lab** with **Trossen AI** robots. It provides USD models, demo scripts, inverse-kinematics helpers, and Isaac Lab tasks for reinforcement learning (RL) and imitation learning (IL).

**This repo is a fork** of [TrossenRobotics/trossen_ai_isaac](https://github.com/TrossenRobotics/trossen_ai_isaac). Upstream content (WidowX AI / Stationary AI assets, demos, WXAI RL tasks, controller API) is kept. On top of that, this fork adds a **Mobile AI simulation → LeRobot dataset → ACT / Pi0 train → closed-loop sim eval** pipeline, including VR teleoperation and recording.

### What this repository offers

**From upstream (Trossen Robotics)**

- Isaac Sim USD models: WidowX AI (WXAI), Stationary AI, Mobile AI
- Robot bringup and pick-and-place / follow-target demos
- Differential IK controller for Cartesian end-effector control
- Isaac Lab RL tasks for WXAI (reach, lift, cabinet) via RSL-RL
- Leader-arm and Se3 teleoperation for WXAI

**Added in this fork (Mobile AI IL project)**

- Mobile AI Reach / Record / Lift play environments for teleop, dataset collection, and policy rollout
- Dual-arm keyboard/gamepad and VR teleoperation
- LeRobot v3 recording, verify, ACT / Pi0 training wrappers, and sim evaluation (sidecar)
- Project reports under [`docs/`](docs/README.md) (Epic 3, Epic 4, IL cheat sheet, ACT eval report)

### Tested environment

- Ubuntu 22.04
- Isaac Sim 5.1.0 (`~/isaacsim/`)
- Isaac Lab 2.3.0 (`~/IsaacLab/`)
- NVIDIA GeForce RTX 5090 (other CUDA GPUs may work; adjust env counts for VRAM)

---

## New here? Start with this

If you just cloned the fork, use this path:

1. **Install** Isaac Lab + this extension → [Installation](#installation)
2. **Sanity-check** that Mobile AI tasks appear → [Quick verification](#quick-verification)
3. **Skim** how the repo is organized → [Repository layout](#repository-layout)
4. **Read the project docs** for the Mobile AI IL story (not everything lives in this README):

| Doc | When to read it |
|-----|-----------------|
| [`docs/README.md`](docs/README.md) | **Repo docs index** — goals, timelines, page maps, env, checklist |
| [`docs/epic3/`](docs/epic3/) | Epic 3 design pages (tasks, teleop, recording, training, evaluation) |
| [`docs/epic4/`](docs/epic4/) | Epic 4 design pages (VR stack, teleop, recording) |
| [`docs/IL_WORKFLOW_CHEATSHEET.md`](docs/IL_WORKFLOW_CHEATSHEET.md) | Collect → verify → train → eval commands |
| [`docs/ACT_EVAL_REPORT_100K.md`](docs/ACT_EVAL_REPORT_100K.md) | ACT 100k closed-loop results (56.7%) |
| [`docs/EPIC3_SIMULATION_TRAINING_PIPELINE.md`](docs/EPIC3_SIMULATION_TRAINING_PIPELINE.md) / [`docs/EPIC4_VR_INTEGRATION.md`](docs/EPIC4_VR_INTEGRATION.md) | **BookStack book intros** (same content as sections in `docs/README.md`) |

Official upstream tutorial: [Trossen AI Isaac](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_ai_isaac.html).

---

## Index

- [Overview](#overview)
- [New here? Start with this](#new-here-start-with-this)
- [Installation](#installation)
- [Quick verification](#quick-verification)
- [Repository layout](#repository-layout)
- [Robot assets](#robot-assets)
- [Isaac Sim demo scripts](#isaac-sim-demo-scripts)
- [Isaac Lab demo tasks](#isaac-lab-demo-tasks)
- [Teleoperation](#teleoperation)
- [Imitation learning pipeline (Mobile AI)](#imitation-learning-pipeline-mobile-ai)
- [Controller API](#controller-api)
- [Related links](#related-links)

---

## Installation

### Prerequisites

Install **Isaac Lab 2.3.0** following the [official installation guide](https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/setup/installation/index.html). That also installs **Isaac Sim 5.1.0**.

Recommended method for this project: **Binary Isaac Sim + source Isaac Lab** (binary download for Isaac Sim, git clone for Isaac Lab). **Note:** That differs from NVIDIA’s “recommended” all-in-one path in the upstream docs; it matches how this workstation and the Epic docs are set up.

Default paths used everywhere below (**examples from this workstation — replace with your real install locations**):

| Component | Path |
|-----------|------|
| Isaac Sim | `~/isaacsim/` |
| Isaac Lab | `~/IsaacLab/` |
| This repo | `~/trossen_ai_isaac/` |

> **Do not copy-paste paths blindly.** Paths like `~/trossen_ai_isaac`, `~/IsaacLab`, `~/isaacsim`, and `~/lerobot_trossen/...` must match where those folders actually live on **your** machine. Adjust every command before running it.

### Clone this fork

```bash
git clone https://github.com/trossenmobileai/trossen_ai_isaac.git
cd ~/trossen_ai_isaac   # if you cloned elsewhere, cd into that clone instead
```

If the clone directory is not already at `~/trossen_ai_isaac`, either move/symlink it there or use your actual path in every command below.

Upstream (optional, for comparing against Trossen Robotics):

```bash
cd ~/trossen_ai_isaac
git remote add upstream https://github.com/TrossenRobotics/trossen_ai_isaac.git
```

### Install the Trossen AI extension (Isaac Lab)

```bash
cd ~/trossen_ai_isaac   # required
~/IsaacLab/isaaclab.sh -p -m pip install -e source/trossen_ai_isaac
```

Verify environments are registered:

```bash
cd ~/trossen_ai_isaac
~/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
```

You should see **WXAI** tasks (upstream) and **Mobile AI** tasks (this fork), including:

```
| … | Isaac-Reach-WXAI-v0                    | … |
| … | Isaac-Lift-Cube-WXAI-v0                | … |
| … | Isaac-Open-Drawer-WXAI-v0              | … |
| … | Isaac-Reach-MobileAI-IK-Abs-Play-v0    | … |  # keyboard / gamepad / VR teleop
| … | Isaac-Reach-MobileAI-Record-Play-v0    | … |  # LeRobot recording
| … | Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0 | … |  # closed-loop policy eval
```

### Extra tools for the IL pipeline

Recording/verify and policy training use separate Python environments from Isaac Sim (different Python versions). Full setup is in [Tasks and scene](docs/epic3/02-tasks-and-scene.md) / [cheat sheet](docs/IL_WORKFLOW_CHEATSHEET.md). In short you will need:

- LeRobot verify venv (e.g. `~/lerobot_trossen/.venv`) for dataset checks
- `lerobot_train` conda env for ACT / Pi0 training and the eval sidecar

You can install Isaac Lab and explore demos/teleop **before** configuring LeRobot.

---

## Quick verification

After installation, always enter the clone first — relative `scripts/...` paths only work from this repository:

```bash
cd ~/trossen_ai_isaac   # required — use YOUR clone path if different
```

> Paths in the commands below (`~/trossen_ai_isaac`, `~/IsaacLab`, `~/isaacsim`, …) are **examples**. Replace them with the real locations on your system before running anything.

Then:

```bash
# List all registered envs, then scan the output manually for Mobile AI tasks.
# Expected Mobile AI entries:
#   Isaac-Reach-MobileAI-IK-Abs-Play-v0          (teleop)
#   Isaac-Reach-MobileAI-Record-Play-v0          (recording)
#   Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0   (closed-loop eval)
~/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py

# USD model loads in standalone Isaac Sim?
~/isaacsim/python.sh scripts/demos/robot_bringup.py mobile_ai

# Keyboard teleop (quit from the Isaac window when done)?
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_switch.py \
  --task Isaac-Reach-MobileAI-IK-Abs-Play-v0 --teleop_device keyboard
```

A longer checklist (including VR) is in [`docs/README.md`](docs/README.md).

---

## Repository layout

Runnable scripts live under `scripts/`; reusable logic lives in the installed `trossen_ai_isaac` package.

```
trossen_ai_isaac/
├── assets/robots/              # USD models (WXAI, Stationary AI, Mobile AI)
├── docs/                       # Epic reports + IL glossary (start here for project depth)
├── scripts/
│   ├── demos/                  # Standalone Isaac Sim demos (isaacsim/python.sh)
│   ├── lib/                    # controller.py, leader_arm.py (used by demos)
│   ├── teleoperation/          # Human-in-the-loop entrypoints (isaaclab.sh -p)
│   ├── imitation_learning/     # Record, verify, train wrappers, play/eval
│   ├── reinforcement_learning/ # RSL-RL train/play (WXAI)
│   └── tools/list_envs.py
├── source/trossen_ai_isaac/    # Installable Isaac Lab extension + task configs
└── outputs/                    # Local train/eval artifacts (not required to clone)
```

| Location | Role | How you run it |
|----------|------|----------------|
| `scripts/demos/` | Standalone Isaac Sim demos | `~/isaacsim/python.sh scripts/demos/...` |
| `scripts/teleoperation/` | Teleop entrypoints | `~/IsaacLab/isaaclab.sh -p scripts/teleoperation/...` |
| `scripts/imitation_learning/` | IL entrypoints | `isaaclab.sh -p` (record) or bash/Python (verify/train/eval) |
| `scripts/reinforcement_learning/` | RSL-RL train/play | `~/IsaacLab/isaaclab.sh -p ...` |
| `source/.../tasks/` | Gym task configs | Imported after `pip install -e` |
| `docs/` | Project reports | Read in a browser / editor |

**Mobile AI gym tasks (this fork):**

| Task | Use |
|------|-----|
| `Isaac-Reach-MobileAI-IK-Abs-Play-v0` | Keyboard / gamepad / VR teleop |
| `Isaac-Reach-MobileAI-Record-Play-v0` | LeRobot demonstration recording |
| `Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0` | Closed-loop ACT / Pi0 rollout |

> **Naming note:** These IDs still say **Reach** / **Lift** because that is how they were registered during development (following Isaac Lab WXAI naming). They are **not** classic reach-to-target or lift-only RL tasks. All three are variants of the same **pick-and-place** scene (table + cube: pick up, lift, place back). The gym IDs were left unchanged so existing scripts and docs keep working.

---

## Robot assets

USD models live in `assets/robots/`:

```
assets/robots/
├── mobile_ai/mobile_ai.usd
├── stationary_ai/stationary_ai.usd
└── wxai/
    ├── wxai_base.usd
    ├── wxai_follower.usd
    ├── wxai_leader_left.usd
    └── wxai_leader_right.usd
```

USD files are generated from URDFs in [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description). See [assets/robots/asset_generation.md](assets/robots/asset_generation.md) for regeneration steps.

---

## Isaac Sim demo scripts

Commands assume Isaac Sim at `~/isaacsim/`.

### Robot bringup

```bash
~/isaacsim/isaac-sim.sh scripts/demos/robot_bringup.py [robot_name]
```

Supported: `wxai_base` (default), `wxai_follower`, `wxai_leader_left`, `wxai_leader_right`, `stationary_ai`, `mobile_ai`.

### Pick and place

```bash
~/isaacsim/python.sh scripts/demos/wxai_pick_place.py
~/isaacsim/python.sh scripts/demos/stationary_ai_pick_place.py
~/isaacsim/python.sh scripts/demos/mobile_ai_pick_place.py
```

### Follow target (differential IK)

```bash
~/isaacsim/python.sh scripts/demos/wxai_follow_target.py
```

---

## Isaac Lab demo tasks

Commands assume Isaac Lab at `~/IsaacLab/`.

**WXAI tasks (upstream):**

- `Isaac-Reach-WXAI-v0` — joint-position reach
- `Isaac-Reach-WXAI-IK-Rel-v0` / `Isaac-Reach-WXAI-IK-Abs-v0` — IK action variants
- `Isaac-Lift-Cube-WXAI-v0` — lift a cube
- `Isaac-Open-Drawer-WXAI-v0` — open a cabinet drawer

### Reinforcement learning (WXAI)

```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-WXAI-v0 \
    --num_envs 32 \
    --max_iterations 4000 \
    --headless
```

- `--num_envs` — parallel envs (tune for GPU memory)
- `--max_iterations` — training length
- `--headless` — no GUI (faster)

Logs/checkpoints: `logs/rsl_rl/<task>/<timestamp>/`.

Resume:

```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-WXAI-v0 \
    --num_envs 32 \
    --headless \
    --resume \
    --load_run <timestamp> \
    --checkpoint <model>.pt
```

Play a trained policy:

```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Reach-WXAI-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/<task>/<timestamp>/<model>.pt
```

### WXAI teleoperation (upstream Se3)

```bash
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_se3_agent.py \
    --task Isaac-Reach-WXAI-IK-Rel-v0 \
    --teleop_device keyboard
```

Devices: `keyboard`, `spacemouse`, `gamepad`.

---

## Teleoperation

| Script | Robot | Notes |
|--------|-------|--------|
| `teleop_dual_arm_switch.py` | Mobile AI | Keyboard/gamepad IK-Abs |
| `teleop_dual_arm_vr.py` | Mobile AI | VR hand tracking (OpenXR + ALVR); see [VR teleoperation](docs/epic4/04-vr-teleoperation.md) |
| `record_dual_arm.py` | Mobile AI | Keyboard/gamepad → LeRobot dataset |
| `record_dual_arm_vr.py` | Mobile AI | VR → LeRobot dataset |
| `teleop_se3_agent.py` | WXAI | Generic Se3 teleop |
| `teleop_leader_arm.py` | WXAI | Hardware leader arm → sim |

```bash
cd ~/trossen_ai_isaac   # required — run teleop from this clone
# Mobile AI keyboard
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_switch.py \
    --task Isaac-Reach-MobileAI-IK-Abs-Play-v0 --teleop_device keyboard

# Mobile AI VR (headset + ALVR/SteamVR OpenXR required)
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_vr.py \
    --task Isaac-Reach-MobileAI-IK-Abs-Play-v0
```

Keyboard/gamepad Mobile AI: **J** resets the environment (same as VR; gamepad uses **B**). VR defaults to single-arm control (TAB switches arms). Pass `--dual_arm` for both arms. Full CLI flags and VR setup: [VR teleoperation](docs/epic4/04-vr-teleoperation.md) / [docs index — Epic 4](docs/README.md#epic-4--vr-integration) (keyboard/gamepad keys: [Teleoperation](docs/epic3/03-teleoperation.md)).

### Leader arm (WXAI hardware)

The `trossen_arm` package is installed with the extension. For standalone Isaac Sim scripts you can also:

```bash
~/isaacsim/python.sh -m pip install trossen_arm
```

```bash
# Standalone Isaac Sim
~/isaacsim/python.sh scripts/demos/wxai_leader_to_sim.py

# Isaac Lab (joint-pos / IK mode inferred from task name)
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_leader_arm.py \
    --task Isaac-Lift-Cube-WXAI-v0
```

Pass `--leader_ip` to override the default (`192.168.1.2`).

---

## Imitation learning pipeline (Mobile AI)

High-level flow for **this fork’s** Mobile AI work:

```text
Record demos (VR production; keyboard optional)
        ↓
Verify LeRobot v3 dataset
        ↓
Train ACT and/or Pi0  (lerobot_train conda)
        ↓
Evaluate in Isaac Sim  (policy sidecar + lift play env)
```

**Production demos:** VR only, `--record_arm right` (`mobile_ai_right_pick_place_20260714_v2`). Keyboard/gamepad recording is supported for smoke tests. Step-by-step: [IL Workflow Cheat Sheet](docs/IL_WORKFLOW_CHEATSHEET.md).

| Stage | Entry point | Details |
|-------|-------------|---------|
| Record (VR, production) | `scripts/imitation_learning/run_collect_dataset.sh` | [VR recording](docs/epic4/05-vr-recording.md), [cheat sheet](docs/IL_WORKFLOW_CHEATSHEET.md) |
| Record (keyboard, smoke) | `scripts/imitation_learning/recording/record_dual_arm.py` | [cheat sheet](docs/IL_WORKFLOW_CHEATSHEET.md) |
| Merge VR shards | `scripts/imitation_learning/run_merge_dataset.sh` | [VR recording](docs/epic4/05-vr-recording.md), [cheat sheet](docs/IL_WORKFLOW_CHEATSHEET.md) |
| Verify | `scripts/imitation_learning/validation/verify_dataset.py` | [cheat sheet](docs/IL_WORKFLOW_CHEATSHEET.md) |
| Train ACT (10k wrapper) | `scripts/imitation_learning/run_verify_and_train.sh` | [Training](docs/epic3/05-training.md) |
| Train Pi0 | `scripts/imitation_learning/run_train_pi0.sh` | [Training](docs/epic3/05-training.md) |
| Open-loop replay | `scripts/imitation_learning/run_play_replay.sh` | [Evaluation](docs/epic3/06-evaluation.md) |
| Closed-loop ACT eval | `scripts/imitation_learning/run_play_act.sh` | [Evaluation](docs/epic3/06-evaluation.md) |
| Closed-loop Pi0 eval | `scripts/imitation_learning/run_play_pi0.sh` | Deferred — [Evaluation](docs/epic3/06-evaluation.md) |

**Example — production VR recording (wrapper):**

```bash
cd ~/trossen_ai_isaac   # required
./scripts/imitation_learning/run_collect_dataset.sh
./scripts/imitation_learning/run_merge_dataset.sh --verify
```

**Example — keyboard recording (alternate):**

```bash
cd ~/trossen_ai_isaac
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm.py \
    --task Isaac-Reach-MobileAI-Record-Play-v0 \
    --repo_id USER/dataset_name \
    --root ~/lerobot_trossen/datasets/dataset_name \
    --fps 60 --enable_cameras --record_arm right
```

**Trained artifacts on this project** (same VR right-arm pick-place dataset):

| Job | Policy | Steps | Role |
|-----|--------|-------|------|
| `act_mobile_ai_right_v2` | ACT | 10 000 | First full train |
| `act_mobile_ai_right_v2_100k` | ACT | 100 000 | Reporting eval model |
| `pi0_mobile_ai_right_v2` | Pi0 | 10 000 | Trained; sim eval deferred |

Checkpoints live under `~/trossen_ai_isaac/outputs/train/<job>/checkpoints/` when present on the machine.

**Reporting ACT eval (summary):** 30 episodes @ 60 FPS on the 100k checkpoint → **56.7% success (17/30)**. Full analysis: [ACT Evaluation Report](docs/ACT_EVAL_REPORT_100K.md). Training / procedure: [Training](docs/epic3/05-training.md), [Evaluation](docs/epic3/06-evaluation.md), [cheat sheet](docs/IL_WORKFLOW_CHEATSHEET.md#5-evaluate-closed-loop).

```bash
cd ~/trossen_ai_isaac
./scripts/imitation_learning/run_play_act.sh \
  ~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2_100k/checkpoints/last/pretrained_model \
  30 60
```

Sim eval uses a **sidecar** process because Isaac Sim (Python 3.11) and LeRobot training (Python 3.12) differ. Real-robot equivalent: [`lerobot-record --policy.path=...`](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot_plugin/train_and_evaluate.html).

Do **not** expect this README to list every flag and hyperparameter — use the [cheat sheet](docs/IL_WORKFLOW_CHEATSHEET.md) or Epic 3 / Epic 4.

---

## Controller API

`TrossenAIController` in `scripts/lib/controller.py` is the unified interface for standalone Isaac Sim demos (upstream).

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from controller import RobotType, TrossenAIController

robot = TrossenAIController(
    robot_path="/World/wxai_robot",
    robot_type=RobotType.WXAI,
    arm_dof_indices=[0, 1, 2, 3, 4, 5],
    gripper_dof_index=6,
    default_dof_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044],
)

robot.set_end_effector_pose(
    position=np.array([0.3, 0.0, 0.2]),
    orientation=np.array([0.7071, 0.0, 0.7071, 0.0]),  # [w, x, y, z]
)
robot.open_gripper()
robot.close_gripper()
robot.reset_to_default_pose()
```

---

## Related links

**Upstream / vendors**

- [Trossen Robotics](https://www.trossenrobotics.com/)
- [Trossen Arm Documentation](https://docs.trossenrobotics.com/trossen_arm/)
- [Trossen AI Isaac tutorial](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_ai_isaac.html)
- [Trossen Arm Description (URDF)](https://github.com/TrossenRobotics/trossen_arm_description)
- [Upstream repo](https://github.com/TrossenRobotics/trossen_ai_isaac)
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

**This fork**

- [Mobile AI docs index](docs/README.md)
- [IL Workflow Cheat Sheet](docs/IL_WORKFLOW_CHEATSHEET.md)
- [ACT Evaluation Report](docs/ACT_EVAL_REPORT_100K.md)
- [Epic 3 pages](docs/epic3/)
- [Epic 4 pages](docs/epic4/)
- BookStack intros: [Epic 3 hub](docs/EPIC3_SIMULATION_TRAINING_PIPELINE.md), [Epic 4 hub](docs/EPIC4_VR_INTEGRATION.md)
