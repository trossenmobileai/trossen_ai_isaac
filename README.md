# Trossen AI Isaac Sim

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-orange.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) [![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.3.0-orange.svg)](https://isaac-sim.github.io/IsaacLab) [![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html) [![Linux](https://img.shields.io/badge/platform-Ubuntu_22.04-lightgrey.svg)](https://releases.ubuntu.com/22.04/)

## Overview

This repository contains NVIDIA Isaac Sim and Isaac Lab integration for Trossen AI robotic arms. It includes USD robot models, inverse kinematics-based task examples, and Isaac Lab tasks for reinforcement learning and imitation learning.

### What This Repository Offers

- Isaac Sim USD models for Trossen AI robots:
  - WidowX AI (single arm base, follower, leader left, leader right)
  - Stationary AI (dual-arm stationary platform)
  - Mobile AI (dual-arm mobile manipulator)
- Robot bringup utilities for quick model visualization and testing
- Differential inverse kinematics controller for Cartesian end-effector control
- Example scripts for pick-and-place and target following tasks
- Isaac Lab tasks for reinforcement learning (eg. reach, lift, cabinet)
- Teleoperation interface for imitation learning data collection

### Tested Environment

- Ubuntu 22.04
- Isaac Sim 5.1.0
- Isaac Lab 2.3.0
- NVIDIA GeForce RTX 5090

---

## Index

- [Overview](#overview)
- [Installation](#installation)
- [Repository Layout](#repository-layout)
- [Robot Assets](#robot-assets)
- [Isaac Sim Demo Scripts](#isaac-sim-demo-scripts)
- [Isaac Lab Demo Tasks](#isaac-lab-demo-tasks)
- [Teleoperation](#teleoperation)
- [Imitation Learning Pipeline](#imitation-learning-pipeline)
- [Controller API](#controller-api)
- [IL Pipeline Branches](docs/IL_PIPELINE_BRANCHES.md)
- [Mobile AI Epic Docs](docs/README.md) — Epic 3 (simulation training) and Epic 4 (VR integration)
- [Related Links](#related-links)

---

## Installation

### Prerequisites

Install Isaac Lab 2.3.0 following the [official installation guide](https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/setup/installation/index.html). This will also install Isaac Sim 5.1.0. Recommended Installation Method: Binary + Source (binary download for Isaac Sim + source via git for Isaac Lab). **Note:** The recommended installation method mentioned here (Binary + Source) differs from the official recommended method.

### Clone Repository

```bash
git clone https://github.com/TrossenRobotics/trossen_ai_isaac.git
cd trossen_ai_isaac
```

### Install Trossen AI Extension (for Isaac Lab)

```bash
~/IsaacLab/isaaclab.sh -p -m pip install -e source/trossen_ai_isaac
```

Verify the environments are registered:

```bash
~/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
```

You should see output similar to:

```
| 1  | Isaac-Open-Drawer-WXAI-v0      | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.cabinet.joint_pos_env_cfg:WXAICabinetEnvCfg           |
| 2  | Isaac-Open-Drawer-WXAI-Play-v0 | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.cabinet.joint_pos_env_cfg:WXAICabinetEnvCfg_PLAY      |
| 3  | Isaac-Lift-Cube-WXAI-v0        | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.lift.config.joint_pos_env_cfg:WXAICubeLiftEnvCfg      |
| 4  | Isaac-Lift-Cube-WXAI-Play-v0   | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.lift.config.joint_pos_env_cfg:WXAICubeLiftEnvCfg_PLAY |
| 5  | Isaac-Lift-Cube-WXAI-IK-Rel-v0 | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.lift.config.ik_rel_env_cfg:WXAICubeLiftEnvCfg         |
| 6  | Isaac-Lift-Cube-WXAI-IK-Abs-v0 | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.lift.config.ik_abs_env_cfg:WXAICubeLiftEnvCfg         |
| 7  | Isaac-Reach-WXAI-v0            | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.reach.config.joint_pos_env_cfg:WXAIReachEnvCfg        |
| 8  | Isaac-Reach-WXAI-Play-v0       | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.reach.config.joint_pos_env_cfg:WXAIReachEnvCfg_PLAY   |
| 9  | Isaac-Reach-WXAI-IK-Rel-v0     | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.reach.config.ik_rel_env_cfg:WXAIReachEnvCfg           |
| 10 | Isaac-Reach-WXAI-IK-Abs-v0     | isaaclab.envs:ManagerBasedRLEnv | trossen_ai_isaac.tasks.manager_based.manipulation.wxai.reach.config.ik_abs_env_cfg:WXAIReachEnvCfg           |
| 11 | Isaac-Reach-MobileAI-IK-Abs-Play-v0 | isaaclab.envs:ManagerBasedRLEnv | ... Mobile AI VR / keyboard teleop |
| 12 | Isaac-Reach-MobileAI-Record-Play-v0 | isaaclab.envs:ManagerBasedRLEnv | ... Mobile AI IL recording (14D obs + 3 cameras) |
```

---

## Repository Layout

Runnable scripts live under `scripts/`; reusable logic lives in the installed `trossen_ai_isaac` package.

```
scripts/
├── teleoperation/              # Human-in-the-loop control (isaaclab.sh -p)
├── imitation_learning/         # Recording, dataset QA, training smoke
│   ├── recording/record_dual_arm.py, record_dual_arm_vr.py
│   ├── smoke/smoke_record_env.py, smoke_record_dataset.py
│   ├── validation/verify_dataset.py
│   └── training/smoke_train_act.py
├── demos/                      # Standalone Isaac Sim scripts (isaacsim/python.sh)
├── lib/                        # controller.py, leader_arm.py (used by demos)
├── reinforcement_learning/     # RSL-RL train/play
└── tools/list_envs.py

source/trossen_ai_isaac/trossen_ai_isaac/
├── teleop/                     # Library: loops, session, CLI helpers, VR package
├── recording/                  # Library: LeRobot writer, smokes, runtime
├── validation/                 # Offline LeRobot dataset checks
└── training/                   # ACT training smoke helpers
```

| Location | Role | How you run it |
|----------|------|----------------|
| `scripts/teleoperation/` | Teleop **entrypoints** | `~/IsaacLab/isaaclab.sh -p scripts/teleoperation/...` |
| `scripts/imitation_learning/` | IL **entrypoints** | `isaaclab.sh -p` (recording) or plain Python (verify/train) |
| `source/.../teleop/` | Teleop **library** | Imported by scripts; not run directly |
| `source/.../recording/` | Recording **library** | Imported by IL scripts |

Mobile AI gym tasks (IL-focused):

- `Isaac-Reach-MobileAI-IK-Abs-Play-v0` — keyboard/gamepad/VR teleop (16D IK-Abs actions)
- `Isaac-Reach-MobileAI-Record-Play-v0` — LeRobot recording (14D joint obs, 3× RGB cameras @ 60 Hz)

---

## Robot Assets

All robot USD models are located in `assets/robots/`:

```
assets/robots/
├── mobile_ai/
│   └── mobile_ai.usd
├── stationary_ai/
│   └── stationary_ai.usd
└── wxai/
    ├── wxai_base.usd
    ├── wxai_follower.usd
    ├── wxai_leader_left.usd
    └── wxai_leader_right.usd
```

### Asset Generation

All USD files are generated from URDF descriptions in [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description). See [assets/robots/asset_generation.md](assets/robots/asset_generation.md) for detailed generation instructions.

---

## Isaac Sim Demo Scripts

Note: Commands below assume Isaac Sim is installed at `~/isaacsim/`. Adjust the path if your installation directory differs.

### Robot Bringup

Load and visualize any robot model:

```bash
~/isaacsim/isaac-sim.sh scripts/demos/robot_bringup.py [robot_name]
```

Supported robots: `wxai_base` (default), `wxai_follower`, `wxai_leader_left`, `wxai_leader_right`, `stationary_ai`, `mobile_ai`

### Pick and Place Demo

```bash
~/isaacsim/python.sh scripts/demos/wxai_pick_place.py
~/isaacsim/python.sh scripts/demos/stationary_ai_pick_place.py
~/isaacsim/python.sh scripts/demos/mobile_ai_pick_place.py
```

### Follow Target Demo

Real-time end-effector tracking using differential IK:

```bash
~/isaacsim/python.sh scripts/demos/wxai_follow_target.py
```

---

## Isaac Lab Demo Tasks

**Note:** Commands below assume Isaac Lab is installed at `~/IsaacLab/`. Adjust the path if your installation directory differs.

Available tasks:
- `Isaac-Reach-WXAI-v0` - Move end-effector to target pose using joint position control
- `Isaac-Reach-WXAI-IK-Rel-v0` - Reach task with relative IK delta actions
- `Isaac-Reach-WXAI-IK-Abs-v0` - Reach task with absolute IK pose actions
- `Isaac-Lift-Cube-WXAI-v0` - Pick up a cube and lift it to a target height
- `Isaac-Open-Drawer-WXAI-v0` - Open a cabinet drawer by grasping and pulling

### Reinforcement Learning

Train a policy using RSL-RL PPO:

```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-WXAI-v0 \
    --num_envs 32 \
    --max_iterations 4000 \
    --headless
```

**Training Options:**
- `--num_envs 32`: Number of parallel environments (adjust based on GPU memory)
- `--max_iterations 4000`: Number of Iterations steps (adjust as per training tasks)
- `--headless`: Run without GUI for faster training

Training logs and checkpoints are saved to `logs/rsl_rl/<task>/<timestamp>/`.

Resume training from a checkpoint:

```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Reach-WXAI-v0 \
    --num_envs 32 \
    --headless \
    --resume \
    --load_run <timestamp> \
    --checkpoint <model>.pt
```

Run a trained policy:

```bash
~/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Reach-WXAI-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/<task>/<timestamp>/<model>.pt
```

### Imitation Learning

WXAI teleoperation for data collection:

```bash
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_se3_agent.py \
    --task Isaac-Reach-WXAI-IK-Rel-v0 \
    --teleop_device keyboard
```

**Teleop Device options:** keyboard, spacemouse, gamepad

---

## Teleoperation

| Script | Robot | Task / notes |
|--------|-------|----------------|
| `teleop_dual_arm_switch.py` | Mobile AI | Keyboard/gamepad IK-Abs teleop (`Isaac-Reach-MobileAI-IK-Abs-Play-v0`) |
| `teleop_dual_arm_vr.py` | Mobile AI | VR hand tracking (OpenXR + ALVR). Single-arm by default (TAB switches active arm, other arm frozen); pass `--dual_arm` for both arms at once |
| `record_dual_arm.py` | Mobile AI | Keyboard/gamepad LeRobot recording |
| `record_dual_arm_vr.py` | Mobile AI | VR LeRobot recording |
| `teleop_se3_agent.py` | WXAI | Generic Se3 keyboard/gamepad teleop |
| `teleop_leader_arm.py` | WXAI | Hardware leader arm → sim |

```bash
# Mobile AI keyboard teleop
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_switch.py \
    --task Isaac-Reach-MobileAI-IK-Abs-Play-v0 --teleop_device keyboard

# Mobile AI VR (requires headset + ALVR/SteamVR OpenXR runtime)
# Single-arm by default: only the active arm tracks its hand, the other holds
# its pose. Press TAB at the workstation to switch the active arm (--start_arm
# sets which arm starts). Add --dual_arm to drive both arms simultaneously.
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_dual_arm_vr.py \
    --task Isaac-Reach-MobileAI-IK-Abs-Play-v0
```

---

## Imitation Learning Pipeline

End-to-end flow for Mobile AI sim demonstrations → LeRobot v3 datasets → ACT training.

| Step | Script | Requires |
|------|--------|----------|
| Env smoke | `scripts/imitation_learning/smoke/smoke_record_env.py` | Isaac Sim |
| Dataset smoke | `scripts/imitation_learning/smoke/smoke_record_dataset.py` | Isaac Sim |
| Record demos (keyboard/gamepad) | `scripts/imitation_learning/recording/record_dual_arm.py` | Isaac Sim + LeRobot |
| Record demos (VR) | `scripts/imitation_learning/recording/record_dual_arm_vr.py` | Isaac Sim + LeRobot + VR stack |
| Merge VR shards (multi-session) | `scripts/imitation_learning/recording/merge_datasets.py` | LeRobot venv |
| Verify | `scripts/imitation_learning/validation/verify_dataset.py` | LeRobot venv (PyAV) |
| Train ACT | `scripts/imitation_learning/run_verify_and_train.sh` | `lerobot_train` conda |
| Train smoke | `scripts/imitation_learning/training/smoke_train_act.py` | `lerobot_train` conda |
| Replay episode (open-loop) | `scripts/imitation_learning/run_play_replay.sh` | Isaac Sim |
| Evaluate ACT policy | `scripts/imitation_learning/run_play_act.sh` | Isaac Sim + `lerobot_train` conda |

```bash
# Record demonstrations — keyboard/gamepad (N=toggle episode, M=discard, R=reset)
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm.py \
    --task Isaac-Reach-MobileAI-Record-Play-v0 \
    --repo_id USER/dataset_name \
    --root ~/lerobot_trossen/datasets/dataset_name \
    --fps 60 --enable_cameras

# Record demonstrations — VR (U=start teleop, N=toggle episode, M=discard, J=reset)
# --record_arm selects what goes into the dataset AND locks teleop to match:
#   both  (default) => 14D state/action + 3 cameras, both arms teleoperated
#   left / right     => 7D that-arm joints + cam_high + that arm's wrist camera,
#                       teleop locked to that arm (TAB disabled). Still a valid
#                       LeRobot v3 dataset, just fewer feature dims/cameras.
# --pose_smoothing (default 0.5): exponential low-pass on IK target pose to reduce
#   Quest hand-tracking jitter. 0=raw, 0.5=balanced, 0.7=very stable/more lag.
# Each call writes a self-contained shard; merge shards after all sessions (see below).
~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/recording/record_dual_arm_vr.py \
    --repo_id USER/dataset_name \
    --root ~/lerobot_trossen/datasets/my_dataset/shards/session_1 \
    --record_arm right \
    --fps 60

# Merge shards from multiple sessions into one LeRobot v3 dataset
~/lerobot_trossen/.venv/bin/python scripts/imitation_learning/recording/merge_datasets.py \
    --shards_dir ~/lerobot_trossen/datasets/my_dataset/shards \
    --repo_id USER/dataset_name \
    --root ~/lerobot_trossen/datasets/my_dataset/merged

# Verify offline
~/lerobot_trossen/.venv/bin/python scripts/imitation_learning/validation/verify_dataset.py \
    --root ~/lerobot_trossen/datasets/my_dataset/merged

# Short ACT training smoke (GPU via lerobot_train conda)
python scripts/imitation_learning/training/smoke_train_act.py \
    --root ~/lerobot_trossen/datasets/my_dataset/merged

# Full ACT training + verify (see scripts/imitation_learning/run_verify_and_train.sh)
./scripts/imitation_learning/run_verify_and_train.sh

# Phase A — open-loop replay sanity check (episode 0, 60 fps)
./scripts/imitation_learning/run_play_replay.sh \
    ~/lerobot_trossen/datasets/my_dataset/merged 0

# Phase B — closed-loop ACT evaluation (10 episodes, metrics in rollout_summary.json)
./scripts/imitation_learning/run_play_act.sh \
    ~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/last/pretrained_model \
    10 60

# Visual rollout (omit headless)
./scripts/imitation_learning/run_play_act.sh \
    ~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2/checkpoints/last/pretrained_model \
    1 --visual
```

Real-robot equivalent: `lerobot-record --policy.path=<checkpoint>` ([Trossen ACT evaluation docs](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/lerobot_plugin/train_and_evaluate.html)). Sim uses a sidecar process for ACT inference because Isaac Sim and LeRobot run on different Python versions.

See [docs/IL_PIPELINE_BRANCHES.md](docs/IL_PIPELINE_BRANCHES.md) for branch history and folder glossary. See [docs/EPIC4_VR_INTEGRATION.md](docs/EPIC4_VR_INTEGRATION.md) for VR recording setup and multi-session workflow details. See [docs/EPIC3_SIMULATION_TRAINING_PIPELINE.md](docs/EPIC3_SIMULATION_TRAINING_PIPELINE.md) §4.8 for sim ACT evaluation architecture and troubleshooting.

### Leader Arm Teleoperation

Control the simulated robot using a real Trossen WXAI leader arm. The `trossen_arm` package is installed automatically with the extension. If you need to install it separately (e.g. for the standalone Isaac Sim script):

```bash
~/isaacsim/python.sh -m pip install trossen_arm
```

**Standalone Isaac Sim:**

```bash
~/isaacsim/python.sh scripts/demos/wxai_leader_to_sim.py
```

**Isaac Lab environment (joint-pos, IK-abs, IK-rel auto-detected from task name):**

```bash
~/IsaacLab/isaaclab.sh -p scripts/teleoperation/teleop_leader_arm.py \
    --task Isaac-Lift-Cube-WXAI-v0
```

Pass `--leader_ip` to change the default arm address (`192.168.1.2`).

---

## Controller API

The `TrossenAIController` class in `scripts/lib/controller.py` provides a unified interface for controlling all Trossen AI robots in standalone Isaac Sim demos.

### Basic Usage

Add `scripts/lib` to your path (demos do this automatically), then:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from controller import RobotType, TrossenAIController

# Initialize controller
robot = TrossenAIController(
    robot_path="/World/wxai_robot",
    robot_type=RobotType.WXAI,
    arm_dof_indices=[0, 1, 2, 3, 4, 5],
    gripper_dof_index=6,
    default_dof_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044],
)

# Command end-effector pose
robot.set_end_effector_pose(
    position=np.array([0.3, 0.0, 0.2]),
    orientation=np.array([0.7071, 0.0, 0.7071, 0.0]),  # [w, x, y, z]
)

# Gripper control
robot.open_gripper()
robot.close_gripper()

# Reset to default pose
robot.reset_to_default_pose()
```

---

## Related Links

- [Trossen Robotics](https://www.trossenrobotics.com/)
- [Trossen Arm Documentation](https://docs.trossenrobotics.com/trossen_arm/)
- [Trossen Arm Description (URDF)](https://github.com/TrossenRobotics/trossen_arm_description)
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
