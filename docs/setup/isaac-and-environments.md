# Isaac Sim, Lab, and environments (one-time)

Procedural install for a **new workstation**. Design history (registration, tasks, scene): [Tasks and scene](../epic3/02-tasks-and-scene.md). Upstream guide: [Trossen AI Isaac installation](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_ai_isaac.html).

> **Paths are examples.** Adjust `~/isaacsim`, `~/IsaacLab`, `~/trossen_ai_isaac`, `~/lerobot_trossen` to your machine.

## Prerequisites

| Component | Example version / location |
|-----------|----------------------------|
| OS | Ubuntu 22.04 |
| Isaac Sim | 5.1.0 (`~/isaacsim/`) |
| Isaac Lab | 2.3.0 (`~/IsaacLab/`) |
| This fork | `~/trossen_ai_isaac/` |

Follow the upstream Trossen guide for Isaac Sim + Isaac Lab install, then use **this fork** (not only upstream) for Mobile AI.

## Clone this fork

The fork lives under the shared **GitHub** account (`trossenmobileai@gmail.com`). Use that login if HTTPS/auth is required; ask the supervisor for the password — see [Shared team accounts](README.md#shared-team-accounts).

```bash
git clone https://github.com/trossenmobileai/trossen_ai_isaac.git
cd ~/trossen_ai_isaac   # if you cloned elsewhere, cd into that clone instead
```

Optional upstream remote (compare against Trossen Robotics):

```bash
cd ~/trossen_ai_isaac
git remote add upstream https://github.com/TrossenRobotics/trossen_ai_isaac.git
```

## Install the Trossen AI extension (Isaac Lab)

```bash
cd ~/trossen_ai_isaac
~/IsaacLab/isaaclab.sh -p -m pip install -e source/trossen_ai_isaac
```

## Verify gym registration

```bash
cd ~/trossen_ai_isaac
~/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
```

Confirm Mobile AI IDs appear:

- `Isaac-Reach-MobileAI-IK-Abs-Play-v0` — keyboard / gamepad / VR teleop
- `Isaac-Reach-MobileAI-Record-Play-v0` — LeRobot recording
- `Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0` — closed-loop policy eval

Optional USD bringup:

```bash
~/isaacsim/python.sh scripts/demos/robot_bringup.py mobile_ai
```

## Extra tools for the IL pipeline

Recording/verify and policy training use separate Python environments from Isaac Sim (different Python versions):

| Tooling | Used for |
|---------|----------|
| `~/IsaacLab/isaaclab.sh` (Isaac Sim Python 3.11) | Teleop, recording, closed-loop eval host, `list_envs` |
| `~/lerobot_trossen/.venv` | `verify_dataset.py` |
| `lerobot_train` conda (Python 3.12) | `lerobot-train`, policy sidecar during eval |

You can install Isaac Lab and explore demos/teleop **before** configuring LeRobot. Day-to-day commands after setup: [IL runbook](../IL_WORKFLOW_RUNBOOK.md).

## VR (if collecting with Quest)

After Isaac is working, complete [VR workstation one-time setup](vr-workstation.md), then every session use [runbook §1](../IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time).

## Continue reading

- [Setup hub](README.md)
- [VR workstation one-time setup](vr-workstation.md)
- [IL Workflow Runbook](../IL_WORKFLOW_RUNBOOK.md)
- [Tasks and scene](../epic3/02-tasks-and-scene.md) (design)
