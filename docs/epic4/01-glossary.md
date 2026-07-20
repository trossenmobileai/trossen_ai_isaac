# Glossary (VR)

VR-specific abbreviations and terms. Shared IL terms (IK, IK-Abs, 16D, OpenXR, etc.) are in the [Epic 3 glossary](../epic3/01-glossary.md).

## Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| **ALVR** | Air Light VR (wireless PC-to-headset streaming) |
| **AR** | Augmented reality (Isaac Sim "Start AR" mode for OpenXR output) |
| **DLSS** | Deep Learning Super Sampling (NVIDIA anti-aliasing used in VR rendering) |
| **FPV** | First-person view |
| **VR** | Virtual reality |

## Terms

| Term | Definition |
|------|------------|
| **Meta Quest 3** | Standalone VR headset used for display and hand tracking in this project. |
| **ALVR** | Open-source wireless streaming bridge from the PC to Quest headsets over Wi-Fi. |
| **SteamVR** | PC VR runtime; manages the compositor and device drivers. |
| **OpenXR** | Khronos standard API; Isaac Sim uses it to read headset pose and hand tracking. |
| **Hand-anchored mode** | Default VR control mode: robot end-effector follows **relative** hand motion from a snapshot, not absolute hand position in the room. |
| **Absolute mode** | Alternative VR control mode: hand pose maps directly to the inverse kinematics (IK) target. Intended for humanoid avatars, not room-scale Mobile AI use. |
| **Retargeter** | Isaac Lab component that converts OpenXR hand data into robot action terms (pose and gripper). |
| **Workstation keyboard sidecar** | Plain `Se3Keyboard` used only for session keys (N/M/B/J); the headset operator has no keyboard. |

## Continue reading

- [Background and stack](02-background-and-stack.md)
- [VR workstation one-time setup](../setup/vr-workstation.md)
- [§1 VR session startup](../IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time)
- [§2 Practice VR teleop](../IL_WORKFLOW_RUNBOOK.md#2-practice-vr-teleop-no-dataset)
- [Epic 4 design index](README.md)
