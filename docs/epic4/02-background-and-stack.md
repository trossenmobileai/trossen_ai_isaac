# Background and Stack

## Integration with Epic 3

VR does **not** replace the task environment. It replaces the **input device**. The same `Isaac-Reach-MobileAI-IK-Abs-Play-v0` task, the same **16D action vector**, and the same IK solver are used. Only the source of actions changes (hand poses instead of keyboard or gamepad input).

```mermaid
flowchart TB
  subgraph epic3 [Epic3_Foundation]
    TaskEnv["Isaac-Reach-MobileAI-IK-Abs-Play-v0"]
    IK["16D IK-Abs actions"]
  end
  subgraph inputs [InputDevices]
    Keyboard["Keyboard or gamepad via teleop_dual_arm_switch.py"]
    VR["VR hands via teleop_dual_arm_vr.py"]
  end
  Keyboard --> IK
  VR --> IK
  IK --> TaskEnv
```

The VR control loop matches Epic 3's teleoperation loop (`input → 16D action → env.step()`), with OpenXR hand tracking as the input source. See [Teleoperation (Epic 3)](../epic3/03-teleoperation.md) and [VR teleoperation](03-vr-teleoperation.md).

## VR system stack

| Component | Role |
|-----------|------|
| **Meta Quest 3** | VR display and hand tracking |
| **ALVR** | Wireless streaming from PC to headset over Wi-Fi |
| **SteamVR** | PC VR runtime; manages compositor and device drivers |
| **OpenXR** | Standard API used by Isaac Sim to read headset pose and hand tracking |
| **Isaac Sim / Isaac Lab** | Renders stereo frames; `OpenXRDevice` converts hand poses to robot actions |

```mermaid
flowchart LR
  Quest3["Meta Quest 3 hand tracking"]
  ALVR["ALVR"]
  SteamVR["SteamVR"]
  OpenXR["OpenXR runtime"]
  IsaacSim["Isaac Sim AR / OpenXR"]
  IsaacLab["Isaac Lab OpenXRDevice"]
  TeleopVR["teleop_dual_arm_vr.py"]
  MobileAI["Mobile AI dual-arm IK"]
  Quest3 --> ALVR --> SteamVR --> OpenXR --> IsaacSim --> IsaacLab --> TeleopVR --> MobileAI
```

### Why this stack

No single product gives **Quest wireless streaming** and **Isaac Lab hand retargeting** into the Mobile AI IK task. The chain above exists because each hop owns one concern; Isaac Sim never talks to the Quest or ALVR directly.

Reading the diagram **left → right**:

1. **Meta Quest 3 (hand tracking)** — Standalone headset: stereo display plus inside-out cameras that track the wearer’s hands. Avoids a tethered PC VR HMD and physical controllers for this project’s demos.

2. **ALVR** — Wireless bridge on the same Wi-Fi LAN. Encodes/decodes frames between the workstation GPU and the Quest, and injects headset + hand tracking into the PC VR path. Chosen over cloud streaming (e.g. NVIDIA CloudXR) because it runs entirely on the local workstation. See [Findings — ALVR selection](05-findings-troubleshooting.md#design-notes).

3. **SteamVR** — PC VR compositor and driver host. ALVR registers as a SteamVR driver; without SteamVR there is no standard place for ALVR to publish devices. Session rule: always **Launch SteamVR from ALVR**, not from the Steam library alone ([§1 VR session startup](../IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time)).

4. **OpenXR runtime** — Khronos API that Isaac Sim uses for XR. SteamVR is set as the active OpenXR runtime so Isaac loads SteamVR’s OpenXR layer, which in turn sees ALVR’s devices. Isaac does not integrate with ALVR’s own API.

5. **Isaac Sim AR / OpenXR** — With Output Plugin = OpenXR and **Start AR**, the sim renders stereo frames into the OpenXR swapchain (what the Quest displays via ALVR).

6. **Isaac Lab `OpenXRDevice` → `teleop_dual_arm_vr.py` → Mobile AI IK** — Lab converts OpenXR hand poses into the same **16D** absolute IK actions as keyboard/gamepad teleop, then `env.step()` drives the dual-arm scene.

```mermaid
flowchart LR
  Quest["Quest sensors"]
  ALVR["ALVR"]
  SteamVR["SteamVR"]
  OpenXR["OpenXR runtime"]
  Isaac["Isaac Sim / Lab"]
  IK["Robot IK"]

  Quest -->|"Wi-Fi tracking"| ALVR
  ALVR -->|"driver"| SteamVR
  SteamVR -->|"OpenXR"| OpenXR
  OpenXR --> Isaac
  Isaac -->|"16D actions"| IK
  Isaac -->|"stereo frames / compositor"| ALVR
  ALVR -->|"Wi-Fi display"| Quest
```

One-time VR host install: [VR workstation setup](../setup/vr-workstation.md). Every session: [runbook §1](../IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time). Copy-paste after VR is live: [§2 Practice VR teleop](../IL_WORKFLOW_RUNBOOK.md#2-practice-vr-teleop-no-dataset) · [§3 Collect VR](../IL_WORKFLOW_RUNBOOK.md#3-collect-demos-vr).

## Continue reading

- [VR workstation one-time setup](../setup/vr-workstation.md) / [§1 VR session startup](../IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time)
- [§2 Practice VR teleop](../IL_WORKFLOW_RUNBOOK.md#2-practice-vr-teleop-no-dataset)
- [§3 Collect VR](../IL_WORKFLOW_RUNBOOK.md#3-collect-demos-vr)
- [Epic 4 hub](../EPIC4_VR_INTEGRATION.md)
