# One-time setup

Use this lane when bringing up a **new workstation** or after a re-image. If the next team continues on the **same** already-validated machine, skip here and go straight to the [day-to-day runbook](../IL_WORKFLOW_RUNBOOK.md).

## When to use

| Situation | What to do |
|-----------|------------|
| New PC / re-image / first clone | Complete the checklist below |
| Same validated workstation | [IL Workflow Runbook](../IL_WORKFLOW_RUNBOOK.md) only |
| Understanding *why* (not how to install) | Design: [`docs/epic3/`](../epic3/), [`docs/epic4/`](../epic4/) |

## Checklist

1. **[Isaac Sim, Lab, and environments](isaac-and-environments.md)** — Ubuntu, Isaac Sim/Lab, editable extension, LeRobot verify + train envs, `list_envs` smoke.
2. **[VR workstation one-time setup](vr-workstation.md)** — Quest hand tracking, ALVR, SteamVR, `setcap`, OpenXR runtime, one-time ALVR→SteamVR smoke.
3. **Day-to-day:** [IL runbook — §1 VR session startup](../IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time) → practice / collect / train / eval.

## Continue reading

- [Isaac Sim, Lab, and environments](isaac-and-environments.md)
- [VR workstation one-time setup](vr-workstation.md)
- [IL Workflow Runbook](../IL_WORKFLOW_RUNBOOK.md)
- [Docs index](../README.md)
