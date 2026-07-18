# Epic 4 — VR Integration

> **Document status:** **BookStack book intro** for Epic 4 (keep this file when uploading). **Canonical narrative** (goals, timelines, checklist): [docs index](README.md). Design pages: [`epic4/`](epic4/). **One-time VR host:** [setup/vr-workstation](setup/vr-workstation.md). **Day-to-day:** [runbook §1](IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time) → [§2](IL_WORKFLOW_RUNBOOK.md#2-practice-vr-teleop-no-dataset) → [§3](IL_WORKFLOW_RUNBOOK.md#3-collect-demos-vr).

## Goal

Connect VR headsets to Isaac Sim for in-simulation teleoperation — safe demonstration practice and synthetic data collection without physical hardware risk.

## Start here

1. **New machine?** [Setup hub](setup/README.md) → [VR workstation one-time setup](setup/vr-workstation.md)
2. **Every session:** [§1 VR session startup](IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time)
3. **[§2 Practice VR teleop](IL_WORKFLOW_RUNBOOK.md#2-practice-vr-teleop-no-dataset)** — try hands before collecting
4. **[§3 Collect VR](IL_WORKFLOW_RUNBOOK.md#3-collect-demos-vr)** — demos + merge (single- or multi-session)
5. **[VR teleoperation](epic4/03-vr-teleoperation.md)** / **[VR recording](epic4/04-vr-recording.md)** — design detail
6. **[§6 Train](IL_WORKFLOW_RUNBOOK.md#6-train)** / **[§7 Evaluate](IL_WORKFLOW_RUNBOOK.md#7-evaluate-closed-loop)** — after demos exist

Full timeline and overview: [docs index — Epic 4](README.md#epic-4-vr-integration). This project’s reporting train set: [runbook project example reference](IL_WORKFLOW_RUNBOOK.md) (VR, `--record_arm right`).

## Pages (BookStack-ready)

| Page | Contents |
|------|----------|
| [Glossary](epic4/01-glossary.md) | VR abbreviations and terms |
| [Background and stack](epic4/02-background-and-stack.md) | Epic 3 integration + VR stack (why each hop) |
| [VR teleoperation](epic4/03-vr-teleoperation.md) | Module, wiring, CLI (design) |
| [VR recording](epic4/04-vr-recording.md) | Dataset modes, shards, smoothing (design) |
| [Findings and troubleshooting](epic4/05-findings-troubleshooting.md) | Limitations and VR/ALVR fixes |
| [Future work](epic4/06-future-work.md) | Follow-ups |

## Continue reading

- [Docs index](README.md) (full timeline / overview)
- [Setup hub](setup/README.md)
- [IL Workflow Runbook](IL_WORKFLOW_RUNBOOK.md)
- [Epic 3 hub](EPIC3_SIMULATION_TRAINING_PIPELINE.md)
