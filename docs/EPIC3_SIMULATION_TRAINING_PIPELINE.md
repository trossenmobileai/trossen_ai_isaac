# Epic 3 — Simulation Training Pipeline

> **Document status:** **BookStack book intro** for Epic 3 (keep this file when uploading). **Canonical narrative** (goals, timelines, environment, checklist): [docs index](README.md). Detailed design: [`epic3/`](epic3/README.md). **Setup:** [docs/setup](setup/README.md). **Runbook:** [IL runbook](IL_WORKFLOW_RUNBOOK.md). Pi0 sim eval remains deferred.

## Goal

Build a digital twin of the Trossen Mobile AI in Isaac Sim and an imitation-learning pipeline: record human demonstrations, train policies (ACT / Pi0), and evaluate closed-loop in simulation.

## Start here

1. **[IL Workflow Runbook](IL_WORKFLOW_RUNBOOK.md)** — day-to-day ([§0](IL_WORKFLOW_RUNBOOK.md#0-prerequisites)–[§7](IL_WORKFLOW_RUNBOOK.md#7-evaluate-closed-loop)); new machine: [setup](setup/README.md) first
2. **[Tasks and scene](epic3/02-tasks-and-scene.md)** — what was built in Isaac Lab (design)
3. **[Recording (LeRobot)](epic3/04-recording-lerobot.md)** — how demos become a v3 dataset (design)
4. **[Training](epic3/05-training.md)** / **[Evaluation](epic3/06-evaluation.md)** — policies and metrics (design)
5. **[ACT Evaluation Report](ACT_EVAL_REPORT_100K.md)** — reporting results
6. **[Epic 4](EPIC4_VR_INTEGRATION.md)** — Quest / ALVR design + VR one-time setup

Full timeline and overview: [docs index — Epic 3](README.md#epic-3-simulation-training-pipeline). This project’s reporting demos: [runbook project example reference](IL_WORKFLOW_RUNBOOK.md) (VR, `--record_arm right`).

## Pages (BookStack-ready)

| Page | Contents |
|------|----------|
| [Glossary](epic3/01-glossary.md) | Abbreviations and terms (incl. policy sidecar) |
| [Tasks and scene](epic3/02-tasks-and-scene.md) | Registration, Reach/Record configs, scene (install → [setup](setup/isaac-and-environments.md)) |
| [Teleoperation](epic3/03-teleoperation.md) | Keyboard/gamepad control model and keys; VR summary |
| [Recording (LeRobot)](epic3/04-recording-lerobot.md) | Pipeline, action labels, Dataset v3.0 on disk |
| [Training](epic3/05-training.md) | ACT / Pi0 jobs and hyperparameters |
| [Evaluation](epic3/06-evaluation.md) | How eval works, success criteria, metrics |
| [Findings and troubleshooting](epic3/07-findings-troubleshooting.md) | Issues addressed, limitations, and fixes |
| [Future work](epic3/08-future-work.md) | Planned follow-ups |

## Continue reading

- [Docs index](README.md) (full timeline / overview)
- [Setup hub](setup/README.md) (new machine)
- [IL Workflow Runbook](IL_WORKFLOW_RUNBOOK.md)
- [Epic 4 hub](EPIC4_VR_INTEGRATION.md)
