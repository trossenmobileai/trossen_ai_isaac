# Future Work

Derived from open items in [Findings and Troubleshooting](05-findings-troubleshooting.md) ([Issues addressed / design decisions](05-findings-troubleshooting.md#issues-addressed--design-decisions) for resolved lessons). Operational troubleshooting fixes (setcap path, POV reset, etc.) stay in that page — only open limitations and design follow-ups are listed here.

## Demonstration collection

- [ ] **Reliable bimanual VR recording** — unused-arm tracking drifts when attention is on the active hand; production demos used `--record_arm right` only ([Unused-arm drift](05-findings-troubleshooting.md#unused-arm-drift-and-record_arm-right) · [Current limitations](05-findings-troubleshooting.md#current-limitations))
- [ ] **Larger-scale VR demonstration sets** — beyond the current reporting right-arm train set ([§3 Collect](../IL_WORKFLOW_RUNBOOK.md#3-collect-demos-vr))

## Teleoperation quality

- [ ] **VR tracking / mapping fine-tuning** — hand-anchored mapping, `--pose_smoothing`, unused-arm drift, and session ergonomics for smoother demos ([Hand-anchored vs absolute](05-findings-troubleshooting.md#hand-anchored-vs-absolute-anchor-mode), [VR teleoperation](03-vr-teleoperation.md), [§1 session](../IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time))

## Stack and operations

- [ ] **Reduce VR setup / session friction** — headset + ALVR + SteamVR + per-session order is still heavy vs keyboard/gamepad smoke ([SteamVR from ALVR](05-findings-troubleshooting.md#steamvr-must-launch-from-alvr) · [Two-operator sessions](05-findings-troubleshooting.md#two-operator-vr-sessions) · [Input device comparison](05-findings-troubleshooting.md#input-device-comparison))
- [ ] **More robust networking for ALVR** — dedicated 5 GHz path or alternate routing where institutional Wi-Fi blocks peer-to-peer ([Current limitations](05-findings-troubleshooting.md#current-limitations))
- [ ] **Evaluate alternate XR streaming** — e.g. NVIDIA CloudXR vs local ALVR ([ALVR vs cloud XR](05-findings-troubleshooting.md#alvr-vs-cloud-xr-streaming))

Policy closed-loop eval (ACT report, Pi0 unblock) is tracked under [Epic 3 future work](../epic3/08-future-work.md) / [Evaluation](../epic3/06-evaluation.md) — trained on this VR-collected set, but not an Epic 4 deliverable.

## Continue reading

- [Findings and Troubleshooting](05-findings-troubleshooting.md)
- [§3 Collect VR](../IL_WORKFLOW_RUNBOOK.md#3-collect-demos-vr) · [§2 Practice](../IL_WORKFLOW_RUNBOOK.md#2-practice-vr-teleop-no-dataset)
- [Epic 3 — Future work](../epic3/08-future-work.md)
- [Epic 4 design index](README.md)
