# Future Work

Derived from open items in [Findings and Troubleshooting](07-findings-troubleshooting.md) (resolved issues under [Issues addressed](07-findings-troubleshooting.md#issues-addressed-during-development) / [Eval and policy sidecar](07-findings-troubleshooting.md#eval-and-policy-sidecar); locked eval early-stop behavior is not repeated here).

## Simulation and transfer

- [ ] **Sim-to-real on physical Mobile AI** — policies are sim-only today ([Current limitations](07-findings-troubleshooting.md#current-limitations))
- [ ] **Close the visual / domain gap** — lighting, materials/textures, and camera appearance closer to RealSense / lab conditions before hardware transfer
- [ ] **Mobile AI RL / PPO tasks** — unlike stock WXAI reach, lift, and cabinet; no Mobile AI RL envs yet
- [ ] **Multiple object types in the Reach scene** — extend the discrete `EventCfg` pattern beyond cube color (e.g. swap `CuboidCfg` for a small asset set / object-type choice per env), reusing the same `env_ids`-aware reset style as `randomize_cube_color_discrete` ([Simulation Scene](02-tasks-and-scene.md#reset-randomization-eventcfg))

## Teleoperation and demos

- [ ] **Keyboard / gamepad teleop fine-tuning** — gains and smoothing for smoother operator motion ([Teleoperation](03-teleoperation.md); VR tracking work lives in [Epic 4 future work](../epic4/06-future-work.md))
- [ ] **Broader demonstration coverage** — production train set is VR right-arm only; expand modalities / arms once collection quality allows ([Recording](04-recording-lerobot.md), [Epic 4 — unused-arm drift](../epic4/05-findings-troubleshooting.md#unused-arm-drift-and-record_arm-right))

## Training and evaluation

- [ ] **Unblock Pi0 closed-loop sim eval** — Inductor/Triton AUTOTUNE exceeds the 120 s sidecar timeout on first step ([Eval and policy sidecar](07-findings-troubleshooting.md#eval-and-policy-sidecar) · [Evaluation](06-evaluation.md) / [§7 Evaluate](../IL_WORKFLOW_RUNBOOK.md#7-evaluate-closed-loop))
- [ ] **Optional eval LeRobot datasets** — closed-loop rollout is metrics-only today; optional `eval_*` dataset writing like some real-robot flows
- [ ] **Richer task success metrics** — Reach recording is an IL sandbox without automated success; broaden manipulation goals / metrics beyond pick–lift–place ([Evaluation](06-evaluation.md))
- [ ] **Improve ACT approach / grasp success** — reporting failures are mostly `no_progress` (never lift); see [ACT Evaluation Report](../ACT_EVAL_REPORT_100K.md)

## Continue reading

- [Findings and Troubleshooting](07-findings-troubleshooting.md)
- [§7 Evaluate](../IL_WORKFLOW_RUNBOOK.md#7-evaluate-closed-loop) · [ACT Evaluation Report](../ACT_EVAL_REPORT_100K.md)
- [Epic 4 — Future work](../epic4/06-future-work.md)
- [Epic 3 hub](../EPIC3_SIMULATION_TRAINING_PIPELINE.md)
