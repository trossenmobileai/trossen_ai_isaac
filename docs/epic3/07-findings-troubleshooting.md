# Findings and Troubleshooting

## Findings and Limitations
### Arm Drift (Resolved)

With early **IK-Rel** control, both arms drifted slowly even when sending zero actions. The switch to IK-Abs is documented in [Tasks and scene](02-tasks-and-scene.md#custom-reach-task-environment).

**Resolution:** Switching to **IK-Abs** fixed the problem. All current teleoperation and recording uses IK-Abs. This issue does not apply to the current pipeline.

### Issues Addressed During Development

- **Base instability:** robot base moved unexpectedly. Addressed with `fix_root_link=True` in [`reach_env_cfg.py`](02-tasks-and-scene.md#reach_env_cfgpy-scene-and-mdp-base)
- **Arm responsiveness:** arms were too fast or unstable. PD gains tuned in [`MOBILE_AI_HIGH_PD_CFG`](02-tasks-and-scene.md#mobile-ai-robot-registration)
- **Blank camera recordings:** camera prims must reference USD `Camera_*` nodes. See [`record_env_cfg.py`](02-tasks-and-scene.md#record_env_cfgpy-il-recording)
- **USD-authored workspace abandoned:** editing `mobile_ai.usd`, referencing a separate env USD, and GUI/payload cleanup all failed; table/cube are procedural in `MobileAIReachSceneCfg` — [Why the scene is procedural](02-tasks-and-scene.md#why-the-scene-is-procedural-not-a-usd-file)

### Scene and randomization

Cube pose/color must vary across demos so policies do not overfit ([Simulation Scene](02-tasks-and-scene.md#simulation-scene)). Two implementation details that bit early development:

- **Discrete color needs a custom event.** Built-in `mdp.randomize_visual_color` takes a continuous two-tuple `[low_rgb, high_rgb]`, not a palette. Passing three RGB colors (red/green/blue) is read as an invalid range and crashes at launch with `ValueError: high - low < 0`. Production uses `randomize_cube_color_discrete` in [`reach_env_cfg.py`](02-tasks-and-scene.md#reset-randomization-eventcfg).
- **`replicate_physics=False`.** With the default `True`, parallel envs can silently share one physics/visual representation so every cube gets the same randomized pose/color. The Reach scene sets `replicate_physics=False` on `MobileAIReachSceneCfg`.

### Current Limitations

- **No sim-to-real yet:** policies are trained and evaluated in simulation only; deployment on the physical Mobile AI is future work ([Future work](08-future-work.md))
- **Sim visual / domain gap:** lighting, materials/textures, and camera appearance are simplified and not near real-world RealSense / lab lighting; expect a transfer gap if moving to hardware
- **Teleoperation fine-tuning still needed:** keyboard/gamepad and VR control (gains, smoothing, hand-anchor / tracking) still need operator-facing tuning for smoother motion and more reliable tracking ([Teleoperation](03-teleoperation.md), [Epic 4 findings](../epic4/05-findings-troubleshooting.md))
- **No Mobile AI RL/PPO** unlike stock WXAI reach, lift, and cabinet tasks
- **Full training is external to Isaac:** policies train in `lerobot_train` via wrappers ([Training](05-training.md), [§6 Train](../IL_WORKFLOW_RUNBOOK.md#6-train)); in-repo wrappers today are ACT and Pi0 — other LeRobot Dataset v3.0 policies need their own wrapper. Only a short ACT smoke lives as an in-repo Python helper
- **Production demos are VR right-arm only:** keyboard/gamepad recording works but was not used for the reporting train set; unused-arm VR tracking remains a limitation for bimanual collection ([Recording](04-recording-lerobot.md))
- **Sim eval is metrics-only:** closed-loop rollout reports success metrics; it does not write an `eval_*` LeRobot dataset like optional real-robot recording
- **Sim eval early-stop (locked):** idle `no_progress` (500), approach hard-cap `no_pick` (1000), place window `no_place` (500 after lift), success tail (60); ACT and Pi0 share this path with separate `outputs/eval/act` vs `outputs/eval/pi0` — see [Evaluation](06-evaluation.md)
- **Pi0 sim eval blocked:** checkpoint trained, but first-step Inductor/Triton AUTOTUNE exceeded the 120 s sidecar client timeout; deferred — reporting uses the ACT 100k 30-episode eval ([ACT Evaluation Report](../ACT_EVAL_REPORT_100K.md), [Evaluation](06-evaluation.md) / [§7 Evaluate](../IL_WORKFLOW_RUNBOOK.md#7-evaluate-closed-loop))
- **Reach task has no automated success metrics:** the Reach *recording* scene is an IL sandbox; the separate *Lift* joint-position env used for ACT rollout does have lift/place metrics

### ACT evaluation results

See the [ACT Evaluation Report](../ACT_EVAL_REPORT_100K.md) for the reporting ACT 100k / 30-episode run.

## Troubleshooting
### IL-specific issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Blank or black camera videos | Wrong camera prim paths | Ensure Record env uses `Camera_high`, `Camera_follower_left`, `Camera_follower_right` |
| `ImportError: lerobot` during recording | LeRobot not in Isaac Sim Python | Install via `isaaclab.sh -p -m pip install lerobot==0.4.4` |
| Verify script fails | Wrong Python interpreter | Use `~/lerobot_trossen/.venv/bin/python` |
| Dataset incomplete after Ctrl+C | Interrupt before finalize | Wait for "dataset finalized" log; script handles SIGINT |
| `SRE module mismatch` in sidecar | Sidecar inherited Isaac `PYTHONPATH` | Fixed: sidecar subprocess uses clean env ([Evaluation](06-evaluation.md)) |
| `BrokenPipeError` / connection reset during eval | Probe connection closed sidecar session | Fixed: single persistent TCP connect in `act_rollout.py` |
| `PreTrainedPolicy` abstract class error | Wrong policy loader in sidecar | Use current `policy_sidecar.py` (`PreTrainedConfig` + `get_policy_class`) |
| Eval policy moves wrong arm | Checkpoint not 7D right-arm | Record/retrain with `--record_arm right` |
| Visual success but `success=False` | Gripper closed while cube in height band | Return requires `cube_is_placed` (open gripper + on-table); see [Evaluation](06-evaluation.md) |
| `success=True` but cube still gripped | Old height-only return detection | Fixed: release requires open gripper |
| Eval runs full timeout on failure | No failure early-stop | Fixed: idle / approach / place caps (`IDLE_STEPS`, `MAX_APPROACH_STEPS`, `MAX_STEPS_AFTER_LIFT`) |
| `success=True` at ~60 steps during approach | Lift/on-table threshold overlap | Fixed: clear lift requires `z > 0.845 m` before return counts |
| Gripper closed at start of next episode | Joint targets carried over after reset | Fixed: eval forces home pose + open grippers after every `env.reset()` |
| Pi0 eval `[FAIL] timed out` on first step; Triton `AUTOTUNE` spam | First Pi0 `select_action` compiles via Torch Inductor (>120 s client timeout) | Deferred: raise timeout / warmup compile / disable compile; see [Evaluation](06-evaluation.md) / [§7 Evaluate](../IL_WORKFLOW_RUNBOOK.md#7-evaluate-closed-loop) |

### Simulation and physics issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Robot base moves or tips | Root link not fixed | Confirm `fix_root_link=True` in [`reach_env_cfg.py`](02-tasks-and-scene.md#reach_env_cfgpy-scene-and-mdp-base) |
| `RuntimeError: Accessed invalid null prim` on Play/Stop | Prim selected in stage (UI bug) | Deselect all prims in Isaac Sim before Play/Stop (no physics impact) |
| `ValueError: high - low < 0` at launch | 3+ colors passed to `mdp.randomize_visual_color` | Use `randomize_cube_color_discrete` for a discrete palette ([Scene and randomization](#scene-and-randomization)) |
| Every parallel env’s cube has the same pose/color | `replicate_physics` left `True` | Set `replicate_physics=False` on `MobileAIReachSceneCfg` ([Simulation Scene](02-tasks-and-scene.md#scene-assets-mobileaireachscenecfg)) |
| Cube spawns off the table | `pose_range` entered as absolute world coords | Use deltas from cube `init_state.pos` ([EventCfg](02-tasks-and-scene.md#reset-randomization-eventcfg)) |
| Only 2–3 positions/colors in a short teleop test | Few `env.reset()` calls (waiting on 12 s timeout) | Press **J** to force resets ([Controls](../IL_WORKFLOW_RUNBOOK.md#controls-quick-reference)) |
| `AttributeError` on `write_root_pose_to_sim` (or similar) at reset | Cube still `AssetBaseCfg` | Declare cube as `RigidObjectCfg` ([Scene assets](02-tasks-and-scene.md#scene-assets-mobileaireachscenecfg)) |

> IK-Rel arm drift workarounds and ROS2 standalone scene checks from early documentation are resolved or deprecated.

## Continue reading

- [§7 Evaluate](../IL_WORKFLOW_RUNBOOK.md#7-evaluate-closed-loop) · [ACT Evaluation Report](../ACT_EVAL_REPORT_100K.md)
- [Future work](08-future-work.md)
- [Epic 3 hub](../EPIC3_SIMULATION_TRAINING_PIPELINE.md)

