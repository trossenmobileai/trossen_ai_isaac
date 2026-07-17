# ACT Evaluation Report — 100k, 30 Episodes

> Reporting result for closed-loop ACT policy quality in the Mobile AI digital twin.
> Architecture and metrics: [Evaluation](epic3/06-evaluation.md). How to run: [IL Workflow Cheat Sheet](IL_WORKFLOW_CHEATSHEET.md#5-evaluate-closed-loop).
> Training context: [Training](epic3/05-training.md).

## Summary

This report documents the closed-loop performance of the **ACT 100k** policy in simulation — the project’s reporting result for policy quality in the digital twin.

**Method**

| Item | Value |
|------|--------|
| Checkpoint | `~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2_100k/checkpoints/last/pretrained_model` |
| Dataset used for training | `mobile_ai_right_pick_place_20260714_v2` (VR, `--record_arm right`; same layout as eval obs/actions) |
| Eval script | [`run_play_act.sh`](../scripts/imitation_learning/run_play_act.sh) → [`play_act.py`](../scripts/imitation_learning/evaluation/play_act.py) → [`act_rollout.py`](../source/trossen_ai_isaac/trossen_ai_isaac/evaluation/act_rollout.py) |
| Environment | `Isaac-Lift-Cube-MobileAI-Joint-Pos-Play-v0` |
| Episodes | 30 |
| Control rate | 60 FPS (matches recording) |
| Success definition | Clear lift (`z > 0.845 m`) then release on table (`cube_is_placed`) |
| Early stop | See [Evaluation — metrics](epic3/06-evaluation.md#metrics) |
| Result artifact | `~/trossen_ai_isaac/outputs/eval/act/rollout_summary_30eps.json` |

Cube color (red / green / blue) is randomized per episode by the play environment; the summary breaks success down by color.

**Headline results**

| Metric | Value |
|--------|--------|
| Episodes | 30 |
| Successes | 17 |
| **Success rate** | **56.7% (17/30)** |
| Failures | 13 (all `stop_reason=no_progress`) |
| Lifted then placed | 17/17 of episodes that achieved a clear lift |
| Avg steps (success) | ~481 |
| Avg steps (failure) | ~626 |

**Success rate by cube color**

| Color | Episodes | Successes | Success rate |
|-------|----------|-----------|--------------|
| Red | 8 | 5 | 62.5% |
| Green | 9 | 5 | 55.6% |
| Blue | 13 | 7 | 53.8% |

**Failure modes**

All 13 failures ended with `no_progress`: the cube never left the on-table height band long enough to count as a clear lift (idle on the table while the arm moved without a completed pick). There were **no** `no_place` failures in this run — whenever the policy lifted the cube, it also completed the place/release criteria within the episode budget.

**Interpretation**

- Mid-50s success on a randomized 30-episode sample shows the 100k ACT policy can complete pick–lift–place in sim, but reliability is not yet near perfect.
- Performance is similar across cube colors (roughly 54–63%), so color randomization is not the dominant failure driver.
- The main gap is **approach / grasp** (never lifting), not placing after a successful lift.

Reproduce or extend the run with the [cheat sheet evaluate section](IL_WORKFLOW_CHEATSHEET.md#5-evaluate-closed-loop). Architecture and metric field definitions: [Evaluation](epic3/06-evaluation.md).

## Related documentation

- [Docs index](README.md#epic-3--simulation-training-pipeline) (Epic 3 section)
- [Evaluation](epic3/06-evaluation.md)
- [IL Workflow Cheat Sheet](IL_WORKFLOW_CHEATSHEET.md)
- BookStack: [Epic 3 hub](EPIC3_SIMULATION_TRAINING_PIPELINE.md)

---

