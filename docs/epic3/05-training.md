# Training

After a verified [LeRobot Dataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) exists, this step **trains a policy** with `lerobot-train` in the `lerobot_train` conda env. Isaac Sim is **not** involved until evaluation ([Evaluation](06-evaluation.md)).

**How to train:** run a repo **wrapper `.sh`**. Open the script and set dataset paths, steps, job name, and policy flags for **your** needs — defaults are this project’s recipe, not a fixed requirement. Copy-paste: [§6 Train](../IL_WORKFLOW_RUNBOOK.md#6-train).

**Wrappers shipped today:** **ACT** ([`run_verify_and_train.sh`](../../scripts/imitation_learning/run_verify_and_train.sh)) and **Pi0** ([`run_train_pi0.sh`](../../scripts/imitation_learning/run_train_pi0.sh)). Both use `conda run --no-capture-output` for a live progress bar.

**Other policies:** any LeRobot-supported policy that consumes Dataset v3.0 can train on the same demos, but this repo does **not** ship wrappers beyond ACT/Pi0 — add a wrapper or call `lerobot-train` yourself.

## This project’s runs (ACT and Pi0)

This project trained **ACT** (twice) and **Pi0** (once) on the **VR-collected** right-arm pick-and-place set (`mobile_ai_right_pick_place_20260714_v2`), then evaluated the longer ACT run in simulation. The tables below document **those runs**, not every possible training configuration.

```mermaid
flowchart LR
  Disk["LeRobot Dataset v3.0 on disk"]
  Verify["verify_dataset.py"]
  Train["lerobot-train in lerobot_train conda"]
  Ckpt["outputs/train/.../checkpoints"]
  Eval["Closed-loop sim eval"]

  Disk --> Verify --> Train --> Ckpt --> Eval
```

**What a newcomer should know**

1. Recording writes a [LeRobot Dataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) (parquet frames + MP4 cameras). That format is the common input for LeRobot trainers.
2. **ACT** is a compact transformer that maps camera images + joint state → a chunk of joint actions. It trains from scratch on the demo set.
3. **Pi0** is a larger pretrained policy (`lerobot/pi0_base`) fine-tuned on the same demos. It also expects LeRobot features, so no dataset conversion is required.
4. Training runs in the external `lerobot_train` conda env (Python 3.12 / CUDA). Isaac Sim is not involved until evaluation.
5. Checkpoints land under `~/trossen_ai_isaac/outputs/train/<job_name>/checkpoints/`. Evaluation uses the `last` (or a numbered) `pretrained_model` folder.

**Shared dataset (this project’s ACT / Pi0 runs)**

| Field | Value |
|-------|--------|
| `repo_id` | `trossen-admin/mobile_ai_right_pick_place_20260714_v2` |
| `root` | `~/lerobot_trossen/datasets/mobile_ai_right_pick_place_20260714_v2` |
| Collection | **VR**, `--record_arm right` ([`run_collect_dataset.sh`](../../scripts/imitation_learning/run_collect_dataset.sh)); see [Recording](04-recording-lerobot.md) |
| Layout | 7D right-arm `observation.state` / `action`; cameras `cam_high` + `cam_right_wrist` (480×640) |
| `video_backend` | `pyav` |
| Image transforms | Disabled during these runs |

**Artifacts produced (this project)**

| Job | Policy | Steps | Output directory | Role |
|-----|--------|-------|------------------|------|
| `act_mobile_ai_right_v2` | ACT | 10 000 | `~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2` | Intermediate / smoke train (not used for the reporting eval) |
| `act_mobile_ai_right_v2_100k` | ACT | 100 000 | `~/trossen_ai_isaac/outputs/train/act_mobile_ai_right_v2_100k` | **Reporting model** — 30-episode closed-loop eval |
| `pi0_mobile_ai_right_v2` | Pi0 | 10 000 | `~/trossen_ai_isaac/outputs/train/pi0_mobile_ai_right_v2` | Fine-tuned Pi0; sim eval deferred ([Evaluation](06-evaluation.md)) |

**How the runs were launched** (examples — edit wrapper settings for your own jobs)

- **ACT (default wrapper recipe):** [`run_verify_and_train.sh`](../../scripts/imitation_learning/run_verify_and_train.sh) — verify, then `lerobot-train` with `--policy.type=act`. Editable: `REPO_ID`, `ROOT`, `OUTPUT_DIR`, `STEPS`. Live progress via `--no-capture-output`.
- **ACT longer run (this project’s reporting model):** same recipe with `--steps=100000`, separate `--output_dir` / `--job_name`, `--save_freq=10000` — copy-paste in [§6 Train](../IL_WORKFLOW_RUNBOOK.md#6-train) (no separate wrapper).
- **Pi0:** [`run_train_pi0.sh`](../../scripts/imitation_learning/run_train_pi0.sh) after [`run_verify_pi0_dataset.sh`](../../scripts/imitation_learning/run_verify_pi0_dataset.sh). Editable vars / flags near the top; live progress same as ACT.

**ACT hyperparameters** (identical for the 10k and 100k jobs except `steps` and `save_freq`)

| Setting | Value |
|---------|--------|
| `batch_size` | 8 |
| `num_workers` | 4 |
| `seed` | 1000 |
| Optimizer | AdamW, `lr=1e-5`, `weight_decay=1e-4`, `grad_clip_norm=10`, betas `(0.9, 0.999)` |
| Scheduler | None |
| `n_obs_steps` | 1 |
| `chunk_size` / `n_action_steps` | 100 |
| `dim_model` | 512 |
| Encoder / decoder layers | 4 / 1 |
| Attention heads | 8 |
| `dim_feedforward` | 3200 |
| `dropout` | 0.1 |
| `kl_weight` | 10.0 |
| Vision backbone | ResNet18 (`IMAGENET1K_V1`) |
| VAE | Enabled (`latent_dim=32`) |
| Device | `cuda` |
| Logging | `log_freq=100`; W&B off; no Hub push |
| Checkpointing | 10k run: `save_freq=1000`; 100k run: `save_freq=10000` |

**Pi0 hyperparameters** (from `run_train_pi0.sh` / saved `train_config.json`)

| Setting | Value |
|---------|--------|
| Base weights | `lerobot/pi0_base` |
| `batch_size` | 8 |
| `steps` | 10 000 |
| `num_workers` | 4 |
| `seed` | 1000 |
| Optimizer | AdamW, `lr=2.5e-5`, `weight_decay=0.01`, `grad_clip_norm=1`, betas `(0.9, 0.95)` |
| `n_obs_steps` | 1 |
| `chunk_size` / `n_action_steps` | 50 |
| `dtype` | `bfloat16` |
| `compile_model` | `true` |
| `gradient_checkpointing` | `true` |
| `train_expert_only` | `true` |
| `num_inference_steps` | 10 |
| Variants | PaliGemma `gemma_2b`, action expert `gemma_300m` |
| Device | `cuda` |
| Logging / save | `log_freq=100`, `save_freq=1000`; W&B off; no Hub push |

Configs for each finished run are stored next to the weights, e.g. `.../checkpoints/last/pretrained_model/train_config.json`.

**Evaluation choice:** The team evaluated **`act_mobile_ai_right_v2_100k`** (not the 10k ACT or Pi0) with a 30-episode closed-loop rollout. Procedure: [Evaluation](06-evaluation.md) / [§7 Evaluate](../IL_WORKFLOW_RUNBOOK.md#7-evaluate-closed-loop). Results: [ACT Evaluation Report](../ACT_EVAL_REPORT_100K.md).

---


---

## How to run

Copy-paste policy training (wrappers + framing): [§6 Train](../IL_WORKFLOW_RUNBOOK.md#6-train). Tables above record **this project’s** ACT/Pi0 runs (single source of truth for those hyperparameters). Open wrapper scripts and adjust settings near the top for your own jobs.

## Continue reading

- [§6 Train](../IL_WORKFLOW_RUNBOOK.md#6-train) · [§7 Evaluate](../IL_WORKFLOW_RUNBOOK.md#7-evaluate-closed-loop)
- [Evaluation](06-evaluation.md)
- [ACT Evaluation Report](../ACT_EVAL_REPORT_100K.md)
- [Epic 3 hub](../EPIC3_SIMULATION_TRAINING_PIPELINE.md)
