# AGENTS

## Project Overview
- **Goal:** Train a GTA V driving policy mapping screen frames to binary controls `W/A/S/D`.
- **Pipeline:** screen capture -> frozen `DINOv3 ViT-B16` features -> `Linear(768->128)` projection -> transformer head -> 4 binary logits.
- **Current focus:** optimize the transformer head on precomputed features before full end-to-end tuning.
- **Proxy scope:** these runs are trend-finding proxy experiments; projection is frozen during this phase.
- **Augmentations:** precomputed features already include 10 versions per frame (1 original + 9 augmented); lack of augmentation is not the proxy limitation.

## Hardware & Environment
- **Baseline hardware:** RTX 5070 Ti (`Blackwell sm_120`, 16GB VRAM) + 64GB system RAM.
- **Platform:** Windows with multiprocessing `spawn` (not `fork`); DataLoader and memmap code must be spawn-safe and picklable.
- **Python execution:** always use `uv run python ...`.
- **Never use:** `python ...` or `.venv/Scripts/python.exe ...`.
- **Matplotlib backend:** use `Agg` for training/eval plotting paths.

## FP8
- **Preference:** FP8 is preferred on `sm_120`; BF16 is fallback.
- **Source of truth:** `fp8_utils.py`.
- **Required wiring in `feature_training.py`:** import FP8 helpers, call `add_fp8_cli_args(parser)`, call `maybe_enable_fp8(model, ...)` after model construction.

## File Ownership
- **Disposable (safe to rewrite):** `feature_training.py`, `fp8_utils.py`, `run_experiments.py`, `test_*.py`.
- **Established (do not modify):** `model.py`, `training_new.py`, `dataloader.py`, `config.py`, `main.py`, `muon.py`.

## Architecture Constraints (Hard)
- **Head shape:** `HeadOnlyModel` consumes `embed_dim=128` token features, optional `Linear(128->model_dim)` downprojection, `EfficientTransformer`, then 4 binary output logits.
- **Class weighting:** keep hardcoded base `pos_weights` exactly `[1.0, 11.285, 25.556/2, 11.392]` for `[W, A, S, D]`.
- **Weight tuning:** only multiplicative scaling via `--a_pos_weight_scale`, `--s_pos_weight_scale`, `--d_pos_weight_scale`.
- **Optimizer split:** `Muon` for transformer parameters with `ndim >= 2`, `AdamW` for everything else; do not remove or replace `Muon`.

## Data Pipeline (Critical)
- **Dataset format:** 10 precomputed feature files (~22GB each, 1 original + 9 augmented) plus `labels.npy`, all loaded via memmap.
- **Spawn-safe memmaps:** constructor stores only paths/shapes/metadata, `__getstate__` drops memmap handles, `_ensure_memmaps()` lazily opens per worker.
- **Sampling:** each training sample randomly picks augmentation version `0-9`.
- **Shuffling:** global shuffle via `randperm` + `shuffle_chunk_size`; do not rely on `DataLoader(shuffle=True)`.
- **Batch reads:** use `__getitems__`; group by augmentation version and sort frame indices for near-sequential memmap access.
- **Traversal behavior:** full dataset coverage before repeats (epoch-like behavior across all 10 versions).
- **Loader/perf:** wrap DataLoader with `AsyncBatchPrefetcher`; keep `num_workers=4`; OS page-cache growth into RAM is expected, not a leak.

## Training Loop
- **Compilation:** use `torch.compile(model)` for throughput.
- **LR schedule:** custom cosine with warmup via `get_lr()`.
- **Invariant:** `lr_decay_iters` must always equal `max_iters`.
- **Warm restarts:** `lr_restart_cycles` controls SGDR-style cosine restarts (`1` = standard cosine).
- **AMP scaler:** enable `GradScaler` only for `float16`, never for `bfloat16`.

## Logging
- **Run config line must include:** `flat_tokens`, `embed_dim`, `model_dim` (if set), `device`, `dtype`, FP8 status, `label_smoothing`.
- **Logs directory:** `temp/experiment_logs/`.
- **Checkpoints directory:** `models/feature_head/`.
- **Plots:** save training/validation plots alongside checkpoints.

## Experiment Philosophy
- Optimize for balanced control quality across `W/A/S/D`; do not over-index on `S` recall alone.
- With label smoothing, loss floors rise by design; compare runs by generalization behavior, not raw minimum loss.
- Do not spend time on threshold sweeps at this stage.
- Prioritize high-signal experiments and keep negative results.
- Require mechanism-level reasoning: explain why a change helped or hurt.

## Non-Negotiable Guardrails
- Preserve spawn-safe lazy memmap semantics.
- Preserve `Muon` + `AdamW` split by parameter dimensionality.
- Preserve base class weights exactly; allow multiplicative scaling only.
- Keep `lr_decay_iters = max_iters` in every run/script.
- Use `uv run python` for all execution.
- Keep logs/checkpoints/plots in canonical directories.

## Known Results (Mutable, Update As Experiments Progress)
- Best architecture so far: `64d/6L` (`exp34`), `val_acc ~44.3%`, `S recall ~64%`, with signs of under-training.
- Prior 10K runs (`exp41/exp42`) were misconfigured (`lr_decay_iters=5000`, `max_iters=10000`).
- `A/D` steering ceiling around `~80/80` has persisted across many settings.
- Failed interventions so far: focal loss, frame dropout, scheduled pos-weight, projection expansion to `256`, dropout `0.5`.
- With `num_workers=4`, training is compute-bound.
- Typical throughput for `64d/6L` + FP8 has been about `75-82 ms/iter`.
