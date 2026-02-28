# AGENTS

## Project Overview
- **Goal:** Train a GTA V driving policy that maps screen frames to 4 binary controls: **W/A/S/D**.
- **Current pipeline:** Screen capture -> frozen **DINOv3 ViT-B16** features -> **Linear(768->128)** projection -> transformer head -> 4-way binary logits.
- **Current focus:** Optimize the **transformer head on precomputed features** before full end-to-end pipeline tuning.
- **Experiment intent:** These are **proxy experiments** (frozen projection, no real augmentations). Use them to detect **trends**, not to claim final production metrics.

## Hardware & Environment
- **Hardware baseline:** RTX 5070 Ti (**Blackwell sm_120**, 16GB VRAM) + 64GB system RAM.
- **OS/runtime:** Windows. Multiprocessing uses **spawn** (not fork). All DataLoader/memmap code must be spawn-safe and picklable.
- **Python execution rule:** Always run Python as `uv run python ...`.
- **Never use:** `python ...` or `.venv/Scripts/python.exe ...`.
- **Matplotlib rule:** Use **Agg** backend for any plotting in training/eval paths; Tk backends can crash from worker-thread contexts.

## FP8
- **Preferred precision:** FP8 is preferred on this hardware and is expected to work on sm_120.
- **Source of truth:** `fp8_utils.py`.
- **Required integration points in `feature_training.py`:**
  - Import FP8 helpers from `fp8_utils.py`.
  - Call `add_fp8_cli_args(parser)` in arg parser creation.
  - Call `maybe_enable_fp8(model, ...)` **after model construction**.
- **Fallback:** BF16 is valid and supported, but FP8 should be the default preference when available.

## File Ownership
- **Disposable (safe to rewrite):** `feature_training.py`, `fp8_utils.py`, `run_experiments.py`, `test_*.py`.
- **Established (do not modify):** `model.py`, `training_new.py`, `dataloader.py`, `config.py`, `main.py`, `muon.py`.

## Architecture (`feature_training.py`)
- **Model shape:** `HeadOnlyModel` consumes 128-d token features.
- **Optional downprojection:** `Linear(128->model_dim)` when `model_dim` is set.
- **Head:** `EfficientTransformer` followed by 4 binary output logits.
- **Best known config:** `--model_dim 64 --num_layers 6 --dropout 0.35` (exp34 baseline).
- **Class weighting (hard constraint):**
  - Base `pos_weights` are hardcoded and must remain: `[1.0, 11.285, 25.556/2, 11.392]` for `[W, A, S, D]`.
  - Allowed tuning is multiplicative scaling only via CLI: `--a_pos_weight_scale`, `--s_pos_weight_scale`, `--d_pos_weight_scale`.
  - Do not change base constants.
- **Optimizer policy (hard constraint):**
  - **Muon** for transformer parameters with `ndim >= 2`.
  - **AdamW** for everything else.
  - Do not remove or replace Muon.

## Data Pipeline (Critical)
- **Storage format:** 10 precomputed feature files (~22GB each) + `labels.npy`, loaded as memory-mapped arrays.
- **Windows spawn compatibility requirements:**
  - Dataset constructor stores only paths/shapes/metadata.
  - Do not create live memmap objects in constructor.
  - `__getstate__` must drop memmap handles before pickling.
  - Memmaps must be opened lazily per worker via `_ensure_memmaps()` (or equivalent).
- **Sampling behavior requirements:**
  - Each sample randomly selects one augmentation version (0-9) during training.
  - Global shuffling uses `randperm` with configurable `shuffle_chunk_size`.
  - Use batched reads via `__getitems__`.
  - In `__getitems__`, group by augmentation version and sort frame indices for near-sequential memmap access.
- **Loader/perf requirements:**
  - Wrap DataLoader with `AsyncBatchPrefetcher` for asynchronous GPU transfer.
  - Keep `num_workers=4` for throughput; `num_workers=0` is typically 3-4x slower.
  - 64GB RAM page-cache growth from memmaps is expected OS cache behavior, not a leak.

## Training Loop
- **Compile:** Use `torch.compile(model)` for meaningful speedup.
- **LR schedule:** Custom cosine with warmup via `get_lr()`.
- **Critical schedule invariant:** `--lr_decay_iters` must equal `--max_iters`.
  - Default decay is 5000.
  - If `max_iters` changes, update `lr_decay_iters` to match or training will sit at `min_lr` for the tail.
- **Warm restarts:** `--lr_restart_cycles` controls SGDR-style cosine restarts (`1` means standard cosine).
- **AMP scaler rule:** Enable `GradScaler` only for `float16`; do not use it for `bfloat16`.
- **Effective batch:** Controlled by `--gradient_accumulation_steps`.

## Logging
- **Run config print line must include:** `flat_tokens`, `embed_dim`, `model_dim` (if set), `device`, `dtype`, `fp8 status`, `label_smoothing`.
- **Experiment logs directory:** `temp/experiment_logs/`.
- **Checkpoints directory:** `models/feature_head/`.
- **Plots:** Save training/validation metric plots next to checkpoints.

## Experiment Philosophy
- Optimize for **balanced control quality** across W/A/S/D; do not overfit to S recall alone.
- With label smoothing, absolute loss floor is higher by construction; compare LS runs by **generalization behavior**, not raw loss minima.
- Do not spend time on threshold sweeps in this stage; threshold tuning is for final models.
- Run high-signal experiments and keep negative results; they still reduce search space.
- Interpret results mechanistically: prioritize understanding **why** a change helped/hurt.

## Known Results Summary
- **Best architecture so far:** 64d / 6 layers (exp34), val_acc ~44.3%, S recall ~64%.
- **Training status:** exp34 appears under-trained (val_acc exceeds train_acc by ~5 points).
- **Prior 10K runs invalid:** exp41/exp42 were misconfigured (`max_iters=10000` but `lr_decay_iters=5000`), so half the run was at `min_lr`.
- **Persistent limit:** A/D steering remains around ~80/80 ceiling across many settings.
- **Failed interventions:** focal loss, frame dropout, scheduled pos-weight, projection expansion to 256.
- **Dropout finding:** 0.5 dropout underperforms baseline 0.35.
- **Performance profile:** training is compute-bound (not IO-bound) with `num_workers=4`.
- **Muon cost:** Muon is ~40% of step time and remains required.

## Non-Negotiable Regression Guardrails
- Preserve data pipeline spawn-safety and lazy memmap semantics.
- Keep Muon + AdamW split exactly by parameter dimensionality policy.
- Preserve hardcoded base class weights and only tune via multiplicative scale args.
- Always align `lr_decay_iters` with `max_iters` in new experiments and scripts.
- Keep logs/checkpoints/plots in the canonical directories above.
- Use `uv run python` for all scripts, tests, and experiment orchestration.
