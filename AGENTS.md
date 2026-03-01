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
- **LR schedule:** `lr_decay_iters` controls when cosine reaches `min_lr`. Can equal `max_iters` (full cosine) or be shorter (fast cosine — model trains at `min_lr` for the tail).
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

## Decision Rule
- Prioritize experiments that can falsify current hypotheses, not only confirm them.

## Non-Negotiable Guardrails
- Preserve spawn-safe lazy memmap semantics.
- Preserve `Muon` + `AdamW` split by parameter dimensionality.
- Preserve base class weights exactly; allow multiplicative scaling only.
- Keep `lr_decay_iters = max_iters` in every run/script.
- **Exception:** fast cosine schedules intentionally use `lr_decay_iters < max_iters`.
- Use `uv run python` for all execution.
- Keep logs/checkpoints/plots in canonical directories.

## Current Hypotheses (Not Gospel)
- These are working hypotheses from recent proxy runs and can be overturned by new seeds, longer training, or broader sweeps.
- `64d/6L` (`exp34`) currently appears competitive (`val_acc ~44.3%`, `S recall ~64%`) and may still be under-trained.
- Prior 10K runs (`exp41/exp42`) were misconfigured (`lr_decay_iters=5000`, `max_iters=10000`) and are lower-confidence evidence.
- An `A/D` steering ceiling around `~80/80` currently appears common across many tested settings, pending more seeds.
- Focal loss, frame dropout, scheduled pos-weight, projection expansion to `256`, and dropout `0.5` have not shown consistent gains so far.
- With `num_workers=4`, training has typically been compute-bound on baseline hardware.
- Typical throughput for `64d/6L` + FP8 has been about `75-82 ms/iter` in recent runs.

### LR Schedule Trends (exp46-61)
- `S` recall often declines with full cosine at 15K in observed runs (for example, from ~50% at 5K to ~17% at 15K), but this should be treated as provisional.
- Fast cosine (`lr_decay_iters` << `max_iters`) currently appears likely to preserve `S` better by reaching `min_lr` earlier and training longer at low LR.
- Best `S` preservation so far appears with short cosine (`lr_decay_iters=3000`) or fast cosine + `label_smoothing=0.01`, pending more seeds.
- Best `val_acc` observed so far is with fast cosine (`lr_decay_iters=5000`, `max_iters=15K-20K`) around ~48-50%, often with lower `S` (~33%).
- A Pareto-like tradeoff (higher `val_acc` vs lower `S` recall) currently appears likely; active cosine phase length seems to move runs along this frontier.
- Label smoothing alone (with full cosine) has usually looked worse for `S`; it has looked more helpful when paired with fast/short cosine.
- Warm restarts (SGDR) have not reliably prevented `S` decline in tested settings.
- Seed variance appears large: `S` recall at 5K has ranged 45-78%; single-seed gaps under ~15 points are likely noise.
- Eval fidelity matters: `eval_iters=12` looked noisier than `eval_iters=50`; default to `eval_iters >= 50` unless intentionally trading fidelity for speed.

### Regularization Trends (exp62-73)
- In the fast cosine regime, dropout `0.2` currently appears stronger than `0.35` (`exp66` vs `exp56`), but this remains seed-sensitive.
- Dropout `0.1` has looked slightly weaker than `0.2` in current evidence.
- Lower weight decay (`0.03` vs `0.075`) alone did not consistently help (`exp65`), though combinations may still be worth targeted retests.
- Batch `512` (with 2x LR) currently appears to improve stability and slightly improve generalization in several runs.
- `128`-dim (no downprojection) currently appears viable: `exp73` (incomplete at 13.5K) reached `A recall 90%`, `S recall 54%`, `D spec 78%`, with no obvious overfit yet.
- `96`-dim underperformed `64`-dim in `exp72` (`val_acc 45.3%`), but this comparison is not definitive.
- `8` layers did not clearly outperform `6` layers in current fast-cosine tests (`exp64` vs `exp66`).
- `128`-dim + `bs512` has run at ~`190ms/iter` (vs ~`80ms` for `64d+bs256`, ~`130ms` for `64d+bs512`) on baseline hardware.

### Current Candidate Configs
- `64d/6L + drop=0.2 + fast cosine + bs512` currently looks like a strong baseline (`val_acc ~50-51%`, `A ~80/83`, `D ~84/72`, `S ~44-48%`, `W spec ~69-73%`).
- `128-dim/6L + drop=0.2 + fast cosine` looks promising (`A recall 90%` observed once) and likely needs more bs512 + longer-run coverage.
- Suggested default starting point (not mandatory): `eval_iters=50`, `batch_size=512`, `learning_rate=1e-4`, `muon_lr=0.003`.
