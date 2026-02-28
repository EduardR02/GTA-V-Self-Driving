# Precompute + Head Fine-Tune Workflow

Use this runbook with `uv` + local Windows venv only.

## 1) Dry-run precompute (fast sanity check)

```powershell
uv run ".venv/Scripts/python.exe" precompute_features.py --checkpoint_path "<FULL_CKPT>" --output_dir "temp/precomputed_dryrun" --dry_run --dry_run_images_per_file 64
```

## 2) Full precompute

```powershell
uv run ".venv/Scripts/python.exe" precompute_features.py --checkpoint_path "<FULL_CKPT>" --output_dir "temp/precomputed_full"
```

Optional custom data roots:

```powershell
uv run ".venv/Scripts/python.exe" precompute_features.py --checkpoint_path "<FULL_CKPT>" --output_dir "temp/precomputed_full" --data_dirs "data/dir_a" "data/dir_b"
```

## 3) Head-only training on precomputed features

```powershell
uv run ".venv/Scripts/python.exe" feature_training.py --feature_dir "temp/precomputed_full" --checkpoint_path "<FULL_CKPT>" --out_dir "models/feature_head" --save_name "feature_head.pt"
```

This outputs a head-only checkpoint with `cls_token`, `transformer.*`, and `fc_head.*`.

## 4) Merge head checkpoint back into full checkpoint

```powershell
uv run ".venv/Scripts/python.exe" merge_head_checkpoint.py --full_ckpt "<FULL_CKPT>" --head_ckpt "models/feature_head/feature_head.pt" --out_ckpt "models/feature_head/full_merged.pt"
```

Strict mode is enabled by default. To intentionally allow partial merges:

```powershell
uv run ".venv/Scripts/python.exe" merge_head_checkpoint.py --full_ckpt "<FULL_CKPT>" --head_ckpt "models/feature_head/feature_head.pt" --out_ckpt "models/feature_head/full_merged.pt" --no_strict
```

## 5) Quick inference note

Use the merged full checkpoint anywhere a normal full-model checkpoint is consumed (precompute, eval, runtime driving). In this repo, update checkpoint config consumed by `training_new.py` / `main.py` to point at the merged file.

## Output-dir immutability and resume safety

- Treat each `--output_dir` as immutable for a specific dataset/config signature.
- `precompute_features.py` validates `metadata.json`; mismatched config aborts instead of mixing incompatible outputs.
- Existing `labels.npy` must match labels rebuilt from source `.h5` files.
- Resume markers in `.progress/*.done` are reused only when metadata is compatible.
- Use `--force_reset` only when you intentionally need to rebuild incompatible memmaps.
