import argparse
import copy
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import torch
from tqdm import tqdm

import config
from dataloader import (
    RandomSkyMask,
    _compute_stuck_start_idx,
    color_jitter_replace,
    diagonal_warp,
    flip_image_with_minimap,
    transform,
    x1,
    x2,
    y1,
    y2,
)
from model import Dinov3ForTimeSeriesClassification


DEFAULT_DATA_DIRS = [config.stuck_data_dir_name, config.new_data_dir_name]
UNWANTED_PREFIX = "_orig_mod."
COMPATIBILITY_SIGNATURE_KEYS = (
    "num_versions",
    "version_info",
    "file_paths",
    "file_boundaries",
    "num_images",
    "num_tokens",
    "embed_dim",
    "label_shift",
    "sequence_len",
    "sequence_stride",
    "train_split",
    "cls_option",
)


@dataclass
class VersionSpec:
    idx: int
    aug_type: str
    warp_direction: str | None = None
    warp_shift_min: int | None = None
    warp_shift_max: int | None = None
    zoom_scale: float | None = None

    def as_dict(self) -> dict:
        payload = {"idx": self.idx, "type": self.aug_type}
        if self.warp_direction is not None:
            payload["warp_direction"] = self.warp_direction
            payload["warp_shift_range"] = [self.warp_shift_min, self.warp_shift_max]
        if self.zoom_scale is not None:
            payload["zoom_scale"] = self.zoom_scale
        return payload


def _parse_range(raw: str, cast_fn, name: str):
    pieces = [p.strip() for p in raw.split(",")]
    if len(pieces) != 2:
        raise ValueError(f"{name} must be formatted as 'min,max', got: {raw}")
    a, b = cast_fn(pieces[0]), cast_fn(pieces[1])
    if a > b:
        raise ValueError(f"{name} min cannot be greater than max: {raw}")
    return a, b


def _strip_compile_prefix(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith(UNWANTED_PREFIX):
            cleaned[key[len(UNWANTED_PREFIX):]] = value
        else:
            cleaned[key] = value
    return cleaned


def _to_wasd(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels[None, :]

    if labels.shape[-1] == 4:
        return labels.astype(np.float32, copy=False)

    out = np.zeros((labels.shape[0], 4), dtype=np.float32)
    w_idx = config.outputs.get("w", 0)
    a_idx = config.outputs.get("a", 1)
    s_idx = config.outputs.get("s", 2)
    d_idx = config.outputs.get("d", 3)
    wa_idx = config.outputs.get("wa", 4)
    wd_idx = config.outputs.get("wd", 5)

    out[:, 0] = labels[:, w_idx] | labels[:, wa_idx] | labels[:, wd_idx]
    out[:, 1] = labels[:, a_idx] | labels[:, wa_idx]
    out[:, 2] = labels[:, s_idx]
    out[:, 3] = labels[:, d_idx] | labels[:, wd_idx]
    return out.astype(np.float32, copy=False)


def _load_data_files(data_dirs: list[str]) -> list[str]:
    files = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        for name in sorted(os.listdir(data_dir)):
            if name.endswith(".h5"):
                files.append(os.path.join(data_dir, name))
    if not files:
        raise RuntimeError("No .h5 files found in data_dirs")
    return files


def _checkpoint_to_model_args(args, checkpoint: dict, state_dict: dict) -> dict:
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    layer_ids = []
    for key in state_dict.keys():
        if key.startswith("transformer.layers."):
            pieces = key.split(".")
            if len(pieces) > 2 and pieces[2].isdigit():
                layer_ids.append(int(pieces[2]))

    inferred_num_layers = (max(layer_ids) + 1) if layer_ids else None
    inferred_dim = None
    if "cls_token" in state_dict:
        inferred_dim = int(state_dict["cls_token"].shape[-1])
    elif "fc_head.weight" in state_dict:
        inferred_dim = int(state_dict["fc_head.weight"].shape[-1])

    return {
        "dino_size": cfg.get("dino_size", args.dino_size),
        "cls_option": cfg.get("cls_option", args.cls_option),
        "num_heads": cfg.get("num_heads") or args.num_heads,
        "num_layers": cfg.get("num_layers") or inferred_num_layers or args.num_layers,
        "transformer_dim": cfg.get("transformer_dim") or inferred_dim or args.transformer_dim,
        "dropout_rate": 0.0,
        "label_smoothing": 0.0,
    }


def _load_feature_extractor(args, device: torch.device, ptdtype: torch.dtype):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = _strip_compile_prefix(state_dict)

    model_args = _checkpoint_to_model_args(args, checkpoint, state_dict)
    model = Dinov3ForTimeSeriesClassification(
        model_args["dino_size"],
        num_classes=4,
        dropout_rate=model_args["dropout_rate"],
        dtype=ptdtype,
        cls_option=model_args["cls_option"],
        num_heads=model_args["num_heads"],
        num_layers=model_args["num_layers"],
        transformer_dim=model_args["transformer_dim"],
        label_smoothing=model_args["label_smoothing"],
    )

    load_msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint_path}")
    print(f"Missing keys: {len(load_msg.missing_keys)} | Unexpected keys: {len(load_msg.unexpected_keys)}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    model_cfg = {
        "dino_size": model_args["dino_size"],
        "cls_option": model.cls_option,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "transformer_dim": model.embed_dim,
    }
    return model, model_cfg


def _build_versions(args, rng: random.Random) -> list[VersionSpec]:
    remaining = args.num_versions - 1 - args.num_flipped - args.num_warped - args.num_zoomed
    if remaining < 0:
        raise ValueError(
            "Invalid version counts: num_versions must be >= 1 + num_flipped + num_warped + num_zoomed"
        )

    versions = [VersionSpec(idx=0, aug_type="unaltered")]
    idx = 1

    for _ in range(args.num_flipped):
        versions.append(VersionSpec(idx=idx, aug_type="flipped"))
        idx += 1

    for _ in range(args.num_warped):
        direction = "left" if rng.random() < 0.5 else "right"
        versions.append(
            VersionSpec(
                idx=idx,
                aug_type="warped",
                warp_direction=direction,
                warp_shift_min=args.warp_shift_range[0],
                warp_shift_max=args.warp_shift_range[1],
            )
        )
        idx += 1

    for _ in range(args.num_zoomed):
        scale = rng.uniform(args.zoom_range[0], args.zoom_range[1])
        versions.append(VersionSpec(idx=idx, aug_type="zoomed", zoom_scale=scale))
        idx += 1

    for _ in range(remaining):
        versions.append(VersionSpec(idx=idx, aug_type="photometric"))
        idx += 1

    if args.dry_run:
        if len(versions) < 2:
            raise ValueError("dry_run requires at least 2 versions to include one augmented version")
        versions = [versions[0], versions[1]]
        versions[1].idx = 1

    for i, version in enumerate(versions):
        version.idx = i
    return versions


def _extract_minimap(img: np.ndarray) -> np.ndarray:
    return np.copy(img[y1:y2, x1:x2])


def _apply_zoom_with_minimap(img: np.ndarray, scale: float) -> np.ndarray:
    zoom_transform = A.Affine(scale=scale, p=1.0)
    minimap = _extract_minimap(img)
    out = zoom_transform(image=img)["image"]
    out[y1:y2, x1:x2] = minimap
    return out


def _apply_photometric_with_minimap(img: np.ndarray, color_jitter: A.BasicTransform, sky_mask: RandomSkyMask) -> np.ndarray:
    minimap = _extract_minimap(img)
    out = color_jitter(image=img)["image"]
    out = sky_mask(image=out)["image"]
    out[y1:y2, x1:x2] = minimap
    return out


def _augment_image(img: np.ndarray, version: VersionSpec, color_jitter: A.BasicTransform, sky_mask: RandomSkyMask, rng: random.Random) -> np.ndarray:
    out = img
    if version.aug_type == "flipped":
        out = flip_image_with_minimap(out)
    elif version.aug_type == "warped":
        shift = rng.randint(version.warp_shift_min, version.warp_shift_max)
        out = diagonal_warp(out, shift=shift, direction=version.warp_direction)
    elif version.aug_type == "zoomed":
        out = _apply_zoom_with_minimap(out, scale=version.zoom_scale)

    if version.aug_type != "unaltered":
        out = _apply_photometric_with_minimap(out, color_jitter=color_jitter, sky_mask=sky_mask)

    return np.ascontiguousarray(out)


def _validate_or_create_memmap(path: Path, shape: tuple[int, ...], reset: bool) -> tuple[np.memmap, bool]:
    created_new = False
    expected_bytes = int(np.prod(shape) * np.dtype(np.float16).itemsize)
    if path.exists():
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            if not reset:
                raise RuntimeError(
                    f"Existing memmap has wrong size: {path} ({actual_bytes} != {expected_bytes}). "
                    "Delete it or run with --force_reset."
                )
            path.unlink()
            created_new = True

    mode = "r+" if path.exists() else "w+"
    if mode == "w+":
        created_new = True
    return np.memmap(path, mode=mode, dtype=np.float16, shape=shape), created_new


def _normalize_version_info(version_info: list[dict]) -> list[dict]:
    normalized = []
    for item in version_info:
        if "idx" not in item or "type" not in item:
            raise ValueError(f"Invalid version_info entry: {item}")

        norm = {
            "idx": int(item["idx"]),
            "type": str(item["type"]),
        }

        if item.get("warp_direction") is not None:
            norm["warp_direction"] = str(item["warp_direction"])

        warp_range = item.get("warp_shift_range")
        if warp_range is not None:
            if len(warp_range) != 2:
                raise ValueError(f"Invalid warp_shift_range in version_info: {item}")
            norm["warp_shift_range"] = [int(warp_range[0]), int(warp_range[1])]

        if item.get("zoom_scale") is not None:
            norm["zoom_scale"] = float(item["zoom_scale"])

        normalized.append(norm)

    return normalized


def _normalize_file_paths(file_paths: list[str]) -> list[str]:
    return [os.path.normpath(path) for path in file_paths]


def _build_compatibility_signature(
    *,
    versions: list[VersionSpec],
    file_paths: list[str],
    file_boundaries: list[int],
    num_images: int,
    num_tokens: int,
    embed_dim: int,
    label_shift: int,
    sequence_len: int,
    sequence_stride: int,
    train_split: float,
    cls_option: str,
) -> dict:
    version_info = [v.as_dict() for v in versions]
    return {
        "num_versions": int(len(versions)),
        "version_info": _normalize_version_info(version_info),
        "file_paths": _normalize_file_paths(file_paths),
        "file_boundaries": [int(x) for x in file_boundaries],
        "num_images": int(num_images),
        "num_tokens": int(num_tokens),
        "embed_dim": int(embed_dim),
        "label_shift": int(label_shift),
        "sequence_len": int(sequence_len),
        "sequence_stride": int(sequence_stride),
        "train_split": float(train_split),
        "cls_option": str(cls_option),
    }


def _compatibility_signature_from_metadata(metadata: dict) -> dict:
    if "compatibility_signature" in metadata:
        signature = metadata["compatibility_signature"]
    else:
        signature = metadata

    cls_option = signature.get("cls_option")
    if cls_option is None:
        cls_option = metadata.get("model", {}).get("cls_option")

    version_info = signature.get("version_info", metadata.get("version_info"))
    if version_info is None:
        version_info = []

    file_paths = signature.get("file_paths", metadata.get("file_paths"))
    if file_paths is None:
        file_paths = []

    file_boundaries = signature.get("file_boundaries", metadata.get("file_boundaries"))
    if file_boundaries is None:
        file_boundaries = []

    return {
        "num_versions": signature.get("num_versions", metadata.get("num_versions")),
        "version_info": _normalize_version_info(version_info) if version_info else [],
        "file_paths": _normalize_file_paths(file_paths),
        "file_boundaries": [int(x) for x in file_boundaries],
        "num_images": signature.get("num_images", metadata.get("num_images")),
        "num_tokens": signature.get("num_tokens", metadata.get("num_tokens")),
        "embed_dim": signature.get("embed_dim", metadata.get("embed_dim")),
        "label_shift": signature.get("label_shift", metadata.get("label_shift")),
        "sequence_len": signature.get("sequence_len", metadata.get("sequence_len")),
        "sequence_stride": signature.get("sequence_stride", metadata.get("sequence_stride")),
        "train_split": signature.get("train_split", metadata.get("train_split")),
        "cls_option": cls_option,
    }


def _describe_signature_difference(key: str, existing, current) -> str:
    if isinstance(existing, list) and isinstance(current, list):
        if len(existing) != len(current):
            return f"{key}: existing length={len(existing)} current length={len(current)}"

        for i, (left, right) in enumerate(zip(existing, current)):
            if left != right:
                return f"{key}: first mismatch at index {i}: existing={left!r} current={right!r}"

    return f"{key}: existing={existing!r} current={current!r}"


def _validate_existing_metadata(metadata_path: Path, current_signature: dict) -> dict | None:
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        existing_metadata = json.load(f)

    existing_signature = _compatibility_signature_from_metadata(existing_metadata)
    mismatch_details = []

    for key in COMPATIBILITY_SIGNATURE_KEYS:
        if existing_signature.get(key) != current_signature.get(key):
            mismatch_details.append(
                _describe_signature_difference(
                    key,
                    existing_signature.get(key),
                    current_signature.get(key),
                )
            )

    if mismatch_details:
        detail = "\n".join(f"  - {item}" for item in mismatch_details)
        raise RuntimeError(
            f"Existing metadata is incompatible with current run: {metadata_path}\n"
            "Refusing to proceed to prevent feature/label/metadata mismatch.\n"
            f"Mismatches:\n{detail}\n"
            "Use a new --output_dir or delete existing artifacts in this output directory."
        )

    return existing_metadata


def _versions_from_metadata(metadata: dict) -> list[VersionSpec]:
    versions = []
    for item in metadata["version_info"]:
        warp_range = item.get("warp_shift_range")
        versions.append(
            VersionSpec(
                idx=int(item["idx"]),
                aug_type=str(item["type"]),
                warp_direction=item.get("warp_direction"),
                warp_shift_min=int(warp_range[0]) if warp_range is not None else None,
                warp_shift_max=int(warp_range[1]) if warp_range is not None else None,
                zoom_scale=float(item["zoom_scale"]) if item.get("zoom_scale") is not None else None,
            )
        )
    return versions


def _build_metadata(
    args,
    versions: list[VersionSpec],
    file_paths: list[str],
    file_boundaries: list[int],
    stuck_offsets: list[int],
    num_images: int,
    num_tokens: int,
    embed_dim: int,
    model_cfg: dict,
    compatibility_signature: dict,
):
    return {
        "num_versions": len(versions),
        "num_images": num_images,
        "num_tokens": num_tokens,
        "embed_dim": embed_dim,
        "version_info": [v.as_dict() for v in versions],
        "file_boundaries": file_boundaries,
        "file_paths": file_paths,
        "stuck_offsets": stuck_offsets,
        "label_shift": args.label_shift,
        "cls_option": model_cfg["cls_option"],
        "train_split": args.train_split,
        "sequence_len": args.sequence_len,
        "sequence_stride": args.sequence_stride,
        "shift_labels": args.shift_labels,
        "data_dirs": args.data_dirs,
        "dry_run": args.dry_run,
        "dry_run_images_per_file": args.dry_run_images_per_file,
        "color_jitter_p": args.color_jitter_p,
        "sky_mask_p": args.sky_mask_p,
        "zoom_range": list(args.zoom_range),
        "warp_shift_range": list(args.warp_shift_range),
        "model": model_cfg,
        "compatibility_signature": compatibility_signature,
    }


def _write_labels_if_needed(labels_path: Path, file_paths: list[str], max_per_file: int | None, expected_total: int):
    all_labels = []
    for file_path in tqdm(file_paths, desc="Loading labels", leave=False):
        with h5py.File(file_path, "r", libver="latest", swmr=True) as h5f:
            count = h5f["labels"].shape[0]
            if max_per_file is not None:
                count = min(count, max_per_file)
            labels = h5f["labels"][:count]
        labels = _to_wasd(labels.astype(np.int8, copy=False))
        all_labels.append(labels)

    source_labels = np.concatenate(all_labels, axis=0).astype(np.float32, copy=False)
    if source_labels.shape[0] != expected_total:
        raise RuntimeError(
            f"Derived labels length mismatch ({source_labels.shape[0]} != expected {expected_total})"
        )

    if labels_path.exists():
        existing_labels = np.load(labels_path).astype(np.float32, copy=False)
        same = (
            existing_labels.shape == source_labels.shape
            and np.array_equal(existing_labels, source_labels)
        )
        if not same:
            raise RuntimeError(
                "Existing labels.npy does not exactly match labels derived from source files. "
                "Refusing to continue to prevent feature/label mismatch. "
                "Use a new --output_dir or delete existing artifacts in this output directory."
            )
        print("Verified existing labels.npy matches source labels")
        return existing_labels

    np.save(labels_path, source_labels)
    return source_labels


def _count_images_per_file(file_paths: list[str], max_per_file: int | None) -> list[int]:
    counts = []
    for file_path in file_paths:
        with h5py.File(file_path, "r", libver="latest", swmr=True) as h5f:
            count = int(h5f["images"].shape[0])
        if max_per_file is not None:
            count = min(count, max_per_file)
        counts.append(count)
    return counts


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute DINOv3+projection features for fast head-only training.")
    parser.add_argument("--num_versions", type=int, default=10)
    parser.add_argument("--num_flipped", type=int, default=3)
    parser.add_argument("--num_warped", type=int, default=1)
    parser.add_argument("--num_zoomed", type=int, default=3)
    parser.add_argument("--color_jitter_p", type=float, default=0.75)
    parser.add_argument("--sky_mask_p", type=float, default=0.25)
    parser.add_argument("--zoom_range", type=str, default="1.05,1.2")
    parser.add_argument("--warp_shift_range", type=str, default="50,100")
    parser.add_argument("--output_dir", type=str, default="temp/precomputed")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_dirs", nargs="+", default=DEFAULT_DATA_DIRS)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
    parser.add_argument("--label_shift", type=int, default=round(config.fps_at_recording_time / config.fps_at_test_time))
    parser.add_argument("--train_split", type=float, default=0.95)
    parser.add_argument("--sequence_len", type=int, default=config.sequence_len)
    parser.add_argument("--sequence_stride", type=int, default=config.sequence_stride)
    parser.add_argument("--shift_labels", action="store_true", default=True)
    parser.add_argument("--no_shift_labels", action="store_false", dest="shift_labels")
    parser.add_argument("--cls_option", type=str, default="both", choices=["patches_only", "both"])
    parser.add_argument("--dino_size", type=str, default="base", choices=["base", "large", "huge"])
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--transformer_dim", type=int, default=128)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_images_per_file", type=int, default=100)
    parser.add_argument("--force_reset", action="store_true", help="Delete incompatible memmaps and restart those versions.")
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    args.zoom_range = _parse_range(args.zoom_range, float, "zoom_range")
    args.warp_shift_range = _parse_range(args.warp_shift_range, int, "warp_shift_range")
    if not args.shift_labels:
        args.label_shift = 0

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = output_dir / ".progress"
    progress_dir.mkdir(parents=True, exist_ok=True)

    file_paths = _load_data_files(args.data_dirs)
    max_per_file = args.dry_run_images_per_file if args.dry_run else None
    images_per_file = _count_images_per_file(file_paths, max_per_file=max_per_file)
    file_boundaries = np.cumsum(images_per_file).tolist()
    total_images = int(file_boundaries[-1])
    stuck_offsets = [_compute_stuck_start_idx(fp) for fp in file_paths]

    print(f"Found {len(file_paths)} files, {total_images} images total")
    if args.dry_run:
        print(f"dry_run enabled: first {args.dry_run_images_per_file} images per file, 2 versions")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    device = torch.device(args.device)
    ptdtype = dtype_map[args.dtype]
    use_amp = (device.type == "cuda") and (ptdtype in (torch.float16, torch.bfloat16))
    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype) if use_amp else nullcontext()

    model, model_cfg = _load_feature_extractor(args, device=device, ptdtype=ptdtype)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, model.expected_input_hw[0], model.expected_input_hw[1], device=device)
        with amp_ctx:
            dummy_feat = model._forward_processor(dummy, batch_size=1, seq_len=1)
        num_tokens = int(dummy_feat.shape[1])
        embed_dim = int(dummy_feat.shape[2])
    del dummy, dummy_feat

    versions = _build_versions(args, rng=rng)

    compatibility_signature = _build_compatibility_signature(
        versions=versions,
        file_paths=file_paths,
        file_boundaries=file_boundaries,
        num_images=total_images,
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        label_shift=args.label_shift,
        sequence_len=args.sequence_len,
        sequence_stride=args.sequence_stride,
        train_split=args.train_split,
        cls_option=model_cfg["cls_option"],
    )

    metadata_path = output_dir / "metadata.json"
    existing_metadata = _validate_existing_metadata(metadata_path, current_signature=compatibility_signature)
    can_resume_from_done = existing_metadata is not None
    if existing_metadata is not None:
        metadata = existing_metadata
        versions = _versions_from_metadata(metadata)
        print(f"Found compatible metadata, reusing existing metadata: {metadata_path}")
    else:
        metadata = None

    print("Version assignment:")
    for spec in versions:
        print(f"  v{spec.idx}: {spec.as_dict()}")

    if not can_resume_from_done:
        stale_markers = list(progress_dir.glob("*.done"))
        for marker in stale_markers:
            marker.unlink()
        if stale_markers:
            print(f"Removed {len(stale_markers)} stale .done markers (no validated metadata resume context)")

    labels_path = output_dir / "labels.npy"
    labels = _write_labels_if_needed(
        labels_path,
        file_paths=file_paths,
        max_per_file=max_per_file,
        expected_total=total_images,
    )
    if labels.shape[0] != total_images:
        raise RuntimeError(f"labels.npy shape mismatch: {labels.shape[0]} != expected {total_images}")

    if metadata is None:
        metadata = _build_metadata(
            args=args,
            versions=versions,
            file_paths=file_paths,
            file_boundaries=file_boundaries,
            stuck_offsets=stuck_offsets,
            num_images=total_images,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            model_cfg=model_cfg,
            compatibility_signature=compatibility_signature,
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    jitter = copy.deepcopy(color_jitter_replace)
    jitter.p = args.color_jitter_p
    sky_mask = RandomSkyMask(height_limit=(0.3, 0.4), p=args.sky_mask_p)

    start_wall = time.time()
    overall_pbar = tqdm(total=len(versions), desc="Versions", position=0)
    shape = (total_images, num_tokens, embed_dim)

    for version in versions:
        print(f"\n[v{version.idx}] Starting {version.aug_type}")
        feature_path = output_dir / f"features_v{version.idx}.dat"
        memmap, created_new = _validate_or_create_memmap(feature_path, shape=shape, reset=args.force_reset)

        if created_new:
            for file_idx in range(len(file_paths)):
                marker = progress_dir / f"v{version.idx}_f{file_idx}.done"
                if marker.exists():
                    marker.unlink()

        if not feature_path.exists():
            raise RuntimeError(f"Failed to create memmap file: {feature_path}")

        version_pbar = tqdm(total=total_images, desc=f"v{version.idx}", position=1, leave=False)
        cumulative_start = 0
        done_count = 0

        for file_idx, (file_path, file_count) in enumerate(zip(file_paths, images_per_file)):
            done_marker = progress_dir / f"v{version.idx}_f{file_idx}.done"
            if done_marker.exists():
                version_pbar.update(file_count)
                cumulative_start += file_count
                done_count += 1
                continue

            print(f"[v{version.idx}] file {file_idx + 1}/{len(file_paths)}: {file_path}")
            with h5py.File(file_path, "r", libver="latest", swmr=True) as h5f:
                num_images_in_file = file_count
                for local_start in range(0, num_images_in_file, args.batch_size):
                    local_end = min(local_start + args.batch_size, num_images_in_file)
                    images_np = h5f["images"][local_start:local_end]

                    batch_tensors = []
                    for image in images_np:
                        aug_image = _augment_image(
                            image,
                            version=version,
                            color_jitter=jitter,
                            sky_mask=sky_mask,
                            rng=rng,
                        )
                        img_tensor = transform(image=aug_image)["image"]
                        batch_tensors.append(img_tensor)

                    if not batch_tensors:
                        continue

                    batch = torch.stack(batch_tensors, dim=0)
                    if device.type == "cuda":
                        batch = batch.pin_memory().to(device, non_blocking=True)
                    else:
                        batch = batch.to(device)

                    with torch.no_grad():
                        with amp_ctx:
                            feats = model._forward_processor(batch, batch_size=batch.shape[0], seq_len=1)

                    global_start = cumulative_start + local_start
                    global_end = cumulative_start + local_end
                    memmap[global_start:global_end] = feats.to(torch.float16).detach().cpu().numpy()
                    version_pbar.update(local_end - local_start)

            memmap.flush()
            done_marker.write_text("done", encoding="utf-8")
            cumulative_start += file_count
            done_count += 1

        memmap.flush()
        elapsed = time.time() - start_wall
        print(
            f"[v{version.idx}] complete ({done_count}/{len(file_paths)} files), "
            f"elapsed {elapsed / 60.0:.1f} min"
        )
        version_pbar.close()
        overall_pbar.update(1)

    overall_pbar.close()
    total_elapsed = time.time() - start_wall
    print(f"\nDone. Wrote {len(versions)} versions in {total_elapsed / 60.0:.1f} min")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
