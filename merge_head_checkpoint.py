import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

UNWANTED_PREFIX = "_orig_mod."
HEAD_PREFIXES = ("transformer.", "fc_head.")


@dataclass
class MergeReport:
    expected_head_keys: int
    replaced_keys: list[str]
    missing_in_head: list[str]
    missing_in_full: list[str]
    out_ckpt: Path


def _strip_compile_prefix(key: str) -> str:
    while key.startswith(UNWANTED_PREFIX):
        key = key[len(UNWANTED_PREFIX):]
    return key


def _is_head_key(key: str) -> bool:
    return key == "cls_token" or key.startswith(HEAD_PREFIXES)


def _extract_state_dict(checkpoint: object, label: str) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{label} checkpoint must deserialize to a dict, got {type(checkpoint).__name__}")

    model_state = checkpoint.get("model")
    if isinstance(model_state, dict):
        return model_state
    return checkpoint


def _build_canonical_key_map(state_dict: dict, label: str) -> dict[str, str]:
    canonical_to_raw: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}

    for raw_key in state_dict.keys():
        canonical_key = _strip_compile_prefix(raw_key)
        existing = canonical_to_raw.get(canonical_key)
        if existing is None:
            canonical_to_raw[canonical_key] = raw_key
            continue
        if existing != raw_key:
            collisions.setdefault(canonical_key, [existing]).append(raw_key)

    if collisions:
        lines = [f"{canonical}: {raw_keys}" for canonical, raw_keys in sorted(collisions.items())]
        details = "\n  ".join(lines)
        raise RuntimeError(
            f"{label} checkpoint has duplicate canonical keys after stripping '{UNWANTED_PREFIX}':\n  {details}"
        )

    return canonical_to_raw


def _preview(keys: list[str], limit: int = 8) -> str:
    if not keys:
        return "none"
    shown = ", ".join(keys[:limit])
    if len(keys) > limit:
        shown += f", ... (+{len(keys) - limit} more)"
    return shown


def merge_head_weights(full_ckpt_path: Path, head_ckpt_path: Path, out_ckpt_path: Path, strict: bool = True) -> MergeReport:
    full_ckpt = torch.load(full_ckpt_path, map_location="cpu")
    head_ckpt = torch.load(head_ckpt_path, map_location="cpu")

    full_state = _extract_state_dict(full_ckpt, "full")
    head_state = _extract_state_dict(head_ckpt, "head")

    full_key_map = _build_canonical_key_map(full_state, "full")
    head_key_map = _build_canonical_key_map(head_state, "head")

    full_head_keys = sorted(k for k in full_key_map if _is_head_key(k))
    head_head_keys = sorted(k for k in head_key_map if _is_head_key(k))

    if not full_head_keys:
        raise RuntimeError("No head keys found in full checkpoint (expected cls_token, transformer.*, fc_head.*)")
    if not head_head_keys:
        raise RuntimeError("No head keys found in head checkpoint (expected cls_token, transformer.*, fc_head.*)")

    full_set = set(full_head_keys)
    head_set = set(head_head_keys)

    missing_in_head = sorted(full_set - head_set)
    missing_in_full = sorted(head_set - full_set)
    replace_keys = [key for key in full_head_keys if key in head_set]

    shape_errors = []
    for key in replace_keys:
        full_tensor = full_state[full_key_map[key]]
        head_tensor = head_state[head_key_map[key]]
        if tuple(full_tensor.shape) != tuple(head_tensor.shape):
            shape_errors.append(f"{key}: full {tuple(full_tensor.shape)} vs head {tuple(head_tensor.shape)}")

    if shape_errors:
        message = "\n  ".join(shape_errors[:8])
        if len(shape_errors) > 8:
            message += f"\n  ... (+{len(shape_errors) - 8} more)"
        raise RuntimeError(f"Shape mismatch for merge keys:\n  {message}")

    if strict and (missing_in_head or missing_in_full):
        raise RuntimeError(
            "Strict merge failed due to missing head keys. "
            f"missing_in_head={len(missing_in_head)} ({_preview(missing_in_head)}), "
            f"missing_in_full={len(missing_in_full)} ({_preview(missing_in_full)}). "
            "Use --no_strict for partial merge."
        )

    for key in replace_keys:
        full_raw_key = full_key_map[key]
        head_raw_key = head_key_map[key]
        full_state[full_raw_key] = head_state[head_raw_key].detach().clone()

    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(full_ckpt, out_ckpt_path)

    return MergeReport(
        expected_head_keys=len(full_head_keys),
        replaced_keys=replace_keys,
        missing_in_head=missing_in_head,
        missing_in_full=missing_in_full,
        out_ckpt=out_ckpt_path,
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge head-only checkpoint weights into a full-model checkpoint. "
            "Copies cls_token, transformer.*, and fc_head.* from --head_ckpt into --full_ckpt."
        )
    )
    parser.add_argument("--full_ckpt", type=Path, required=True, help="Path to full-model checkpoint")
    parser.add_argument("--head_ckpt", type=Path, required=True, help="Path to head-only checkpoint")
    parser.add_argument("--out_ckpt", type=Path, required=True, help="Output path for merged checkpoint")
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Fail if any expected merge key is missing (default)",
    )
    parser.add_argument(
        "--no_strict",
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow partial merge when keys are missing",
    )
    return parser


def _print_report(report: MergeReport) -> None:
    print(f"expected_head_keys: {report.expected_head_keys}")
    print(f"replaced: {len(report.replaced_keys)}")
    print(f"missing_in_head: {len(report.missing_in_head)}")
    if report.missing_in_head:
        print(f"missing_in_head_keys: {_preview(report.missing_in_head)}")
    print(f"missing_in_full: {len(report.missing_in_full)}")
    if report.missing_in_full:
        print(f"missing_in_full_keys: {_preview(report.missing_in_full)}")
    print(f"out_ckpt: {report.out_ckpt}")


def main() -> int:
    args = create_parser().parse_args()
    try:
        report = merge_head_weights(
            full_ckpt_path=args.full_ckpt,
            head_ckpt_path=args.head_ckpt,
            out_ckpt_path=args.out_ckpt,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
