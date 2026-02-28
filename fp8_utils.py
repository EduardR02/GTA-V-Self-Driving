import importlib
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


FP8_MODES = ("auto", "on", "off")
MIN_FP8_CAPABILITY = (8, 9)


@dataclass(frozen=True)
class FP8RuntimeState:
    mode: str
    enabled: bool
    reason: str
    capability: tuple[int, int] | None = None
    recipe: str | None = None
    linear_param_coverage: float | None = None


def normalize_fp8_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in FP8_MODES:
        valid = ", ".join(FP8_MODES)
        raise ValueError(f"Invalid fp8 mode '{mode}'. Expected one of: {valid}")
    return normalized


def add_fp8_cli_args(parser, default_mode: str = "on"):
    parser.add_argument(
        "--fp8",
        type=str,
        default=normalize_fp8_mode(default_mode),
        choices=FP8_MODES,
        help="FP8 mode for DINOv3 runtime (auto/on/off).",
    )


def _is_supported_capability(capability: tuple[int, int]) -> bool:
    major, minor = capability
    return (major, minor) >= MIN_FP8_CAPABILITY


def _emit(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _is_fp8_compatible_linear(module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if module.in_features < 16 or module.out_features < 16:
        return False
    if (module.in_features % 16) != 0 or (module.out_features % 16) != 0:
        return False
    return True


def _safe_linear_filter(module: nn.Module, fqn: str) -> bool:
    del fqn
    return _is_fp8_compatible_linear(module)


def _linear_skip_reason(module: nn.Linear) -> str:
    if module.in_features < 16 or module.out_features < 16:
        return "min_dim_lt_16"
    if (module.in_features % 16) != 0:
        return "in_features_not_multiple_of_16"
    if (module.out_features % 16) != 0:
        return "out_features_not_multiple_of_16"
    return "filtered_out"


def _collect_linear_stats(model: nn.Module, module_filter_fn: Callable[[nn.Module, str], bool]):
    total_linear_params = 0
    eligible_linear_params = 0
    total_linear_modules = 0
    eligible_linear_modules = 0
    skipped_reasons: dict[str, dict[str, int]] = {}

    for fqn, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        total_linear_modules += 1
        params = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
        total_linear_params += params

        is_eligible = False
        try:
            is_eligible = bool(module_filter_fn(module, fqn))
        except Exception:
            is_eligible = False

        if is_eligible:
            eligible_linear_modules += 1
            eligible_linear_params += params
            continue

        reason = _linear_skip_reason(module)
        bucket = skipped_reasons.setdefault(reason, {"modules": 0, "params": 0})
        bucket["modules"] += 1
        bucket["params"] += params

    coverage = 0.0
    if total_linear_params > 0:
        coverage = float(eligible_linear_params) / float(total_linear_params)

    return {
        "total_linear_modules": total_linear_modules,
        "eligible_linear_modules": eligible_linear_modules,
        "total_linear_params": total_linear_params,
        "eligible_linear_params": eligible_linear_params,
        "coverage": coverage,
        "skipped_reasons": skipped_reasons,
    }


def _format_skipped_reasons(skipped_reasons: dict[str, dict[str, int]]) -> str:
    if not skipped_reasons:
        return "none"
    parts = []
    for reason in sorted(skipped_reasons.keys()):
        entry = skipped_reasons[reason]
        parts.append(f"{reason}: modules={entry['modules']} params={entry['params']}")
    return "; ".join(parts)


def _build_config(float8_module, recipe_name: str | None):
    if recipe_name is None:
        return None

    config_cls = getattr(float8_module, "Float8LinearConfig", None)
    recipe_enum = getattr(float8_module, "Float8LinearRecipeName", None)
    if config_cls is None or recipe_enum is None or not hasattr(config_cls, "from_recipe_name"):
        return None

    enum_item = getattr(recipe_enum, recipe_name, None)
    if enum_item is None:
        return None

    return config_cls.from_recipe_name(enum_item)


def _get_conversion_plans(capability: tuple[int, int]):
    safe_filter = _safe_linear_filter
    is_blackwell_or_newer = capability[0] >= 12

    if is_blackwell_or_newer:
        return [
            ("tensorwise_all_linear", "TENSORWISE", safe_filter),
            ("rowwise_all_linear", "ROWWISE", safe_filter),
            ("default_all_linear", None, safe_filter),
        ]

    return [
        ("rowwise_all_linear", "ROWWISE", safe_filter),
        ("tensorwise_all_linear", "TENSORWISE", safe_filter),
        ("default_all_linear", None, safe_filter),
    ]


def _disabled_result(
    model,
    state: FP8RuntimeState,
    *,
    logger: Callable[[str], None] | None,
    strict_required: bool,
):
    _emit(logger, f"[fp8] disabled: {state.reason}")
    if strict_required:
        raise RuntimeError(f"[fp8] mode='on' requested but FP8 is unavailable: {state.reason}")
    return model, state


def maybe_enable_fp8(
    model,
    *,
    mode: str = "auto",
    use_dinov3: bool,
    device: str | torch.device,
    logger: Callable[[str], None] | None = print,
    import_module: Callable[[str], object] = importlib.import_module,
):
    fp8_mode = normalize_fp8_mode(mode)
    device_obj = torch.device(device)

    if fp8_mode == "off":
        state = FP8RuntimeState(fp8_mode, False, "disabled by configuration")
        _emit(logger, f"[fp8] disabled: {state.reason}")
        return model, state

    strict_required = (fp8_mode == "on") and use_dinov3

    if not use_dinov3:
        state = FP8RuntimeState(fp8_mode, False, "non-DINOv3 run")
        return _disabled_result(model, state, logger=logger, strict_required=False)

    if device_obj.type != "cuda":
        state = FP8RuntimeState(fp8_mode, False, f"device '{device_obj.type}' does not support FP8")
        return _disabled_result(model, state, logger=logger, strict_required=strict_required)

    if not torch.cuda.is_available():
        state = FP8RuntimeState(fp8_mode, False, "CUDA unavailable")
        return _disabled_result(model, state, logger=logger, strict_required=strict_required)

    try:
        device_index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_index)
    except Exception as exc:
        state = FP8RuntimeState(fp8_mode, False, f"could not query CUDA capability ({exc})")
        return _disabled_result(model, state, logger=logger, strict_required=strict_required)

    if not _is_supported_capability(capability):
        reason = (
            f"GPU capability sm_{capability[0]}{capability[1]} is below required "
            f"sm_{MIN_FP8_CAPABILITY[0]}{MIN_FP8_CAPABILITY[1]}"
        )
        state = FP8RuntimeState(fp8_mode, False, reason, capability=capability)
        return _disabled_result(model, state, logger=logger, strict_required=strict_required)

    try:
        float8_module = import_module("torchao.float8")
    except Exception as exc:
        state = FP8RuntimeState(fp8_mode, False, f"torchao.float8 import failed ({exc})", capability=capability)
        return _disabled_result(model, state, logger=logger, strict_required=strict_required)

    convert_fn = getattr(float8_module, "convert_to_float8_training", None)
    if convert_fn is None:
        state = FP8RuntimeState(fp8_mode, False, "torchao.float8 missing convert_to_float8_training", capability=capability)
        return _disabled_result(model, state, logger=logger, strict_required=strict_required)

    plans = _get_conversion_plans(capability)
    module_filter_fn = _safe_linear_filter
    stats = _collect_linear_stats(model, module_filter_fn)
    coverage = float(stats["coverage"])

    _emit(
        logger,
        "[fp8] linear coverage: "
        f"eligible_params={stats['eligible_linear_params']} total_params={stats['total_linear_params']} "
        f"coverage={coverage:.3f}",
    )
    if stats["eligible_linear_params"] < stats["total_linear_params"]:
        _emit(logger, f"[fp8] skipped linear modules: {_format_skipped_reasons(stats['skipped_reasons'])}")

    if stats["total_linear_params"] == 0:
        state = FP8RuntimeState(
            fp8_mode,
            False,
            "no linear modules found for FP8 conversion",
            capability=capability,
            linear_param_coverage=0.0,
        )
        return _disabled_result(model, state, logger=logger, strict_required=strict_required)

    if stats["eligible_linear_params"] == 0:
        state = FP8RuntimeState(
            fp8_mode,
            False,
            "no eligible modules for FP8 conversion",
            capability=capability,
            linear_param_coverage=coverage,
        )
        return _disabled_result(model, state, logger=logger, strict_required=strict_required)

    last_error = None
    converted_model = model
    recipe_name = "default"
    conversion_succeeded = False
    for plan_name, recipe_enum_name, plan_filter_fn in plans:
        config = None
        using_default_config = (recipe_enum_name is None)
        if recipe_enum_name is not None:
            try:
                config = _build_config(float8_module, recipe_enum_name)
                using_default_config = (config is None)
                if using_default_config:
                    _emit(logger, f"[fp8] warning: {recipe_enum_name} config unavailable, using default config")
            except Exception as exc:
                using_default_config = True
                config = None
                _emit(logger, f"[fp8] warning: could not build {recipe_enum_name} config ({exc}); using default config")

        try:
            converted_model = convert_fn(model, module_filter_fn=plan_filter_fn, config=config)
            recipe_name = "default" if using_default_config else recipe_enum_name.lower()
            _emit(logger, f"[fp8] conversion plan selected: {plan_name}")
            conversion_succeeded = True
            break
        except Exception as exc:
            last_error = f"{plan_name} failed ({exc})"
            _emit(logger, f"[fp8] warning: {last_error}")

    if not conversion_succeeded:
        if last_error is not None:
            reason = f"torchao FP8 conversion failed ({last_error})"
        else:
            reason = "torchao FP8 conversion did not succeed"
        state = FP8RuntimeState(
            fp8_mode,
            False,
            reason,
            capability=capability,
            linear_param_coverage=coverage,
        )
        return _disabled_result(model, state, logger=logger, strict_required=strict_required)

    state = FP8RuntimeState(
        fp8_mode,
        True,
        f"enabled for DINOv3 on sm_{capability[0]}{capability[1]}",
        capability=capability,
        recipe=recipe_name,
        linear_param_coverage=coverage,
    )
    _emit(
        logger,
        f"[fp8] enabled: recipe={recipe_name}, capability=sm_{capability[0]}{capability[1]}, coverage={coverage:.3f}",
    )
    return converted_model, state
