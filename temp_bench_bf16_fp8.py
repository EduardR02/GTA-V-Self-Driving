import ast
import gc
import time
from contextlib import nullcontext
from pathlib import Path

import torch

import config
from fp8_utils import maybe_enable_fp8
from model import Dinov3ForTimeSeriesClassification


def load_training_new_constants(training_new_path: Path):
    wanted = {
        "dino_size",
        "dropout_p",
        "cls_option",
        "num_heads",
        "num_layers",
        "transformer_dim",
        "label_smoothing",
        "weight_decay",
        "learning_rate",
        "beta1",
        "beta2",
        "muon_learning_rate",
        "classifier_type",
    }
    values = {}
    module = ast.parse(training_new_path.read_text(encoding="utf-8"), filename=str(training_new_path))
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        if name not in wanted:
            continue
        try:
            values[name] = ast.literal_eval(node.value)
        except Exception:
            pass
    missing = sorted(wanted.difference(values.keys()))
    if missing:
        raise RuntimeError(f"Missing required constants in training_new.py: {missing}")
    return values


def train_step(model, optimizer, x, y, autocast_ctx):
    with autocast_ctx:
        _, loss = model(x, labels=y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def benchmark(use_fp8, cfg, device, warmup_steps=10, measure_steps=30):
    if cfg["classifier_type"] == "bce":
        num_classes = 4
    else:
        num_classes = len(config.outputs)

    model = Dinov3ForTimeSeriesClassification(
        cfg["dino_size"],
        num_classes,
        dropout_rate=cfg["dropout_p"],
        dtype=torch.bfloat16,
        cls_option=cfg["cls_option"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        transformer_dim=cfg["transformer_dim"],
        label_smoothing=cfg["label_smoothing"],
    )

    model.to(device)
    fp8_enabled_line = ""
    if use_fp8:
        fp8_logs = []

        def _logger(msg):
            fp8_logs.append(str(msg))
            print(msg)

        model, state = maybe_enable_fp8(
            model,
            mode="on",
            use_dinov3=True,
            device="cuda",
            logger=_logger,
        )
        fp8_enabled_line = next((line for line in fp8_logs if "[fp8] enabled:" in line), "")
        if not fp8_enabled_line:
            fp8_enabled_line = f"[fp8] enabled state: {state.enabled}, reason={state.reason}"

    model = torch.compile(model)
    optimizer = model.configure_optimizers(
        cfg["weight_decay"],
        cfg["learning_rate"],
        (cfg["beta1"], cfg["beta2"]),
        "cuda",
        muon_lr=cfg["muon_learning_rate"],
    )

    x = torch.randn(256, 3, 3, 192, 240, device=device, dtype=torch.float32)
    y = torch.randint(0, 2, (256, 3, num_classes), device=device, dtype=torch.float32)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    optimizer.zero_grad(set_to_none=True)
    for _ in range(warmup_steps):
        train_step(model, optimizer, x, y, autocast_ctx)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(measure_steps):
        train_step(model, optimizer, x, y, autocast_ctx)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / measure_steps) * 1000.0
    peak_mem = torch.cuda.max_memory_allocated(device)

    del x, y, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "avg_ms": avg_ms,
        "peak_mem": peak_mem,
        "fp8_enabled_line": fp8_enabled_line,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required for this benchmark, but CUDA BF16 is not supported")

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    cfg = load_training_new_constants(Path(__file__).parent / "training_new.py")
    device = torch.device("cuda")

    bf16 = benchmark(use_fp8=False, cfg=cfg, device=device)
    fp8 = benchmark(use_fp8=True, cfg=cfg, device=device)

    speedup = bf16["avg_ms"] / fp8["avg_ms"]

    print("\n=== BENCHMARK RESULTS ===")
    print(f"FP8 enabled log line: {fp8['fp8_enabled_line']}")
    print(f"Avg step time BF16 (ms): {bf16['avg_ms']:.3f}")
    print(f"Avg step time FP8 (ms): {fp8['avg_ms']:.3f}")
    print(f"Speedup ratio (BF16/FP8): {speedup:.3f}x")
    print(f"Peak GPU memory BF16 (MiB): {bf16['peak_mem'] / (1024 ** 2):.2f}")
    print(f"Peak GPU memory FP8 (MiB): {fp8['peak_mem'] / (1024 ** 2):.2f}")


if __name__ == "__main__":
    main()
