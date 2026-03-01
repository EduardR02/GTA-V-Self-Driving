import argparse
import bisect
from collections import deque
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from fp8_utils import add_fp8_cli_args, maybe_enable_fp8
from model import EfficientTransformer, EfficientTransformerBlock
from muon import SingleDeviceMuonWithAuxAdam


UNWANTED_PREFIX = "_orig_mod."
VALID_WARP_LABEL = np.array([1, 0, 0, 0], dtype=np.float32)
ZERO_LABEL = np.array([0, 0, 0, 0], dtype=np.float32)
FINAL_POS_WEIGHT = torch.tensor([1.0, 11.285, 25.556 / 2, 11.392], dtype=torch.float32)


def get_effective_pos_weight(
    base_pos_weight: torch.Tensor | None = None,
    *,
    a_pos_weight_scale: float = 1.0,
    s_pos_weight_scale: float = 1.0,
    d_pos_weight_scale: float = 1.0,
) -> torch.Tensor:
    base = FINAL_POS_WEIGHT if base_pos_weight is None else base_pos_weight
    scaled = base.clone().to(dtype=torch.float32)
    scaled[1] *= float(a_pos_weight_scale)
    scaled[2] *= float(s_pos_weight_scale)
    scaled[3] *= float(d_pos_weight_scale)
    return scaled


def _sample_mixed_versions(num_versions: int, batch_size: int) -> np.ndarray:
    if batch_size <= 0:
        return np.empty((0,), dtype=np.int64)
    if num_versions <= 1:
        return np.zeros((batch_size,), dtype=np.int64)

    versions = np.random.randint(0, num_versions, size=batch_size, dtype=np.int64)
    if batch_size > 1 and np.all(versions == versions[0]):
        replace_pos = int(np.random.randint(0, batch_size))
        replacement = int(np.random.randint(0, num_versions - 1))
        if replacement >= versions[0]:
            replacement += 1
        versions[replace_pos] = replacement
    return versions


class ChunkedRandomSampler(Sampler[int]):
    def __init__(self, data_source: Dataset, chunk_size: int, seed: int = 0):
        self.data_source = data_source
        self.chunk_size = max(int(chunk_size), 1)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        if n <= 0:
            return iter(())

        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        perm = torch.randperm(n, generator=generator)
        num_chunks = (n + self.chunk_size - 1) // self.chunk_size
        if num_chunks <= 1:
            return iter(perm.tolist())

        chunk_order = torch.randperm(num_chunks, generator=generator).tolist()

        def _yield_indices():
            for chunk_idx in chunk_order:
                start = chunk_idx * self.chunk_size
                end = min(start + self.chunk_size, n)
                for item in perm[start:end].tolist():
                    yield int(item)

        return _yield_indices()


class AsyncBatchPrefetcher:
    def __init__(self, loader, device: torch.device, prefetch_batches: int = 2, **_ignored):
        self.loader = loader
        self.device = torch.device(device)
        self.prefetch_batches = max(int(prefetch_batches), 1)

    def __len__(self):
        return len(self.loader)

    @staticmethod
    def _to_device(batch, device: torch.device):
        if torch.is_tensor(batch):
            if batch.device == device:
                return batch
            return batch.to(device, non_blocking=True)
        if isinstance(batch, tuple):
            return tuple(AsyncBatchPrefetcher._to_device(v, device) for v in batch)
        if isinstance(batch, list):
            return [AsyncBatchPrefetcher._to_device(v, device) for v in batch]
        if isinstance(batch, dict):
            return {k: AsyncBatchPrefetcher._to_device(v, device) for k, v in batch.items()}
        return batch

    @staticmethod
    def _record_stream(batch, stream):
        if torch.is_tensor(batch):
            batch.record_stream(stream)
            return
        if isinstance(batch, tuple) or isinstance(batch, list):
            for item in batch:
                AsyncBatchPrefetcher._record_stream(item, stream)
            return
        if isinstance(batch, dict):
            for item in batch.values():
                AsyncBatchPrefetcher._record_stream(item, stream)

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        data_iter = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)
        queue = deque()

        def _prefetch_one() -> bool:
            try:
                batch = next(data_iter)
            except StopIteration:
                return False
            with torch.cuda.stream(stream):
                moved = self._to_device(batch, self.device)
            queue.append(moved)
            return True

        for _ in range(self.prefetch_batches):
            if not _prefetch_one():
                break

        while queue:
            current_stream = torch.cuda.current_stream(device=self.device)
            current_stream.wait_stream(stream)
            batch = queue.popleft()
            self._record_stream(batch, current_stream)
            yield batch
            _prefetch_one()


AsyncPrefetcher = AsyncBatchPrefetcher


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def _strip_compile_prefix(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith(UNWANTED_PREFIX):
            cleaned[key[len(UNWANTED_PREFIX):]] = value
        else:
            cleaned[key] = value
    return cleaned


def _moving_average(values: list[float], window: int) -> np.ndarray:
    if len(values) < window:
        return np.array([])
    cumsum = np.cumsum(np.insert(np.asarray(values), 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


class PrecomputedFeatureDataset(Dataset):
    def __init__(
        self,
        feature_dir: str,
        is_train: bool,
        sequence_len: int | None = None,
        sequence_stride: int | None = None,
        label_shift: int | None = None,
        train_split: float | None = None,
    ):
        super().__init__()
        self.feature_dir = Path(feature_dir)
        metadata_path = self.feature_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.is_train = is_train
        self.num_versions = int(self.metadata["num_versions"])
        self.num_images = int(self.metadata["num_images"])
        self.num_tokens = int(self.metadata["num_tokens"])
        self.embed_dim = int(self.metadata["embed_dim"])

        self.version_info = self.metadata["version_info"]
        self.file_paths = self.metadata["file_paths"]
        self.file_boundaries = self.metadata["file_boundaries"]
        self.stuck_offsets = self.metadata["stuck_offsets"]

        self.sequence_len = int(sequence_len if sequence_len is not None else self.metadata.get("sequence_len", 3))
        self.sequence_stride = int(sequence_stride if sequence_stride is not None else self.metadata.get("sequence_stride", 10))
        self.label_shift = int(label_shift if label_shift is not None else self.metadata.get("label_shift", 0))
        self.train_split = float(train_split if train_split is not None else self.metadata.get("train_split", 0.95))

        self.feature_shape = (self.num_images, self.num_tokens, self.embed_dim)
        self.feature_paths = []
        for version_idx in range(self.num_versions):
            path = self.feature_dir / f"features_v{version_idx}.dat"
            if not path.exists():
                raise FileNotFoundError(f"Missing precomputed features: {path}")
            self.feature_paths.append(path)

        labels_path = self.feature_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.npy not found: {labels_path}")
        labels_probe = np.load(labels_path, mmap_mode="r")
        labels_shape = tuple(int(dim) for dim in labels_probe.shape)
        if labels_shape[0] != self.num_images:
            raise RuntimeError(f"labels shape mismatch: {labels_shape[0]} != metadata.num_images={self.num_images}")
        handle = getattr(labels_probe, "_mmap", None)
        if handle is not None:
            handle.close()

        self.labels_path = labels_path
        self.labels_shape = labels_shape
        self.labels = None
        self.features = None

        self.file_lengths = self._compute_file_lengths(self.file_boundaries)
        self.active_file_indices, self.lookup_table = self._create_lookup_table()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["features"] = None
        state["labels"] = None
        return state

    def _ensure_memmaps(self):
        if self.features is None:
            self.features = [
                np.memmap(path, mode="r", dtype=np.float16, shape=self.feature_shape)
                for path in self.feature_paths
            ]
        if self.labels is None:
            self.labels = np.load(self.labels_path, mmap_mode="r")
        return self.features, self.labels

    def close(self):
        if self.features is not None:
            for mmap in self.features:
                handle = getattr(mmap, "_mmap", None)
                if handle is not None:
                    handle.close()
            self.features = None
        if self.labels is not None:
            handle = getattr(self.labels, "_mmap", None)
            if handle is not None:
                handle.close()
            self.labels = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _compute_file_lengths(file_boundaries: list[int]) -> list[int]:
        lengths = []
        prev = 0
        for boundary in file_boundaries:
            lengths.append(boundary - prev)
            prev = boundary
        return lengths

    def _create_lookup_table(self):
        lookup = []
        active_files = []
        total_samples = 0

        effective_sequence_length = (self.sequence_len - 1) * self.sequence_stride + self.label_shift
        for file_idx, file_len in enumerate(self.file_lengths):
            file_samples = file_len - effective_sequence_length
            file_path = str(self.file_paths[file_idx])
            if "stuck" in file_path:
                stuck_start = int(self.stuck_offsets[file_idx])
                file_samples -= max(stuck_start - effective_sequence_length, 0)

            if file_samples <= 0:
                continue

            total_samples += file_samples
            active_files.append(file_idx)
            lookup.append(total_samples)

        if not lookup:
            raise RuntimeError("No valid sequence samples in precomputed dataset")
        return active_files, lookup

    def _get_split_idx(self):
        return int(self.lookup_table[-1] * self.train_split)

    def __len__(self):
        split_idx = self._get_split_idx()
        total = self.lookup_table[-1]
        return split_idx if self.is_train else total - split_idx

    def _global_offset(self, file_idx: int) -> int:
        return self.file_boundaries[file_idx - 1] if file_idx > 0 else 0

    def _resolve_frame_indices(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")
        if not self.is_train:
            idx += self._get_split_idx()

        file_pos = bisect.bisect_right(self.lookup_table, idx)
        file_idx = self.active_file_indices[file_pos]
        prev = self.lookup_table[file_pos - 1] if file_pos > 0 else 0
        local_idx = idx - prev

        sequence_range = (self.sequence_len - 1) * self.sequence_stride
        file_path = str(self.file_paths[file_idx])
        if "stuck" in file_path:
            start_k = int(self.stuck_offsets[file_idx])
            local_idx += max(start_k - sequence_range - self.label_shift, 0)

        img_indices_local = np.array([local_idx + i * self.sequence_stride for i in range(self.sequence_len)], dtype=np.int64)
        label_indices_local = np.array(
            [
            local_idx + self.label_shift + i * self.sequence_stride for i in range(self.sequence_len)
            ],
            dtype=np.int64,
        )
        offset = self._global_offset(file_idx)
        return img_indices_local + offset, label_indices_local + offset

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        if len(indices) == 0:
            return []

        features_memmaps, labels_memmap = self._ensure_memmaps()
        batch_size = len(indices)

        if self.is_train:
            version_indices = _sample_mixed_versions(self.num_versions, batch_size)
        else:
            version_indices = np.zeros((batch_size,), dtype=np.int64)

        records = []
        for out_idx, idx in enumerate(indices):
            img_indices, label_indices = self._resolve_frame_indices(int(idx))
            records.append(
                {
                    "out_idx": out_idx,
                    "img_indices": img_indices,
                    "label_indices": label_indices,
                    "version_idx": int(version_indices[out_idx]),
                }
            )

        grouped: dict[int, list[dict]] = {}
        for rec in records:
            grouped.setdefault(rec["version_idx"], []).append(rec)

        out = [None] * batch_size

        for version_idx, grouped_records in grouped.items():
            all_img_indices = np.concatenate([rec["img_indices"] for rec in grouped_records])
            img_order = np.argsort(all_img_indices, kind="stable")
            sorted_img = all_img_indices[img_order]
            sorted_features = np.asarray(features_memmaps[version_idx][sorted_img], dtype=np.float16)
            inv_img_order = np.empty_like(img_order)
            inv_img_order[img_order] = np.arange(img_order.size)
            grouped_features = sorted_features[inv_img_order].reshape(
                len(grouped_records),
                self.sequence_len,
                self.num_tokens,
                self.embed_dim,
            )

            all_label_indices = np.concatenate([rec["label_indices"] for rec in grouped_records])
            label_order = np.argsort(all_label_indices, kind="stable")
            sorted_labels = np.asarray(labels_memmap[all_label_indices[label_order]], dtype=np.float32)
            inv_label_order = np.empty_like(label_order)
            inv_label_order[label_order] = np.arange(label_order.size)
            grouped_labels = sorted_labels[inv_label_order].reshape(len(grouped_records), self.sequence_len, -1)

            version_meta = self.version_info[version_idx]
            aug_type = version_meta.get("type", "unaltered")

            for local_pos, rec in enumerate(grouped_records):
                labels = grouped_labels[local_pos].copy()
                features = grouped_features[local_pos]

                if aug_type == "flipped":
                    labels[:, [1, 3]] = labels[:, [3, 1]]
                elif aug_type == "warped":
                    last_label = labels[-1]
                    qualifies = np.array_equal(last_label, VALID_WARP_LABEL) or np.array_equal(last_label, ZERO_LABEL)
                    if not qualifies:
                        features = np.asarray(features_memmaps[0][rec["img_indices"]], dtype=np.float16)
                    else:
                        direction = version_meta.get("warp_direction", "left")
                        if direction == "left":
                            labels[-1, 1] = 1.0
                        else:
                            labels[-1, 3] = 1.0

                flat = np.asarray(features, dtype=np.float16).reshape(self.sequence_len * self.num_tokens, self.embed_dim)
                out[rec["out_idx"]] = (
                    torch.from_numpy(flat),
                    torch.from_numpy(labels.astype(np.float32, copy=False)),
                )

        return out


class HeadOnlyModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        model_dim: int | None = None,
        num_classes: int = 4,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        pos_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = int(model_dim) if model_dim is not None else None
        self.num_classes = num_classes
        self.label_smoothing = float(label_smoothing)

        hidden_dim = self.embed_dim
        self.projection = None
        if self.model_dim is not None and self.model_dim != self.embed_dim:
            self.projection = nn.Linear(self.embed_dim, self.model_dim)
            hidden_dim = self.model_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.transformer = EfficientTransformer(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_xformers=True,
            max_seq_len=max_seq_len,
        )
        self.fc_head = nn.Linear(hidden_dim, num_classes)
        if pos_weights is None:
            pos_weights = FINAL_POS_WEIGHT.clone()
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None):
        if self.projection is not None:
            features = self.projection(features)

        batch = features.shape[0]
        cls = self.cls_token.expand(batch, 1, -1)
        x = torch.cat([cls, features], dim=1)
        x = self.transformer(x)[:, 0]
        logits = self.fc_head(x)

        loss = None
        if labels is not None:
            targets = labels[:, -1]
            if self.label_smoothing > 0.0:
                eps = self.label_smoothing
                targets = targets * (1.0 - eps) + 0.5 * eps
            loss = self.loss_fct(logits, targets)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, muon_lr):
        hidden_params = [p for p in self.transformer.parameters() if p.requires_grad]
        hidden_weights = [p for p in hidden_params if p.ndim >= 2]
        hidden_1d = [p for p in hidden_params if p.ndim < 2]

        head_params = [p for p in self.fc_head.parameters() if p.requires_grad]
        head_decay = [p for p in head_params if p.ndim >= 2]
        head_nodecay = [p for p in head_params if p.ndim < 2]

        projection_params = []
        if self.projection is not None:
            projection_params = [p for p in self.projection.parameters() if p.requires_grad]

        proj_decay = [p for p in projection_params if p.ndim >= 2]
        proj_nodecay = [p for p in projection_params if p.ndim < 2]

        adam_decay = head_decay + proj_decay
        adam_nodecay = head_nodecay + hidden_1d + [self.cls_token]
        adam_nodecay += proj_nodecay

        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=muon_lr, weight_decay=weight_decay),
            dict(params=adam_decay, use_muon=False, lr=learning_rate, betas=betas, weight_decay=weight_decay),
            dict(params=adam_nodecay, use_muon=False, lr=learning_rate, betas=betas, weight_decay=0.0),
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        embed_dim: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int,
        model_dim: int | None = None,
        pos_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = _strip_compile_prefix(state_dict)

        keep = {}
        for key, value in state_dict.items():
            if key == "cls_token" or key.startswith("projection.") or key.startswith("transformer.") or key.startswith("fc_head."):
                keep[key] = value

        model = cls(
            embed_dim=embed_dim,
            model_dim=model_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pos_weights=pos_weights,
            label_smoothing=label_smoothing,
        )
        msg = model.load_state_dict(keep, strict=False)
        print(f"Loaded head checkpoint: {checkpoint_path}")
        print(f"Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")
        return model


def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr, lr_restart_cycles=1):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr

    decay_total = lr_decay_iters - warmup_iters
    if decay_total <= 0:
        return min_lr

    cycles = max(int(lr_restart_cycles), 1)
    if cycles == 1:
        decay_ratio = (it - warmup_iters) / decay_total
    else:
        if it == lr_decay_iters:
            decay_ratio = 1.0
        else:
            cycle_length = decay_total / cycles
            cycle_pos = (it - warmup_iters) % cycle_length
            decay_ratio = cycle_pos / cycle_length

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def update_lr(args, optimizer, current_lr, iter_num):
    lr = get_lr(
        iter_num,
        args.warmup_iters,
        args.lr_decay_iters,
        args.learning_rate,
        args.min_lr,
        args.lr_restart_cycles,
    )
    if current_lr == lr:
        return current_lr

    ratio = lr / args.learning_rate if args.learning_rate > 0 else 1.0
    for group in optimizer.param_groups:
        if group.get("use_muon", False):
            group["lr"] = args.muon_lr * ratio
        else:
            group["lr"] = lr
    return lr


@torch.no_grad()
def calc_accuracy(logits: torch.Tensor, labels: torch.Tensor, return_per_class: bool = False):
    preds = torch.sigmoid(logits.float()) >= 0.5
    targets = labels[:, -1].to(dtype=torch.bool, device=preds.device)
    exact = (preds == targets).all(dim=-1).float().mean().item()

    if not return_per_class:
        return exact

    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    per_class_total = np.mean(preds_np == targets_np, axis=0)
    per_class_recall = np.array(
        [
            np.mean(preds_np[targets_np[:, i], i]) if np.any(targets_np[:, i]) else np.nan
            for i in range(targets_np.shape[1])
        ]
    )
    per_class_spec = np.array(
        [
            np.mean(~preds_np[~targets_np[:, i], i]) if np.any(~targets_np[:, i]) else np.nan
            for i in range(targets_np.shape[1])
        ]
    )
    return exact, (per_class_total, per_class_recall, per_class_spec)


def _next_batch(loader, loader_iter):
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        loader_iter = iter(loader)
        try:
            return next(loader_iter), loader_iter
        except StopIteration:
            return None, loader_iter


def _move_to_device_if_needed(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    if device.type == "cuda" and tensor.device.type == "cpu":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, args, ctx, device):
    model.eval()
    out = {}
    empty_per_class = tuple(np.full(4, np.nan, dtype=np.float32) for _ in range(3))

    for split, loader in (("train", train_loader), ("val", val_loader)):
        available_batches = len(loader)
        if available_batches == 0:
            print(f"{split} eval skipped: DataLoader has zero batches")
            out[split] = float("nan")
            out[f"{split}_accuracy"] = float("nan")
            out[f"{split}_per_class"] = empty_per_class
            continue

        eval_steps = min(args.eval_iters, available_batches)
        loader_iter = iter(loader)
        losses = []
        all_logits = []
        all_labels = []

        for _ in tqdm(range(eval_steps), desc=f"{split} eval", leave=False):
            batch, loader_iter = _next_batch(loader, loader_iter)
            if batch is None:
                break

            x, y = batch
            x = _move_to_device_if_needed(x, device)
            y_dev = _move_to_device_if_needed(y, device)

            with ctx:
                logits, loss = model(x, labels=y_dev)

            losses.append(float(loss.detach().cpu().item()))
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

        if not losses:
            print(f"{split} eval skipped: no batches were produced")
            out[split] = float("nan")
            out[f"{split}_accuracy"] = float("nan")
            out[f"{split}_per_class"] = empty_per_class
            continue

        logits_t = torch.cat(all_logits, dim=0)
        labels_t = torch.cat(all_labels, dim=0)
        acc, per_class = calc_accuracy(logits_t, labels_t, return_per_class=True)

        out[split] = float(np.mean(losses))
        out[f"{split}_accuracy"] = acc
        out[f"{split}_per_class"] = per_class

    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def print_eval(iter_num, eval_stats):
    print("=" * 80)
    print(f"EVAL @ step {iter_num}")
    print("=" * 80)
    print(
        f"Train Loss {eval_stats['train']:.4f} | Val Loss {eval_stats['val']:.4f} | "
        f"Train Acc {eval_stats['train_accuracy'] * 100:.2f}% | Val Acc {eval_stats['val_accuracy'] * 100:.2f}%"
    )
    train_total, train_recall, train_spec = eval_stats["train_per_class"]
    val_total, val_recall, val_spec = eval_stats["val_per_class"]
    names = ["W", "A", "S", "D"]
    print("Class | TrainTot | ValTot | TrainRec | ValRec | TrainSpec | ValSpec")
    for i, name in enumerate(names):
        print(
            f"{name:>5} | {train_total[i]*100:>7.2f}% | {val_total[i]*100:>6.2f}% | "
            f"{train_recall[i]*100:>8.2f}% | {val_recall[i]*100:>6.2f}% | "
            f"{train_spec[i]*100:>9.2f}% | {val_spec[i]*100:>7.2f}%"
        )
    print("=" * 80)


def plot_metrics(metrics: dict, output_path: Path, smooth_window: int = 50):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax1.plot(metrics["iters"], metrics["train_loss"], alpha=0.25, color="#1f77b4", label="Train Loss")
    smoothed = _moving_average(metrics["train_loss"], smooth_window)
    if smoothed.size > 0:
        idx0 = smooth_window - 1
        ax1.plot(metrics["iters"][idx0:idx0 + smoothed.shape[0]], smoothed, color="#1f77b4", label="Train Loss (smoothed)")
    ax1.plot(metrics["eval_iters"], metrics["val_loss"], color="#d62728", marker="o", linestyle="--", label="Val Loss")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.plot(metrics["iters"], metrics["train_acc"], alpha=0.25, color="#2ca02c", label="Train Acc")
    smoothed_acc = _moving_average(metrics["train_acc"], smooth_window)
    if smoothed_acc.size > 0:
        idx0 = smooth_window - 1
        ax2.plot(metrics["iters"][idx0:idx0 + smoothed_acc.shape[0]], smoothed_acc, color="#2ca02c", label="Train Acc (smoothed)")
    ax2.plot(metrics["eval_iters"], metrics["eval_train_acc"], color="#9467bd", marker="s", linestyle="--", label="Eval Train Acc")
    ax2.plot(metrics["eval_iters"], metrics["val_acc"], color="#ff7f0e", marker="o", linestyle="--", label="Val Acc")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_checkpoint(model, optimizer, args, iter_num, best_val_loss, out_path: Path):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state_dict = raw_model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key == "cls_token" or key.startswith("transformer.") or key.startswith("fc_head.")
    }
    ckpt = {
        "model": filtered,
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": vars(args),
    }
    torch.save(ckpt, out_path)


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Train transformer head on precomputed DINOv3 features")
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="models/feature_head")
    parser.add_argument("--save_name", type=str, default="feature_head.pt")
    parser.add_argument("--metrics_name", type=str, default="feature_head_metrics.png")

    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing epsilon for BCE targets (recommended small values like 0.05)",
    )
    parser.add_argument("--num_classes", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--log_memory_interval", type=int, default=0)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--muon_lr", type=float, default=0.0015)
    parser.add_argument("--min_lr", type=float, default=3e-6)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--lr_decay_iters", type=int, default=5000)
    parser.add_argument("--lr_restart_cycles", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.075)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.995)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--a_pos_weight_scale", type=float, default=1.0)
    parser.add_argument("--s_pos_weight_scale", type=float, default=1.0)
    parser.add_argument("--d_pos_weight_scale", type=float, default=1.0)

    parser.add_argument("--sequence_len", type=int, default=None)
    parser.add_argument("--sequence_stride", type=int, default=None)
    parser.add_argument("--label_shift", type=int, default=None)
    parser.add_argument("--train_split", type=float, default=None)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--async_prefetch_batches", type=int, default=2)
    parser.add_argument("--shuffle_chunk_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
    parser.add_argument("--compile", type=_parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no_compile", action="store_false", dest="compile")
    parser.add_argument("--resume_optimizer", action="store_true", default=True)
    parser.add_argument("--no_resume_optimizer", action="store_false", dest="resume_optimizer")
    parser.add_argument("--always_save_checkpoint", action="store_true", default=True)
    add_fp8_cli_args(parser, default_mode="auto")
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    metadata_path = Path(args.feature_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json missing in feature_dir: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if args.embed_dim is None:
        args.embed_dim = int(metadata["embed_dim"])
    if args.sequence_len is None:
        args.sequence_len = int(metadata.get("sequence_len", 3))
    if args.sequence_stride is None:
        args.sequence_stride = int(metadata.get("sequence_stride", 10))
    if args.label_shift is None:
        args.label_shift = int(metadata.get("label_shift", 0))
    if args.train_split is None:
        args.train_split = float(metadata.get("train_split", 0.95))

    num_tokens = int(metadata["num_tokens"])
    flat_tokens = args.sequence_len * num_tokens
    max_seq_len = flat_tokens + 1
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PrecomputedFeatureDataset(
        feature_dir=args.feature_dir,
        is_train=True,
        sequence_len=args.sequence_len,
        sequence_stride=args.sequence_stride,
        label_shift=args.label_shift,
        train_split=args.train_split,
    )
    val_dataset = PrecomputedFeatureDataset(
        feature_dir=args.feature_dir,
        is_train=False,
        sequence_len=args.sequence_len,
        sequence_stride=args.sequence_stride,
        label_shift=args.label_shift,
        train_split=args.train_split,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": True,
        "num_workers": args.num_workers,
        "persistent_workers": (args.num_workers > 0),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_sampler = ChunkedRandomSampler(
        train_dataset,
        chunk_size=args.shuffle_chunk_size,
        seed=args.seed,
    )

    base_train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    train_loader = AsyncBatchPrefetcher(
        base_train_loader,
        device=device,
        prefetch_batches=args.async_prefetch_batches,
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            "Train DataLoader has zero batches. Reduce --batch_size, lower --train_split, "
            "or change training loader drop_last behavior."
        )

    pos_weights = get_effective_pos_weight(
        FINAL_POS_WEIGHT,
        a_pos_weight_scale=args.a_pos_weight_scale,
        s_pos_weight_scale=args.s_pos_weight_scale,
        d_pos_weight_scale=args.d_pos_weight_scale,
    )

    if args.checkpoint_path:
        model = HeadOnlyModel.from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            embed_dim=args.embed_dim,
            model_dim=args.model_dim,
            num_classes=args.num_classes,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_len=max_seq_len,
            pos_weights=pos_weights,
            label_smoothing=args.label_smoothing,
        )
    else:
        model = HeadOnlyModel(
            embed_dim=args.embed_dim,
            model_dim=args.model_dim,
            num_classes=args.num_classes,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_len=max_seq_len,
            pos_weights=pos_weights,
            label_smoothing=args.label_smoothing,
        )

    model = model.to(device)

    model, fp8_state = maybe_enable_fp8(
        model,
        mode=args.fp8,
        use_dinov3=True,
        device=device,
    )

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        muon_lr=args.muon_lr,
    )

    iter_num = 0
    best_val_loss = float("inf")
    if args.checkpoint_path and args.resume_optimizer:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "optimizer" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    iter_num = int(ckpt.get("iter_num", 0))
                    best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
                except Exception as err:
                    print(f"Optimizer resume skipped: {err}")

    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as err:
            print(f"torch.compile failed, continuing without compile: {err}")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {args.dtype}")
    ptdtype = dtype_map[args.dtype]
    ctx = nullcontext() if device.type == "cpu" else torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.dtype == "float16"))

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    fp8_status = "enabled" if fp8_state.enabled else f"disabled ({fp8_state.reason})"
    if fp8_state.enabled and fp8_state.recipe is not None:
        fp8_status = f"enabled:{fp8_state.recipe}"

    config_parts = [
        f"flat_tokens={flat_tokens}",
        f"embed_dim={args.embed_dim}",
    ]
    if args.model_dim is not None:
        config_parts.append(f"model_dim={args.model_dim}")
    config_parts.extend(
        [
            f"device={device}",
            f"dtype={args.dtype}",
            f"fp8={fp8_status}",
            f"label_smoothing={args.label_smoothing}",
        ]
    )
    print(", ".join(config_parts))

    metrics = {
        "iters": [],
        "train_loss": [],
        "train_acc": [],
        "eval_iters": [],
        "val_loss": [],
        "val_acc": [],
        "eval_train_acc": [],
    }

    train_iter = iter(train_loader)
    t0 = time.time()
    current_lr = None

    while iter_num <= args.max_iters:
        current_lr = update_lr(args, optimizer, current_lr, iter_num)

        if iter_num > 0 and (iter_num % args.eval_interval == 0):
            eval_stats = estimate_loss(model, train_loader, val_loader, args, ctx, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print_eval(iter_num, eval_stats)

            metrics["eval_iters"].append(iter_num)
            metrics["val_loss"].append(eval_stats["val"])
            metrics["val_acc"].append(eval_stats["val_accuracy"])
            metrics["eval_train_acc"].append(eval_stats["train_accuracy"])

            should_save = eval_stats["val"] < best_val_loss or args.always_save_checkpoint
            if eval_stats["val"] < best_val_loss:
                best_val_loss = eval_stats["val"]
            if should_save:
                ckpt_path = out_dir / args.save_name
                save_checkpoint(model, optimizer, args, iter_num, best_val_loss, ckpt_path)

            plot_metrics(metrics, out_dir / args.metrics_name)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        loss_accum = 0.0
        logits_list = []
        labels_list = []

        for _ in range(args.gradient_accumulation_steps):
            batch, train_iter = _next_batch(train_loader, train_iter)
            if batch is None:
                raise RuntimeError(
                    "Train DataLoader produced no batches during training iteration. "
                    "Reduce --batch_size or adjust data split settings."
                )

            x, y = batch
            x = _move_to_device_if_needed(x, device)
            y_dev = _move_to_device_if_needed(y, device)

            with ctx:
                logits, loss = model(x, labels=y_dev)
                loss = loss / args.gradient_accumulation_steps

            loss_accum += float(loss.detach().cpu().item())
            logits_list.append(logits.detach().cpu())
            labels_list.append(y.detach().cpu())
            scaler.scale(loss).backward()
            del loss, logits, x, y_dev

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        logits_t = torch.cat(logits_list, dim=0)
        labels_t = torch.cat(labels_list, dim=0)
        train_acc = calc_accuracy(logits_t, labels_t, return_per_class=False)
        del logits_t, labels_t, logits_list, labels_list

        metrics["iters"].append(iter_num)
        metrics["train_loss"].append(loss_accum)
        metrics["train_acc"].append(train_acc)

        if iter_num % args.log_interval == 0:
            dt = (time.time() - t0) * 1000.0
            t0 = time.time()
            print(
                f"iter {iter_num}: loss {loss_accum:.4f}, acc {train_acc * 100:.2f}%, "
                f"lr {current_lr:.6f}, {dt:.2f}ms"
            )

        if device.type == "cuda" and args.log_memory_interval > 0 and iter_num % args.log_memory_interval == 0:
            allocated_mb = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            print(f"cuda_mem iter {iter_num}: allocated {allocated_mb:.1f} MiB, max_allocated {peak_mb:.1f} MiB")

        iter_num += 1

    final_ckpt = out_dir / args.save_name
    save_checkpoint(model, optimizer, args, iter_num, best_val_loss, final_ckpt)
    plot_metrics(metrics, out_dir / args.metrics_name)
    print(f"Training complete. Saved to {final_ckpt}")


if __name__ == "__main__":
    main()
