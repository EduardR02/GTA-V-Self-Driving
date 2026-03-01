import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from feature_training import (
    AsyncBatchPrefetcher,
    ChunkedRandomSampler,
    FINAL_POS_WEIGHT,
    PrecomputedFeatureDataset,
    create_arg_parser,
    get_effective_pos_weight,
)


class _MockLoader:
    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        yield from self._batches

    def __len__(self):
        return len(self._batches)


class _SpyArray:
    def __init__(self, arr):
        self.arr = arr
        self.requests = []

    def __getitem__(self, idx):
        idx_array = np.asarray(idx)
        self.requests.append(idx_array.copy())
        return self.arr[idx]


def _write_tiny_feature_dir(root: Path):
    metadata = {
        "num_versions": 2,
        "num_images": 8,
        "num_tokens": 1,
        "embed_dim": 1,
        "version_info": [
            {"idx": 0, "type": "unaltered"},
            {"idx": 1, "type": "unaltered"},
        ],
        "file_paths": ["data/tiny.h5"],
        "file_boundaries": [8],
        "stuck_offsets": [0],
        "sequence_len": 2,
        "sequence_stride": 1,
        "label_shift": 0,
        "train_split": 1.0,
    }
    with open(root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    labels = np.zeros((8, 4), dtype=np.float32)
    labels[:, 0] = 1.0
    np.save(root / "labels.npy", labels)

    for version_idx in range(2):
        arr = np.memmap(root / f"features_v{version_idx}.dat", mode="w+", dtype=np.float16, shape=(8, 1, 1))
        for frame_idx in range(8):
            arr[frame_idx, 0, 0] = np.float16(frame_idx + version_idx * 100)
        arr.flush()


class FeatureTrainingRegressionTests(unittest.TestCase):
    def test_parser_exposes_fp8_prefetch_and_pos_weight_scaling_args(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x"])

        self.assertTrue(hasattr(args, "fp8"))
        self.assertEqual(args.async_prefetch_batches, 3)
        self.assertEqual(args.a_pos_weight_scale, 1.0)
        self.assertEqual(args.s_pos_weight_scale, 1.0)
        self.assertEqual(args.d_pos_weight_scale, 1.0)

    def test_get_effective_pos_weight_scales_only_a_s_d_entries(self):
        scaled = get_effective_pos_weight(
            FINAL_POS_WEIGHT,
            a_pos_weight_scale=0.8,
            s_pos_weight_scale=1.25,
            d_pos_weight_scale=0.7,
        )

        self.assertAlmostEqual(float(scaled[0]), float(FINAL_POS_WEIGHT[0]))
        self.assertAlmostEqual(float(scaled[1]), float(FINAL_POS_WEIGHT[1] * 0.8))
        self.assertAlmostEqual(float(scaled[2]), float(FINAL_POS_WEIGHT[2] * 1.25))
        self.assertAlmostEqual(float(scaled[3]), float(FINAL_POS_WEIGHT[3] * 0.7))

    def test_chunked_random_sampler_returns_full_permutation(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(25))
        sampler = ChunkedRandomSampler(dataset, chunk_size=6, seed=123)

        order = list(iter(sampler))

        self.assertEqual(len(order), 25)
        self.assertEqual(sorted(order), list(range(25)))

    def test_dataset_keeps_labels_memmap_lazy_until_first_access(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_tiny_feature_dir(root)

            ds = PrecomputedFeatureDataset(feature_dir=str(root), is_train=True)

            self.assertIsNone(ds.labels)
            self.assertIsNone(ds.features)
            state = ds.__getstate__()
            self.assertIsNone(state["labels"])
            self.assertIsNone(state["features"])
            ds.close()

    def test_getitems_sorts_version_group_indices_for_memmap_reads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_tiny_feature_dir(root)

            ds = PrecomputedFeatureDataset(feature_dir=str(root), is_train=True)
            ds._ensure_memmaps()
            original_features = list(ds.features)
            spy_features = [_SpyArray(original_features[0]), _SpyArray(original_features[1])]
            ds.features = spy_features

            with mock.patch("feature_training._sample_mixed_versions", return_value=np.array([1, 0, 1, 0], dtype=np.int64)):
                samples = ds.__getitems__([3, 1, 2, 0])

            self.assertEqual(len(samples), 4)
            self.assertTrue(np.all(np.diff(spy_features[0].requests[0]) >= 0))
            self.assertTrue(np.all(np.diff(spy_features[1].requests[0]) >= 0))
            first_features, _ = samples[0]
            self.assertEqual(float(first_features[0, 0]), 103.0)

            ds.features = original_features
            ds.close()

    def test_async_batch_prefetcher_cpu_passthrough(self):
        batches = [
            (torch.full((2, 3), float(i)), torch.full((2, 1, 4), float(i)))
            for i in range(3)
        ]
        loader = _MockLoader(batches)
        prefetcher = AsyncBatchPrefetcher(loader, device=torch.device("cpu"), prefetch_batches=3)

        got = list(prefetcher)

        self.assertEqual(len(got), len(batches))
        for expected, actual in zip(batches, got):
            self.assertTrue(torch.equal(expected[0], actual[0]))
            self.assertTrue(torch.equal(expected[1], actual[1]))


if __name__ == "__main__":
    unittest.main()
