import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from feature_training import PrecomputedFeatureDataset


def _write_tiny_feature_dir(root: Path):
    metadata = {
        "num_versions": 2,
        "num_images": 12,
        "num_tokens": 16,
        "embed_dim": 16,
        "version_info": [
            {"idx": 0, "type": "unaltered"},
            {"idx": 1, "type": "flipped"},
        ],
        "file_paths": ["data/tiny.h5"],
        "file_boundaries": [12],
        "stuck_offsets": [0],
        "sequence_len": 1,
        "sequence_stride": 1,
        "label_shift": 0,
        "train_split": 1.0,
    }
    with open(root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    labels = np.zeros((12, 4), dtype=np.float32)
    labels[:, 0] = 1.0
    np.save(root / "labels.npy", labels)

    shape = (12, 16, 16)
    for version_idx, fill in enumerate((1, 2)):
        arr = np.memmap(root / f"features_v{version_idx}.dat", mode="w+", dtype=np.float16, shape=shape)
        arr[:] = np.float16(fill)
        arr.flush()


class StreamingDataPipelineTests(unittest.TestCase):
    def test_dataset_reads_requested_versions_without_cache_layer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_tiny_feature_dir(root)

            ds = PrecomputedFeatureDataset(
                feature_dir=str(root),
                is_train=True,
                split_gap_frames=0,
            )

            _ = ds[(0, 0)]
            _ = ds[(5, 0)]
            features_v1, _ = ds[(9, 1)]
            ds.close()

            self.assertTrue(torch.all(features_v1 == torch.tensor(2.0, dtype=torch.float16)))

    def test_version_sampling_mixes_within_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_tiny_feature_dir(root)

            ds = PrecomputedFeatureDataset(
                feature_dir=str(root),
                is_train=True,
                split_gap_frames=0,
            )

            with mock.patch(
                "feature_training.np.random.randint",
                side_effect=[
                    np.zeros(4, dtype=np.int64),
                    1,
                    0,
                ],
            ):
                features, _ = ds.__getitems__([0, 1, 2, 3])

            ds.close()

            unique_feature_values = set(float(v.item()) for v in torch.unique(features))
            self.assertGreater(len(unique_feature_values), 1)

    def test_dataset_getitems_returns_batched_tensors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_tiny_feature_dir(root)

            ds = PrecomputedFeatureDataset(
                feature_dir=str(root),
                is_train=True,
                split_gap_frames=0,
            )

            features, labels = ds.__getitems__([(0, 0), (1, 1)])
            ds.close()

            self.assertIsInstance(features, torch.Tensor)
            self.assertIsInstance(labels, torch.Tensor)
            self.assertEqual(tuple(features.shape), (2, 16, 16))
            self.assertEqual(tuple(labels.shape), (2, 1, 4))
            self.assertEqual(features.dtype, torch.float16)
            self.assertEqual(labels.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
