import json
import tempfile
import unittest
from unittest import mock

import numpy as np

import torch

from feature_training import PrecomputedFeatureDataset, _sample_mixed_versions


class VersionSamplingTests(unittest.TestCase):
    def test_single_version_always_returns_zero(self):
        sampled = _sample_mixed_versions(num_versions=1, batch_size=8)

        self.assertTrue(np.array_equal(sampled, np.zeros(8, dtype=np.int64)))

    def test_multi_version_forces_mixing_when_initial_draw_uniform(self):
        with mock.patch(
            "feature_training.np.random.randint",
            side_effect=[
                np.zeros(6, dtype=np.int64),
                2,
                0,
            ],
        ):
            sampled = _sample_mixed_versions(num_versions=4, batch_size=6)

        self.assertGreater(len(np.unique(sampled)), 1)


class PrecomputedFeatureDatasetTests(unittest.TestCase):
    def _build_tiny_feature_dir(
        self,
        *,
        num_images: int = 6,
        sequence_len: int = 2,
        sequence_stride: int = 1,
        train_split: float = 0.5,
    ) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = tmp.name

        if num_images < 4:
            raise ValueError("num_images must be at least 4")

        metadata = {
            "num_versions": 3,
            "num_images": num_images,
            "num_tokens": 2,
            "embed_dim": 2,
            "version_info": [
                {"idx": 0, "type": "unaltered"},
                {"idx": 1, "type": "flipped"},
                {"idx": 2, "type": "warped", "warp_direction": "right"},
            ],
            "file_paths": ["data\\stuck\\tiny.h5"],
            "file_boundaries": [num_images],
            "stuck_offsets": [0],
            "sequence_len": sequence_len,
            "sequence_stride": sequence_stride,
            "label_shift": 0,
            "train_split": train_split,
        }
        with open(f"{root}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        base_labels = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        repeats = (num_images + base_labels.shape[0] - 1) // base_labels.shape[0]
        labels = np.tile(base_labels, (repeats, 1))[:num_images]
        np.save(f"{root}/labels.npy", labels)

        shape = (num_images, 2, 2)
        for version_idx, fill in enumerate((1, 5, 9)):
            arr = np.memmap(
                f"{root}/features_v{version_idx}.dat",
                mode="w+",
                dtype=np.float16,
                shape=shape,
            )
            arr[:] = np.float16(fill)
            arr.flush()

        return root

    def test_dataset_split_lengths_match_expected(self):
        feature_dir = self._build_tiny_feature_dir()
        train_ds = PrecomputedFeatureDataset(feature_dir=feature_dir, is_train=True, split_gap_frames=0)
        val_ds = PrecomputedFeatureDataset(feature_dir=feature_dir, is_train=False, split_gap_frames=0)

        self.assertEqual(len(train_ds), 2)
        self.assertEqual(len(val_ds), 3)

    def test_flipped_version_swaps_a_and_d_labels(self):
        feature_dir = self._build_tiny_feature_dir()
        ds = PrecomputedFeatureDataset(feature_dir=feature_dir, is_train=True, split_gap_frames=0)

        with mock.patch("feature_training.np.random.randint", return_value=np.array([1], dtype=np.int64)):
            _, labels = ds[0]

        # last frame label starts as [0,1,0,0] and becomes [0,0,0,1]
        self.assertTrue(torch.equal(labels[-1], torch.tensor([0.0, 0.0, 0.0, 1.0])))

    def test_warped_invalid_label_falls_back_to_unaltered_features(self):
        feature_dir = self._build_tiny_feature_dir()
        ds = PrecomputedFeatureDataset(feature_dir=feature_dir, is_train=True, split_gap_frames=0)

        with mock.patch("feature_training.np.random.randint", return_value=np.array([2], dtype=np.int64)):
            features, _ = ds[0]

        self.assertTrue(torch.all(features == torch.tensor(1.0, dtype=torch.float16)))

    def test_warped_valid_label_keeps_warped_features_and_sets_direction(self):
        feature_dir = self._build_tiny_feature_dir()
        ds = PrecomputedFeatureDataset(feature_dir=feature_dir, is_train=True, split_gap_frames=0)

        # sample index 1 has last label [1,0,0,0], so warped version is valid
        with mock.patch("feature_training.np.random.randint", return_value=np.array([2], dtype=np.int64)):
            features, labels = ds[1]

        self.assertTrue(torch.all(features == torch.tensor(9.0, dtype=torch.float16)))
        self.assertEqual(float(labels[-1, 3]), 1.0)

    def test_default_split_gap_prevents_train_val_temporal_overlap(self):
        feature_dir = self._build_tiny_feature_dir(num_images=30, sequence_len=3, sequence_stride=2, train_split=0.5)
        train_ds = PrecomputedFeatureDataset(feature_dir=feature_dir, is_train=True)
        val_ds = PrecomputedFeatureDataset(feature_dir=feature_dir, is_train=False)

        train_last = int(train_ds.sequence_starts.max())
        val_first = int(val_ds.sequence_starts.min())
        span = (train_ds.sequence_len - 1) * train_ds.sequence_stride

        self.assertGreaterEqual(val_first - train_last, train_ds.sequence_len * train_ds.sequence_stride)
        self.assertLess(train_last + span, val_first)

    def test_getitems_empty_indices_returns_correct_empty_batch_shapes(self):
        feature_dir = self._build_tiny_feature_dir(sequence_len=3)
        ds = PrecomputedFeatureDataset(feature_dir=feature_dir, is_train=True, split_gap_frames=0)

        features, labels = ds.__getitems__([])

        self.assertEqual(tuple(features.shape), (0, 3 * 2, 2))
        self.assertEqual(tuple(labels.shape), (0, 3, 4))
        self.assertEqual(features.dtype, torch.float16)
        self.assertEqual(labels.dtype, torch.float32)

    def test_read_frames_keeps_original_request_order(self):
        feature_dir = self._build_tiny_feature_dir(num_images=8)
        ds = PrecomputedFeatureDataset(feature_dir=feature_dir, is_train=True, split_gap_frames=0)

        shape = (8, 2, 2)
        path = f"{feature_dir}/features_v0.dat"
        arr = np.memmap(path, mode="r+", dtype=np.float16, shape=shape)
        for i in range(shape[0]):
            arr[i] = np.float16(i)
        arr.flush()

        memmap = ds._ensure_feature_memmaps()[0]
        frame_indices = np.array([6, 2, 5], dtype=np.int64)
        frames = ds._read_frames(memmap, frame_indices)

        self.assertTrue(np.all(frames[0] == np.float16(6)))
        self.assertTrue(np.all(frames[1] == np.float16(2)))
        self.assertTrue(np.all(frames[2] == np.float16(5)))


if __name__ == "__main__":
    unittest.main()
