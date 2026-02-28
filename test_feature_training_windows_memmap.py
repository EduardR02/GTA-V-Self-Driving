import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_training import PrecomputedFeatureDataset, create_arg_parser


def _write_feature_dir(root: Path):
    metadata = {
        "num_versions": 2,
        "num_images": 8,
        "num_tokens": 4,
        "embed_dim": 4,
        "version_info": [
            {"idx": 0, "type": "unaltered"},
            {"idx": 1, "type": "flipped"},
        ],
        "file_paths": ["data/tiny.h5"],
        "file_boundaries": [8],
        "stuck_offsets": [0],
        "sequence_len": 1,
        "sequence_stride": 1,
        "label_shift": 0,
        "train_split": 1.0,
    }
    with open(root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    labels = np.zeros((8, 4), dtype=np.float32)
    labels[:, 0] = 1.0
    np.save(root / "labels.npy", labels)

    shape = (8, 4, 4)
    for version_idx in range(2):
        arr = np.memmap(root / f"features_v{version_idx}.dat", mode="w+", dtype=np.float16, shape=shape)
        arr[:] = np.float16(version_idx + 1)
        arr.flush()


class WindowsMemmapDatasetTests(unittest.TestCase):
    def test_dataset_pickles_without_serializing_memmap_payload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_feature_dir(root)

            ds = PrecomputedFeatureDataset(feature_dir=str(root), is_train=True)
            _ = ds[0]

            payload = pickle.dumps(ds, protocol=pickle.HIGHEST_PROTOCOL)
            self.assertLess(len(payload), 200_000)

            restored = pickle.loads(payload)
            features, labels = restored[0]
            self.assertEqual(tuple(features.shape), (4, 4))
            self.assertEqual(tuple(labels.shape), (1, 4))
            restored.close()
            ds.close()

    def test_compile_flag_accepts_explicit_false_value(self):
        parser = create_arg_parser()

        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x", "--compile", "False"])

        self.assertFalse(args.compile)


if __name__ == "__main__":
    unittest.main()
