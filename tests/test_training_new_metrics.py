import unittest

import numpy as np

from training_new import _extract_min_recall


class ExtractMinRecallTests(unittest.TestCase):
    def test_returns_minimum_recall_when_values_present(self):
        per_class_metrics = (
            np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            np.array([0.5, 0.6, 0.4, 0.8], dtype=np.float32),
            np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
        )

        min_recall = _extract_min_recall(per_class_metrics)

        self.assertAlmostEqual(min_recall, 0.4)

    def test_ignores_nan_values(self):
        per_class_metrics = (
            np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            np.array([np.nan, 0.61, 0.42, np.nan], dtype=np.float32),
            np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
        )

        min_recall = _extract_min_recall(per_class_metrics)

        self.assertAlmostEqual(min_recall, 0.42)

    def test_returns_none_when_input_missing_or_invalid(self):
        self.assertIsNone(_extract_min_recall(None))
        self.assertIsNone(_extract_min_recall((np.array([0.1]),)))
        self.assertIsNone(
            _extract_min_recall(
                (
                    np.array([0.9, 0.8], dtype=np.float32),
                    np.array([np.nan, np.nan], dtype=np.float32),
                    np.array([0.9, 0.9], dtype=np.float32),
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
