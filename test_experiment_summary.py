import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "temp"))

import experiment_summary as es  # noqa: E402


class ExperimentSummaryParsingTests(unittest.TestCase):
    def test_parse_eval_blocks_supports_extended_class_rows(self):
        text = """
================================================================================
EVAL @ step 5000
================================================================================
Train Loss 0.4798 | Val Loss 0.5812 | Train Acc 49.18% | Val Acc 46.13%
Class | TrainTot | ValTot | TrainRec | ValRec | TrainSpec | ValSpec | TrainF1 | ValF1
    W |   88.34% |  88.11% |    92.08% |  95.16% |     78.10% |   67.80% |   92.04% | 92.23%
    A |   76.19% |  74.73% |    87.28% |  92.08% |     75.39% |   73.78% |   33.13% | 27.38%
    S |   96.86% |  95.14% |    97.64% |  68.80% |     96.76% |   95.80% |   86.96% | 40.86%
    D |   75.57% |  76.29% |    86.55% |  74.47% |     74.74% |   76.45% |   33.07% | 34.16%
================================================================================
iter 5000: loss 0.6155, acc 45.31%, lr 0.000003, 3486.04ms
"""
        blocks = es.parse_eval_blocks(text)
        self.assertEqual(len(blocks), 1)

        block = blocks[0]
        self.assertEqual(block.step, 5000)
        self.assertAlmostEqual(block.val_loss, 0.5812, places=4)
        self.assertAlmostEqual(block.val_acc, 46.13, places=2)
        self.assertAlmostEqual(block.recall["A"], 92.08, places=2)
        self.assertAlmostEqual(block.specificity["D"], 76.45, places=2)

    def test_name_hints_and_lr_decay_inference(self):
        hints = es.parse_name_hints("73_15k_fastcos_nodim_drop02_seed42_batch512")
        self.assertTrue(str(hints["model_dim"]) == "nan")
        self.assertEqual(hints["max_iters"], 15000.0)
        self.assertEqual(hints["lr_decay_iters"], 5000.0)
        self.assertEqual(hints["dropout"], 0.2)
        self.assertEqual(hints["seed"], 42.0)
        self.assertEqual(hints["batch_size"], 512.0)

        iter_lr = [(0, 0.0), (100, 0.00005), (2000, 0.00002), (5000, 0.000003), (15000, 0.000003)]
        inferred = es.infer_lr_decay_iters(iter_lr, 15000)
        self.assertEqual(inferred, 5000)


if __name__ == "__main__":
    unittest.main()
