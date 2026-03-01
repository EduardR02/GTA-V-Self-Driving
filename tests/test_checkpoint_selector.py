import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "temp"))

import checkpoint_selector as cs  # noqa: E402


class CheckpointSelectorTests(unittest.TestCase):
    def test_parse_eval_blocks_supports_extended_rows(self):
        text = """
================================================================================
EVAL @ step 500
================================================================================
Train Loss 0.6993 | Val Loss 0.5553 | Train Acc 33.62% | Val Acc 45.67%
Class | TrainTot | ValTot | TrainRec | ValRec | TrainSpec | ValSpec | TrainF1 | ValF1
    W |   76.33% |  88.59% |    85.87% |  94.49% |     58.91% |   55.75% |   88.11% | 87.55%
    A |   64.80% |  67.15% |    88.00% |  88.47% |     62.82% |   65.95% |   25.20% | 22.41%
    S |   90.29% |  96.07% |    79.48% |  54.08% |     90.73% |   96.44% |   53.42% | 31.00%
    D |   76.51% |  82.89% |    87.00% |  61.90% |     74.46% |   84.50% |   41.52% | 40.10%
================================================================================
iter 500: loss 0.7194, acc 36.33%, lr 0.000098, 29989.30ms
"""
        blocks = cs.parse_eval_blocks(text, log_file="x.log", experiment="x")
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(block.step, 500)
        self.assertAlmostEqual(block.val_loss, 0.5553, places=4)
        self.assertAlmostEqual(block.val_acc, 45.67, places=2)
        self.assertAlmostEqual(block.recall["A"], 88.47, places=2)
        self.assertAlmostEqual(block.specificity["D"], 84.50, places=2)

    def test_parse_eval_blocks_skips_incomplete_block(self):
        text = """
EVAL @ step 500
Train Loss 0.7000 | Val Loss 0.6000 | Train Acc 30.00% | Val Acc 40.00%
Class | TrainTot | ValTot | TrainRec | ValRec | TrainSpec | ValSpec
    W |   70.00% |  80.00% |    85.00% |  90.00% |     60.00% |   55.00%
    A |   60.00% |  65.00% |    80.00% |  75.00% |     62.00% |   63.00%

EVAL @ step 1000
Train Loss 0.6500 | Val Loss 0.5000 | Train Acc 35.00% | Val Acc 50.00%
Class | TrainTot | ValTot | TrainRec | ValRec | TrainSpec | ValSpec
    W |   70.00% |  80.00% |    85.00% |  90.00% |     60.00% |   55.00%
    A |   60.00% |  65.00% |    80.00% |  75.00% |     62.00% |   63.00%
    S |   90.00% |  95.00% |    78.00% |  55.00% |     92.00% |   95.00%
    D |   72.00% |  75.00% |    84.00% |  70.00% |     70.00% |   74.00%
iter 1000: loss 0.6500, acc 35.00%, lr 0.00009, 1234.00ms
"""
        blocks = cs.parse_eval_blocks(text, log_file="x.log", experiment="x")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].step, 1000)

    def test_run_writes_outputs_and_applies_s_floor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_dir = tmp_path / "experiment_logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            log_dir.joinpath("exp_alpha.log").write_text(
                """
EVAL @ step 500
Train Loss 0.6000 | Val Loss 0.4500 | Train Acc 45.00% | Val Acc 55.00%
Class | TrainTot | ValTot | TrainRec | ValRec | TrainSpec | ValSpec
    W |   80.00% |  90.00% |    90.00% |  95.00% |     70.00% |   70.00%
    A |   70.00% |  80.00% |    85.00% |  90.00% |     75.00% |   90.00%
    S |   92.00% |  96.00% |    80.00% |  20.00% |     92.00% |   95.00%
    D |   70.00% |  80.00% |    85.00% |  90.00% |     75.00% |   90.00%
iter 500: loss 0.6000, acc 45.00%, lr 0.00010, 1000.00ms

EVAL @ step 1000
Train Loss 0.5500 | Val Loss 0.4300 | Train Acc 46.00% | Val Acc 56.00%
Class | TrainTot | ValTot | TrainRec | ValRec | TrainSpec | ValSpec
    W |   80.00% |  90.00% |    90.00% |  95.00% |     70.00% |   75.00%
    A |   70.00% |  80.00% |    85.00% |  80.00% |     75.00% |   80.00%
    S |   92.00% |  96.00% |    80.00% |  40.00% |     92.00% |   95.00%
    D |   70.00% |  80.00% |    85.00% |  80.00% |     75.00% |   80.00%
iter 1000: loss 0.5500, acc 46.00%, lr 0.00009, 1000.00ms
""",
                encoding="utf-8",
            )

            log_dir.joinpath("exp_beta.log").write_text(
                """
EVAL @ step 500
Train Loss 0.6500 | Val Loss 0.5000 | Train Acc 42.00% | Val Acc 53.00%
Class | TrainTot | ValTot | TrainRec | ValRec | TrainSpec | ValSpec
    W |   80.00% |  90.00% |    90.00% |  95.00% |     70.00% |   72.00%
    A |   70.00% |  80.00% |    85.00% |  85.00% |     75.00% |   85.00%
    S |   92.00% |  96.00% |    80.00% |  50.00% |     92.00% |   95.00%
    D |   70.00% |  80.00% |    85.00% |  85.00% |     75.00% |   85.00%
iter 500: loss 0.6500, acc 42.00%, lr 0.00010, 1000.00ms
""",
                encoding="utf-8",
            )

            csv_output = tmp_path / "checkpoint_selector_summary.csv"
            top_output = tmp_path / "checkpoint_selector_top.txt"
            row_count, parsed_logs = cs.run(log_dir, csv_output, top_output)

            self.assertEqual(parsed_logs, 2)
            self.assertEqual(row_count, 3)
            self.assertTrue(csv_output.exists())
            self.assertTrue(top_output.exists())

            with csv_output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            alpha_step_500 = next(row for row in rows if row["experiment"] == "exp_alpha" and row["step"] == "500")
            self.assertEqual(alpha_step_500["drive_quality_sfloor"], "")

            top_text = top_output.read_text(encoding="utf-8")
            self.assertIn("Top 20 by drive_quality", top_text)
            self.assertIn("Top 20 by drive_quality_sfloor", top_text)
            self.assertIn("exp_alpha: drive_quality: step=500", top_text)
            self.assertIn("drive_quality_sfloor: step=1000", top_text)


if __name__ == "__main__":
    unittest.main()
