import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "temp"))

import build_dashboard as bd  # noqa: E402
import experiment_summary as es  # noqa: E402


class BuildDashboardTests(unittest.TestCase):
    def test_classify_schedule_group_distinguishes_fast_short_and_ls(self):
        cfg_fast = {"max_iters": 15000.0, "lr_decay_iters": 3000.0, "label_smoothing": 0.0}
        cfg_full_ls = {"max_iters": 15000.0, "lr_decay_iters": 15000.0, "label_smoothing": 0.01}

        self.assertEqual(bd.classify_schedule_group(cfg_fast), "Fast/short cosine")
        self.assertEqual(bd.classify_schedule_group(cfg_full_ls), "Full cosine + LS")

    def test_record_to_row_keeps_nodim_label(self):
        record = es.ExperimentRecord(
            name="exp_nodim",
            path=Path("exp_nodim.log"),
            exp_id=73,
            era="exp46_73",
            has_eval=True,
            val_acc=49.5,
            val_loss=0.55,
            step=15000,
            recall={"W": 96.0, "A": 82.0, "S": 50.0, "D": 80.0},
            specificity={"W": 55.0, "A": 80.0, "S": 95.0, "D": 78.0},
            config={
                "model_dim": math.nan,
                "num_layers": 6.0,
                "dropout": 0.2,
                "max_iters": 15000.0,
                "lr_decay_iters": 5000.0,
                "label_smoothing": 0.01,
                "batch_size": 512.0,
                "seed": 42.0,
            },
        )

        row = bd.record_to_row(record)
        self.assertEqual(row["model_dim"], "nodim")
        self.assertEqual(row["group"], "Fast/short cosine + LS")

    def test_build_payload_uses_data_ranges_not_zero_locked(self):
        rows = [
            {
                "name": "exp1",
                "has_eval": True,
                "group": "Fast/short cosine",
                "val_acc": 48.1,
                "val_loss": 0.62,
                "W_rec": 95.0,
                "W_spec": 62.0,
                "A_rec": 79.0,
                "A_spec": 81.0,
                "S_rec": 51.0,
                "S_spec": 95.0,
                "D_rec": 80.0,
                "D_spec": 79.0,
                "model_dim": 64,
                "layers": 6,
                "dropout": 0.2,
                "batch_size": 512,
                "max_iters": 15000,
                "lr_decay_iters": 5000,
                "label_smoothing": 0.01,
                "seed": 42,
                "config_notes": "64d 6L",
            },
            {
                "name": "exp2",
                "has_eval": True,
                "group": "Full cosine",
                "val_acc": 45.3,
                "val_loss": 0.71,
                "W_rec": 97.0,
                "W_spec": 58.0,
                "A_rec": 74.0,
                "A_spec": 78.0,
                "S_rec": 33.0,
                "S_spec": 97.0,
                "D_rec": 77.0,
                "D_spec": 76.0,
                "model_dim": 64,
                "layers": 6,
                "dropout": 0.35,
                "batch_size": 256,
                "max_iters": 15000,
                "lr_decay_iters": 15000,
                "label_smoothing": 0.0,
                "seed": 1337,
                "config_notes": "64d 6L",
            },
        ]

        payload = bd.build_payload(rows)
        self.assertIn("https://cdn.plot.ly/plotly-2.35.2.min.js", bd.build_html(payload))
        self.assertGreater(payload["ranges"]["a"]["x"][0], 0.0)
        self.assertGreater(payload["ranges"]["d"]["y"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
