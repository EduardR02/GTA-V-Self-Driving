import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import run_experiments


class RunExperimentsTests(unittest.TestCase):
    def test_parse_args_rejects_out_of_range_start_index(self):
        with self.assertRaises(SystemExit):
            run_experiments.parse_args(["--start_from", "0"])

    def test_parse_args_rejects_unknown_experiment_id(self):
        with self.assertRaises(SystemExit):
            run_experiments.parse_args(["--experiments", "9999"])

    def test_main_runs_only_requested_experiment_subset(self):
        fake_experiments = [
            {"name": "1_first", "desc": "first", "args": ["--tag", "1"]},
            {"name": "2_second", "desc": "second", "args": ["--tag", "2"]},
            {"name": "3_third", "desc": "third", "args": ["--tag", "3"]},
        ]
        commands = []

        def fake_run(cmd, stdout, stderr, timeout):
            commands.append(cmd)
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                with mock.patch.object(run_experiments, "EXPERIMENTS", fake_experiments):
                    with mock.patch.object(run_experiments, "BASE_ARGS", []):
                        with mock.patch("run_experiments.subprocess.run", side_effect=fake_run):
                            run_experiments.main(["--start_from", "2"])
            finally:
                os.chdir(cwd)

        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][-1], "2")
        self.assertEqual(commands[1][-1], "3")

    def test_main_runs_only_requested_experiment_ids(self):
        fake_experiments = [
            {"id": 46, "name": "46_one", "desc": "one", "args": ["--tag", "46"]},
            {"id": 47, "name": "47_two", "desc": "two", "args": ["--tag", "47"]},
            {"id": 48, "name": "48_three", "desc": "three", "args": ["--tag", "48"]},
        ]
        commands = []

        def fake_run(cmd, stdout, stderr, timeout):
            commands.append(cmd)
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                with mock.patch.object(run_experiments, "EXPERIMENTS", fake_experiments):
                    with mock.patch.object(run_experiments, "BASE_ARGS", []):
                        with mock.patch("run_experiments.subprocess.run", side_effect=fake_run):
                            run_experiments.main(["--experiments", "48", "46"])
            finally:
                os.chdir(cwd)

        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][-1], "48")
        self.assertEqual(commands[1][-1], "46")

    def test_new_experiments_define_required_lr_decay_and_overrides(self):
        experiments_by_id = {exp.get("id"): exp for exp in run_experiments.EXPERIMENTS}

        self.assertIn(46, experiments_by_id)
        self.assertIn(47, experiments_by_id)
        self.assertIn(48, experiments_by_id)

        for exp_id in [46, 47, 48]:
            args = experiments_by_id[exp_id]["args"]
            self.assertIn("--lr_decay_iters", args)
            self.assertEqual(args[args.index("--lr_decay_iters") + 1], "15000")

        exp47_args = experiments_by_id[47]["args"]
        self.assertIn("--lr_restart_cycles", exp47_args)
        self.assertEqual(exp47_args[exp47_args.index("--lr_restart_cycles") + 1], "3")

        exp48_args = experiments_by_id[48]["args"]
        self.assertIn("--batch_size", exp48_args)
        self.assertEqual(exp48_args[exp48_args.index("--batch_size") + 1], "512")
        self.assertIn("--learning_rate", exp48_args)
        self.assertEqual(exp48_args[exp48_args.index("--learning_rate") + 1], "1e-4")
        self.assertIn("--muon_lr", exp48_args)
        self.assertEqual(exp48_args[exp48_args.index("--muon_lr") + 1], "0.003")


if __name__ == "__main__":
    unittest.main()
