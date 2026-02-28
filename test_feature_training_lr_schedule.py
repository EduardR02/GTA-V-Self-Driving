import math
import unittest

from feature_training import create_arg_parser, get_lr


def _legacy_cosine_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


class FeatureTrainingLRScheduleTests(unittest.TestCase):
    def test_parser_defaults_lr_restart_cycles_to_one(self):
        parser = create_arg_parser()

        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x"])

        self.assertEqual(args.lr_restart_cycles, 1)

    def test_parser_accepts_custom_lr_restart_cycles(self):
        parser = create_arg_parser()

        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x", "--lr_restart_cycles", "3"])

        self.assertEqual(args.lr_restart_cycles, 3)

    def test_parser_accepts_model_dim_and_loader_tuning_args(self):
        parser = create_arg_parser()

        args = parser.parse_args(
            [
                "--feature_dir",
                "temp/precomputed_full_10x",
                "--model_dim",
                "64",
                "--prefetch_factor",
                "3",
                "--shuffle_chunk_size",
                "8192",
            ]
        )

        self.assertEqual(args.model_dim, 64)
        self.assertEqual(args.prefetch_factor, 3)
        self.assertEqual(args.shuffle_chunk_size, 8192)

    def test_get_lr_with_one_cycle_matches_legacy_schedule(self):
        warmup_iters = 100
        lr_decay_iters = 1000
        learning_rate = 5e-5
        min_lr = 3e-6

        for it in [0, 1, 99, 100, 250, 500, 750, 1000, 1001]:
            expected = _legacy_cosine_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)
            actual = get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr, lr_restart_cycles=1)
            self.assertAlmostEqual(actual, expected, places=12)

    def test_get_lr_with_restart_cycles_restarts_each_cycle(self):
        warmup_iters = 100
        lr_decay_iters = 1000
        learning_rate = 5e-5
        min_lr = 3e-6

        self.assertAlmostEqual(
            get_lr(100, warmup_iters, lr_decay_iters, learning_rate, min_lr, lr_restart_cycles=3),
            learning_rate,
            places=12,
        )
        self.assertAlmostEqual(
            get_lr(400, warmup_iters, lr_decay_iters, learning_rate, min_lr, lr_restart_cycles=3),
            learning_rate,
            places=12,
        )
        self.assertAlmostEqual(
            get_lr(700, warmup_iters, lr_decay_iters, learning_rate, min_lr, lr_restart_cycles=3),
            learning_rate,
            places=12,
        )

        before_restart = get_lr(399, warmup_iters, lr_decay_iters, learning_rate, min_lr, lr_restart_cycles=3)
        at_restart = get_lr(400, warmup_iters, lr_decay_iters, learning_rate, min_lr, lr_restart_cycles=3)
        self.assertLess(before_restart, at_restart)

        self.assertAlmostEqual(
            get_lr(1000, warmup_iters, lr_decay_iters, learning_rate, min_lr, lr_restart_cycles=3),
            min_lr,
            places=12,
        )
        self.assertAlmostEqual(
            get_lr(1001, warmup_iters, lr_decay_iters, learning_rate, min_lr, lr_restart_cycles=3),
            min_lr,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
