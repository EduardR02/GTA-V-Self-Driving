import unittest

import torch

from feature_training import FINAL_POS_WEIGHT, create_arg_parser, get_effective_pos_weight


class FeatureTrainingPosWeightTests(unittest.TestCase):
    def test_parser_defaults_ad_and_s_scales_to_one(self):
        parser = create_arg_parser()

        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x"])

        self.assertEqual(args.a_pos_weight_scale, 1.0)
        self.assertEqual(args.s_pos_weight_scale, 1.0)
        self.assertEqual(args.d_pos_weight_scale, 1.0)

    def test_parser_accepts_custom_ad_and_s_scales(self):
        parser = create_arg_parser()

        args = parser.parse_args([
            "--feature_dir",
            "temp/precomputed_full_10x",
            "--a_pos_weight_scale",
            "0.85",
            "--s_pos_weight_scale",
            "1.5",
            "--d_pos_weight_scale",
            "0.7",
        ])

        self.assertEqual(args.a_pos_weight_scale, 0.85)
        self.assertEqual(args.s_pos_weight_scale, 1.5)
        self.assertEqual(args.d_pos_weight_scale, 0.7)

    def test_effective_pos_weight_defaults_leave_weights_unchanged(self):
        base = FINAL_POS_WEIGHT.clone()

        scaled = get_effective_pos_weight(base)

        self.assertTrue(torch.equal(base, FINAL_POS_WEIGHT))
        self.assertTrue(torch.equal(scaled, base))

    def test_effective_pos_weight_scales_a_d_and_s_classes(self):
        base = FINAL_POS_WEIGHT.clone()

        scaled = get_effective_pos_weight(
            base,
            s_pos_weight_scale=1.5,
            a_pos_weight_scale=0.85,
            d_pos_weight_scale=0.7,
        )

        self.assertTrue(torch.equal(base, FINAL_POS_WEIGHT))
        self.assertAlmostEqual(float(scaled[0]), float(base[0]))
        self.assertAlmostEqual(float(scaled[1]), float(base[1] * 0.85))
        self.assertAlmostEqual(float(scaled[2]), float(base[2] * 1.5))
        self.assertAlmostEqual(float(scaled[3]), float(base[3] * 0.7))


if __name__ == "__main__":
    unittest.main()
