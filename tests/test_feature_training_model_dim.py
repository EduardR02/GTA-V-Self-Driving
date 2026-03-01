import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import feature_training
from feature_training import HeadOnlyModel, _resolve_num_workers, create_arg_parser, save_checkpoint


class FeatureTrainingModelDimTests(unittest.TestCase):
    def test_parser_defaults_model_dim_to_none(self):
        parser = create_arg_parser()

        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x"])

        self.assertIsNone(args.model_dim)

    def test_parser_accepts_custom_model_dim(self):
        parser = create_arg_parser()

        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x", "--model_dim", "96"])

        self.assertEqual(args.model_dim, 96)

    def test_parser_defaults_num_workers_to_zero_for_windows_memmap_safety(self):
        parser = create_arg_parser()

        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x"])

        self.assertEqual(args.num_workers, 0)

    def test_parser_accepts_fp8_mode_argument(self):
        parser = create_arg_parser()

        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x", "--fp8", "on"])

        self.assertEqual(args.fp8, "on")

    def test_auto_num_workers_defaults_to_zero_on_windows_cuda(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x"])
        self.assertEqual(args.num_workers, 0)

        with mock.patch("feature_training.os.cpu_count", return_value=12):
            resolved = _resolve_num_workers(args, torch.device("cuda"))

        self.assertEqual(resolved, 0)

    def test_explicit_num_workers_zero_disables_auto_worker_selection(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x", "--num_workers", "0"])

        with mock.patch("feature_training.os.cpu_count", return_value=12):
            resolved = _resolve_num_workers(args, torch.device("cuda"), explicit_num_workers=True)

        self.assertEqual(resolved, 0)

    def test_main_enables_fp8_before_torch_compile(self):
        call_order = []

        class _FakeDataset:
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                del idx
                return torch.zeros((2, 8), dtype=torch.float32), torch.zeros((1, 4), dtype=torch.float32)

        class _FakeLoader:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def __len__(self):
                return 1

            def __iter__(self):
                yield torch.zeros((1, 2, 8), dtype=torch.float32), torch.zeros((1, 1, 4), dtype=torch.float32)

        class _FakeOptimizer:
            def __init__(self):
                self.param_groups = []

            def state_dict(self):
                return {}

        class _FakeHeadOnlyModel(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                del args, kwargs
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def to(self, *args, **kwargs):
                del args, kwargs
                return self

            def configure_optimizers(self, *args, **kwargs):
                del args, kwargs
                return _FakeOptimizer()

            def forward(self, x, labels=None):
                del labels
                logits = torch.zeros((x.shape[0], 4), dtype=torch.float32)
                loss = torch.zeros((), dtype=torch.float32, requires_grad=True)
                return logits, loss

        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_dir = Path(tmp_dir) / "features"
            out_dir = Path(tmp_dir) / "out"
            feature_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(feature_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "embed_dim": 8,
                        "num_tokens": 2,
                        "sequence_len": 1,
                        "sequence_stride": 1,
                        "label_shift": 0,
                        "train_split": 0.9,
                    },
                    f,
                )

            def _fake_enable_fp8(model, **kwargs):
                del kwargs
                call_order.append("fp8")
                return model, SimpleNamespace(enabled=True, reason="enabled")

            def _fake_compile(model):
                call_order.append("compile")
                return model

            argv = [
                "feature_training.py",
                "--feature_dir",
                str(feature_dir),
                "--out_dir",
                str(out_dir),
                "--save_name",
                "test.pt",
                "--metrics_name",
                "test.png",
                "--batch_size",
                "1",
                "--max_iters",
                "-1",
                "--lr_decay_iters",
                "1",
                "--warmup_iters",
                "1",
                "--eval_interval",
                "100",
                "--eval_iters",
                "1",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--fp8",
                "on",
            ]

            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(feature_training, "PrecomputedFeatureDataset", return_value=_FakeDataset()):
                    with mock.patch.object(feature_training, "DataLoader", _FakeLoader):
                        with mock.patch.object(feature_training, "HeadOnlyModel", _FakeHeadOnlyModel):
                            with mock.patch.object(feature_training, "maybe_enable_fp8", side_effect=_fake_enable_fp8):
                                with mock.patch.object(feature_training.torch, "compile", side_effect=_fake_compile):
                                    with mock.patch.object(feature_training, "save_checkpoint"):
                                        with mock.patch.object(feature_training, "plot_metrics"):
                                            feature_training.main()

        self.assertGreaterEqual(len(call_order), 2)
        self.assertEqual(call_order[0], "fp8")
        self.assertEqual(call_order[1], "compile")

    def test_default_model_dim_keeps_projection_disabled(self):
        model = HeadOnlyModel(embed_dim=128, model_dim=None, num_heads=4, num_layers=1, max_seq_len=32)
        features = torch.randn(2, 6, 128)
        labels = torch.zeros(2, 1, 4)

        logits, loss = model(features, labels=labels)

        self.assertIsNone(model.projection)
        self.assertEqual(model.cls_token.shape[-1], 128)
        self.assertEqual(model.fc_head.in_features, 128)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertIsNotNone(loss)

    def test_custom_model_dim_enables_projection_and_forward(self):
        model = HeadOnlyModel(embed_dim=128, model_dim=96, num_heads=4, num_layers=1, max_seq_len=32)
        features = torch.randn(3, 8, 128)
        labels = torch.zeros(3, 1, 4)

        logits, loss = model(features, labels=labels)

        self.assertIsInstance(model.projection, torch.nn.Linear)
        projection = model.projection
        assert projection is not None
        self.assertEqual(projection.in_features, 128)
        self.assertEqual(projection.out_features, 96)
        self.assertEqual(model.cls_token.shape[-1], 96)
        self.assertEqual(model.fc_head.in_features, 96)
        self.assertEqual(tuple(logits.shape), (3, 4))
        self.assertIsNotNone(loss)

    def test_model_dim_downprojection_shapes_match_expected(self):
        model = HeadOnlyModel(embed_dim=128, model_dim=64, num_heads=4, num_layers=1, max_seq_len=32)
        features = torch.randn(2, 7, 128)

        projection = model.projection
        assert projection is not None
        projected = projection(features)
        logits, _ = model(features, labels=None)

        self.assertEqual(tuple(projected.shape), (2, 7, 64))
        self.assertEqual(model.transformer.norm.weight.shape[0], 64)
        self.assertEqual(model.fc_head.in_features, 64)
        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_model_dim_none_keeps_original_no_projection_path(self):
        model = HeadOnlyModel(embed_dim=96, model_dim=None, num_heads=4, num_layers=1, max_seq_len=32)
        features = torch.randn(2, 5, 96)

        logits, _ = model(features, labels=None)

        self.assertIsNone(model.projection)
        self.assertEqual(model.transformer.norm.weight.shape[0], 96)
        self.assertEqual(model.cls_token.shape[-1], 96)
        self.assertEqual(model.fc_head.in_features, 96)
        self.assertEqual(tuple(logits.shape), (2, 4))

    def test_from_checkpoint_loads_projection_weights_when_model_dim_set(self):
        source = HeadOnlyModel(embed_dim=128, model_dim=96, num_heads=4, num_layers=1, max_seq_len=32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "head.pt"
            torch.save({"model": source.state_dict()}, ckpt_path)

            loaded = HeadOnlyModel.from_checkpoint(
                checkpoint_path=str(ckpt_path),
                embed_dim=128,
                model_dim=96,
                num_classes=4,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
                max_seq_len=32,
            )

        assert loaded.projection is not None
        assert source.projection is not None
        self.assertTrue(torch.equal(loaded.projection.weight, source.projection.weight))

    def test_save_checkpoint_keeps_projection_weights_when_model_dim_set(self):
        model = HeadOnlyModel(embed_dim=128, model_dim=96, num_heads=4, num_layers=1, max_seq_len=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = SimpleNamespace(model_dim=96)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "head.pt"
            save_checkpoint(model, optimizer, args, iter_num=7, best_val_loss=0.5, out_path=ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")

        self.assertIn("projection.weight", ckpt["model"])
        self.assertIn("projection.bias", ckpt["model"])
        self.assertEqual(ckpt["config"]["model_dim"], 96)

    def test_save_checkpoint_dequantizes_float8_tensors(self):
        float8_dtype = getattr(torch, "float8_e4m3fn", None)
        if float8_dtype is None:
            self.skipTest("float8 dtype not available in this torch build")

        class _FakeFloat8Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def state_dict(self, *args, **kwargs):
                del args, kwargs
                return {
                    "transformer.float8_weight": torch.ones((8, 8), dtype=float8_dtype),
                    "cls_token": torch.zeros((1, 1, 8), dtype=torch.float32),
                }

        model = _FakeFloat8Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = SimpleNamespace(dtype="bfloat16")

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "head.pt"
            save_checkpoint(model, optimizer, args, iter_num=1, best_val_loss=0.1, out_path=ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu")

        self.assertEqual(ckpt["model"]["transformer.float8_weight"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
