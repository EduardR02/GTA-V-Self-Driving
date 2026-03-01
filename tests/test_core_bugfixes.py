import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch
import torch.nn as nn

import training_new
from dataloader import H5Dataset, TimeSeriesDataset
from feature_training import HeadOnlyModel, create_arg_parser, save_checkpoint
from model import (
    Dinov2ForTimeSeriesClassification,
    Dinov3ForTimeSeriesClassification,
    EfficientTransformerBlock,
)


def _write_h5(path: Path, num_samples: int):
    images = np.zeros((num_samples, 4, 4, 3), dtype=np.uint8)
    labels = np.zeros((num_samples, 7), dtype=np.int8)
    if num_samples > 0:
        labels[:, 0] = 1
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("labels", data=labels)


class _DummyDinov2Backbone(nn.Module):
    def __init__(self, embed_dim: int, patches_per_frame: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.patches_per_frame = patches_per_frame
        self.marker = nn.Parameter(torch.ones(1))
        self.grad_enabled_calls = []

    def get_intermediate_layers(
        self, x, reshape=False, return_class_token=False, norm=True
    ):
        del reshape, norm
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        batch = x.shape[0]
        patch_tokens = torch.ones(
            (batch, self.patches_per_frame, self.embed_dim),
            dtype=x.dtype,
            device=x.device,
        )
        if return_class_token:
            cls_tokens = torch.ones(
                (batch, self.embed_dim), dtype=x.dtype, device=x.device
            )
            return [(patch_tokens, cls_tokens)]
        return [patch_tokens]


class _DummyDinov3Backbone(nn.Module):
    def __init__(self, embed_dim: int, patches_per_frame: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.patches_per_frame = patches_per_frame
        self.marker = nn.Parameter(torch.ones(1))
        self.grad_enabled_calls = []

    def get_intermediate_layers(self, x, **kwargs):
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        batch = x.shape[0]
        patch_tokens = torch.ones(
            (batch, self.patches_per_frame, self.embed_dim),
            dtype=x.dtype,
            device=x.device,
        )
        if kwargs.get("return_class_token", False):
            cls_tokens = torch.ones(
                (batch, self.embed_dim), dtype=x.dtype, device=x.device
            )
            return [(patch_tokens, cls_tokens)]
        return [patch_tokens]


class _FakeTensor:
    def __init__(self):
        self.pin_memory_called = False
        self.to_kwargs = None

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, *args, **kwargs):
        del args
        self.to_kwargs = kwargs
        return self

    def pin_memory(self):
        self.pin_memory_called = True
        return self


class CoreBugFixTests(unittest.TestCase):
    def test_attention_scale_uses_inverse_sqrt_head_dim(self):
        block = EfficientTransformerBlock(
            hidden_size=64, num_heads=4, dropout=0.0, use_xformers=False, max_seq_len=16
        )
        self.assertAlmostEqual(block.scale, block.head_dim**-0.5)

    def test_feature_training_default_eval_iters_is_50(self):
        parser = create_arg_parser()
        args = parser.parse_args(["--feature_dir", "temp/precomputed_full_10x"])
        self.assertEqual(args.eval_iters, 50)

    def test_save_checkpoint_keeps_projection_weights(self):
        model = HeadOnlyModel(
            embed_dim=16,
            model_dim=8,
            num_heads=4,
            num_layers=1,
            max_seq_len=16,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "head.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                args=SimpleNamespace(model_dim=8),
                iter_num=1,
                best_val_loss=0.5,
                out_path=ckpt_path,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu")

        self.assertIn("projection.weight", ckpt["model"])
        self.assertIn("projection.bias", ckpt["model"])

    def test_lookup_table_filters_invalid_files_without_mutating_iteration_list(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_h5(root / "a_invalid.h5", num_samples=0)
            _write_h5(root / "b_valid.h5", num_samples=2)
            _write_h5(root / "c_invalid.h5", num_samples=0)

            dataset = H5Dataset(
                data_dirs=[str(root)],
                train_split=1.0,
                is_train=True,
                classifier_type="bce",
                flip_prob=0.0,
                warp_prob=0.0,
                zoom_prob=0.0,
                shift_labels=False,
            )

            self.assertEqual([Path(p).name for p in dataset.data_files], ["b_valid.h5"])
            self.assertEqual(dataset.lookup_table, [2])
            self.assertEqual(len(dataset._stuck_offsets), len(dataset.data_files))

    def test_h5dataset_reuses_lazy_file_handle_and_drops_handles_on_pickle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_h5(root / "sample.h5", num_samples=3)

            real_open = h5py.File
            open_calls = 0

            def counting_open(*args, **kwargs):
                nonlocal open_calls
                open_calls += 1
                return real_open(*args, **kwargs)

            with mock.patch("dataloader.h5py.File", side_effect=counting_open):
                dataset = H5Dataset(
                    data_dirs=[str(root)],
                    train_split=0.0,
                    is_train=False,
                    classifier_type="bce",
                    flip_prob=0.0,
                    warp_prob=0.0,
                    zoom_prob=0.0,
                    shift_labels=False,
                )
                self.assertEqual(open_calls, 1)

                dataset[0]
                dataset[0]

                self.assertEqual(open_calls, 2)
                self.assertEqual(len(dataset._file_handles), 1)
                self.assertEqual(dataset.__getstate__()["_file_handles"], {})
                dataset._close_file_handles()

    def test_timeseriesdataset_reuses_lazy_file_handle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_h5(root / "sample.h5", num_samples=5)

            real_open = h5py.File
            open_calls = 0

            def counting_open(*args, **kwargs):
                nonlocal open_calls
                open_calls += 1
                return real_open(*args, **kwargs)

            with mock.patch("dataloader.h5py.File", side_effect=counting_open):
                dataset = TimeSeriesDataset(
                    data_dirs=[str(root)],
                    train_split=0.0,
                    is_train=False,
                    classifier_type="bce",
                    flip_prob=0.0,
                    warp_prob=0.0,
                    zoom_prob=0.0,
                    sequence_len=2,
                    sequence_stride=1,
                    shift_labels=False,
                )
                self.assertEqual(open_calls, 1)

                dataset[0]
                dataset[0]

                self.assertEqual(open_calls, 2)
                self.assertEqual(len(dataset._file_handles), 1)
                dataset._close_file_handles()

    def test_dinov2_forward_uses_no_grad_when_backbone_frozen(self):
        model = Dinov2ForTimeSeriesClassification.__new__(
            Dinov2ForTimeSeriesClassification
        )
        nn.Module.__init__(model)

        model.patches_per_frame = 2
        model.embed_dim = 4
        model.return_class_token = False
        model.cls_option = "patches_only"
        model.projection = nn.Identity()
        model.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
        model.transformer = nn.Identity()
        model.fc_head = nn.Linear(4, 4, bias=False)
        model.loss_fct = nn.BCEWithLogitsLoss()
        model.num_classes = 4
        model.dinov2 = _DummyDinov2Backbone(embed_dim=4, patches_per_frame=2)

        x = torch.randn(2, 1, 3, 8, 8)

        model.backbone_frozen = True
        model(x)
        self.assertFalse(model.dinov2.grad_enabled_calls[-1])

        model.backbone_frozen = False
        model(x)
        self.assertTrue(model.dinov2.grad_enabled_calls[-1])

    def test_dinov3_forward_uses_no_grad_when_backbone_frozen(self):
        model = Dinov3ForTimeSeriesClassification.__new__(
            Dinov3ForTimeSeriesClassification
        )
        nn.Module.__init__(model)

        model.patches_per_frame = 2
        model.embed_dim = 4
        model.projection = nn.Identity()
        model.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
        model.transformer = nn.Identity()
        model.fc_head = nn.Linear(4, 4, bias=False)
        model.loss_fct = nn.BCEWithLogitsLoss()
        model.label_smoothing = 0.0
        model.dinov3 = _DummyDinov3Backbone(embed_dim=4, patches_per_frame=2)
        model._token_extractor = model._extract_tokens_patches_only_torch_hub
        model._forward_processor = model._forward_patches_only

        x = torch.randn(2, 1, 3, 8, 8)

        model.backbone_frozen = True
        model(x)
        self.assertFalse(model.dinov3.grad_enabled_calls[-1])

        model.backbone_frozen = False
        model(x)
        self.assertTrue(model.dinov3.grad_enabled_calls[-1])

    def test_training_get_batch_cuda_path_does_not_pin_memory_again(self):
        x = _FakeTensor()
        y = _FakeTensor()
        dataloader_iter = iter([(x, y)])

        with (
            mock.patch.object(training_new, "device_type", "cuda"),
            mock.patch.object(training_new, "device", "cpu"),
        ):
            x_out, y_out, y_cpu, _ = training_new.get_batch(
                dataloader_iter, split="train"
            )

        self.assertIs(x_out, x)
        self.assertIs(y_out, y)
        self.assertIs(y_cpu, y)
        self.assertFalse(x.pin_memory_called)
        self.assertFalse(y.pin_memory_called)
        self.assertEqual(x.to_kwargs, {"non_blocking": True})
        self.assertEqual(y.to_kwargs, {"non_blocking": True})

    def test_load_model_marks_backbone_frozen_when_fine_tune_disabled(self):
        class _DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dinov3_backbone = nn.Linear(4, 4)
                self.head = nn.Linear(4, 4)

            def configure_optimizers(
                self, weight_decay, learning_rate, betas, device_type, muon_lr=None
            ):
                del weight_decay, learning_rate, betas, device_type, muon_lr
                return torch.optim.SGD(self.parameters(), lr=0.01)

        dummy = _DummyModel()
        fp8_state = SimpleNamespace(enabled=False, reason="disabled", recipe=None)

        with (
            mock.patch.object(training_new, "init_from", "scratch"),
            mock.patch.object(training_new, "fine_tune", False),
            mock.patch.object(training_new, "compile", False),
            mock.patch.object(training_new, "device", "cpu"),
            mock.patch.object(training_new, "device_type", "cpu"),
            mock.patch.object(training_new, "dtype", "float32"),
            mock.patch.object(training_new.config, "use_dinov3", True),
            mock.patch.object(
                training_new, "Dinov3ForTimeSeriesClassification", return_value=dummy
            ),
            mock.patch.object(
                training_new, "maybe_enable_fp8", return_value=(dummy, fp8_state)
            ),
            mock.patch.object(
                training_new.torch.amp, "GradScaler", return_value=mock.Mock()
            ),
        ):
            loaded = training_new.load_model(sample_only=False)

        self.assertIs(loaded, dummy)
        self.assertTrue(loaded.backbone_frozen)
        self.assertFalse(loaded.dinov3_backbone.weight.requires_grad)
        self.assertTrue(loaded.head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
