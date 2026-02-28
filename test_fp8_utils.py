import unittest
from unittest import mock

import torch

from fp8_utils import maybe_enable_fp8, normalize_fp8_mode


class _FakeFloat8Module:
    class Float8LinearRecipeName:
        ROWWISE = "rowwise"
        TENSORWISE = "tensorwise"

    class Float8LinearConfig:
        @staticmethod
        def from_recipe_name(recipe_name):
            return {"recipe": recipe_name}

    @staticmethod
    def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
        module._fp8_test_config = config
        module._fp8_selected = []
        if module_filter_fn is not None:
            for name, submodule in module.named_modules():
                if not name:
                    continue
                try:
                    if module_filter_fn(submodule, name):
                        module._fp8_selected.append(name)
                except Exception:
                    continue
        return module


class _FakeDinoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov3 = object()
        self.projection = torch.nn.Linear(768, 128, bias=False)
        self.fc_head = torch.nn.Linear(128, 4, bias=False)

    def forward(self, x):
        return x


class _FakeDinoIdentityProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov3 = object()
        self.projection = torch.nn.Identity()
        self.fc_head = torch.nn.Linear(128, 4, bias=False)

    def forward(self, x):
        return x


class FP8UtilsTests(unittest.TestCase):
    def test_normalize_fp8_mode_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            normalize_fp8_mode("invalid")

    def test_auto_mode_skips_non_dinov3_runs(self):
        model = torch.nn.Linear(4, 4)
        logs = []

        _, state = maybe_enable_fp8(
            model,
            mode="auto",
            use_dinov3=False,
            device="cuda",
            logger=logs.append,
        )

        self.assertFalse(state.enabled)
        self.assertIn("non-DINOv3", state.reason)
        self.assertTrue(any("disabled" in line for line in logs))

    def test_auto_mode_skips_cpu_device(self):
        model = torch.nn.Linear(4, 4)

        _, state = maybe_enable_fp8(
            model,
            mode="auto",
            use_dinov3=True,
            device="cpu",
            logger=None,
        )

        self.assertFalse(state.enabled)
        self.assertIn("device 'cpu'", state.reason)

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(8, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_auto_mode_skips_unsupported_capability(self, *_mocks):
        model = torch.nn.Linear(4, 4)

        _, state = maybe_enable_fp8(
            model,
            mode="auto",
            use_dinov3=True,
            device="cuda",
            logger=None,
        )

        self.assertFalse(state.enabled)
        self.assertEqual(state.capability, (8, 0))
        self.assertIn("below required", state.reason)

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(9, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_on_mode_raises_when_torchao_import_fails(self, *_mocks):
        model = torch.nn.Linear(4, 4)

        with self.assertRaisesRegex(RuntimeError, "mode='on' requested"):
            maybe_enable_fp8(
                model,
                mode="on",
                use_dinov3=True,
                device="cuda",
                logger=None,
                import_module=mock.Mock(side_effect=ImportError("no torchao")),
            )

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(8, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_on_mode_raises_for_unsupported_capability(self, *_mocks):
        model = torch.nn.Linear(4, 4)

        with self.assertRaisesRegex(RuntimeError, "below required"):
            maybe_enable_fp8(
                model,
                mode="on",
                use_dinov3=True,
                device="cuda",
                logger=None,
            )

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(9, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_auto_mode_enables_fp8_with_torchao_conversion(self, *_mocks):
        model = torch.nn.Linear(32, 32)
        logs = []

        converted, state = maybe_enable_fp8(
            model,
            mode="auto",
            use_dinov3=True,
            device="cuda",
            logger=logs.append,
            import_module=lambda _: _FakeFloat8Module,
        )

        self.assertTrue(state.enabled)
        self.assertEqual(state.recipe, "rowwise")
        self.assertIs(converted, model)
        self.assertEqual(model._fp8_test_config, {"recipe": "rowwise"})
        self.assertTrue(any("enabled" in line for line in logs))

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(12, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_auto_mode_no_eligible_modules_is_not_enabled(self, *_mocks):
        model = _FakeDinoIdentityProjection()

        _, state = maybe_enable_fp8(
            model,
            mode="auto",
            use_dinov3=True,
            device="cuda",
            logger=None,
            import_module=lambda _: _FakeFloat8Module,
        )

        self.assertFalse(state.enabled)
        self.assertIn("no eligible modules", state.reason)

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(12, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_on_mode_raises_when_projection_has_no_eligible_modules(self, *_mocks):
        model = _FakeDinoIdentityProjection()

        with self.assertRaisesRegex(RuntimeError, "no eligible modules"):
            maybe_enable_fp8(
                model,
                mode="on",
                use_dinov3=True,
                device="cuda",
                logger=None,
                import_module=lambda _: _FakeFloat8Module,
            )

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(12, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_blackwell_strategy_converts_all_compatible_and_skips_small_head(self, *_mocks):
        model = _FakeDinoModel()
        logs = []

        converted, state = maybe_enable_fp8(
            model,
            mode="auto",
            use_dinov3=True,
            device="cuda",
            logger=logs.append,
            import_module=lambda _: _FakeFloat8Module,
        )

        self.assertTrue(state.enabled)
        self.assertIs(converted, model)
        self.assertEqual(model._fp8_selected, ["projection"])
        self.assertTrue(any("conversion plan selected: tensorwise_all_linear" in line for line in logs))
        self.assertTrue(any("min_dim_lt_16" in line for line in logs))

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(9, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_h100_strategy_selects_rowwise_all_linear_plan(self, *_mocks):
        model = _FakeDinoModel()
        logs = []

        _, state = maybe_enable_fp8(
            model,
            mode="auto",
            use_dinov3=True,
            device="cuda",
            logger=logs.append,
            import_module=lambda _: _FakeFloat8Module,
        )

        self.assertTrue(state.enabled)
        self.assertEqual(state.recipe, "rowwise")
        self.assertTrue(any("conversion plan selected: rowwise_all_linear" in line for line in logs))

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(12, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_auto_mode_falls_to_next_plan_when_first_plan_fails(self, *_mocks):
        class _FallbackFloat8Module(_FakeFloat8Module):
            @staticmethod
            def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
                if isinstance(config, dict) and config.get("recipe") == "tensorwise":
                    raise RuntimeError("tensorwise failed")
                return _FakeFloat8Module.convert_to_float8_training(module, config=config, module_filter_fn=module_filter_fn)

        model = _FakeDinoModel()
        logs = []

        _, state = maybe_enable_fp8(
            model,
            mode="auto",
            use_dinov3=True,
            device="cuda",
            logger=logs.append,
            import_module=lambda _: _FallbackFloat8Module,
        )

        self.assertTrue(state.enabled)
        self.assertEqual(state.recipe, "rowwise")
        self.assertTrue(any("tensorwise_all_linear failed" in line for line in logs))

    @mock.patch("fp8_utils.torch.cuda.get_device_capability", return_value=(12, 0))
    @mock.patch("fp8_utils.torch.cuda.current_device", return_value=0)
    @mock.patch("fp8_utils.torch.cuda.is_available", return_value=True)
    def test_recipe_label_is_default_when_recipe_config_unavailable(self, *_mocks):
        class _NoConfigFloat8Module:
            Float8LinearRecipeName = _FakeFloat8Module.Float8LinearRecipeName

            @staticmethod
            def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
                return _FakeFloat8Module.convert_to_float8_training(module, config=config, module_filter_fn=module_filter_fn)

        model = _FakeDinoModel()
        logs = []
        _, state = maybe_enable_fp8(
            model,
            mode="auto",
            use_dinov3=True,
            device="cuda",
            logger=logs.append,
            import_module=lambda _: _NoConfigFloat8Module,
        )

        self.assertTrue(state.enabled)
        self.assertEqual(state.recipe, "default")
        self.assertTrue(any("using default config" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
