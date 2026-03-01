import unittest
from unittest import mock

import torch

import training_new


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, labels=None):
        return self.linear(x), None


class TrainingNewFP8InferenceTests(unittest.TestCase):
    @mock.patch("training_new.torch.load")
    @mock.patch("training_new.maybe_enable_fp8")
    @mock.patch("training_new.Dinov3ForTimeSeriesClassification")
    def test_sample_only_load_uses_fp8_helper_for_inference(self, model_ctor, maybe_enable, torch_load):
        dummy_model = _DummyModel()
        model_ctor.return_value = dummy_model
        maybe_enable.return_value = (dummy_model, mock.Mock(enabled=True))
        torch_load.return_value = {
            "model": {},
            "iter_num": 0,
            "best_val_loss": 1.0,
            "config": {},
        }

        with mock.patch.object(training_new, "compile", False), mock.patch.object(training_new.config, "use_dinov3", True):
            model = training_new.load_model(sample_only=True)

        self.assertIs(model, dummy_model)
        maybe_enable.assert_called_once()
        _, kwargs = maybe_enable.call_args
        self.assertEqual(kwargs["mode"], training_new.fp8)
        self.assertTrue(kwargs["use_dinov3"])


if __name__ == "__main__":
    unittest.main()
