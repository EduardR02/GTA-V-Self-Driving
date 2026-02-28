import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn as nn

from feature_training import AsyncPrefetcher, estimate_loss


class _MockLoader:
    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def __len__(self):
        return len(self._batches)


class _CountingLoader:
    def __init__(self, num_batches: int, batch_size: int = 2):
        self.num_batches = int(num_batches)
        self.batch_size = int(batch_size)
        self.yield_count = 0

    def __iter__(self):
        for _ in range(self.num_batches):
            self.yield_count += 1
            x = torch.zeros((self.batch_size, 3), dtype=torch.float32)
            y = torch.zeros((self.batch_size, 1, 4), dtype=torch.float32)
            yield x, y

    def __len__(self):
        return self.num_batches


class _ToyModel(nn.Module):
    def forward(self, features, labels=None):
        logits = torch.zeros((features.shape[0], 4), dtype=torch.float32, device=features.device)
        loss = torch.tensor(0.0, dtype=torch.float32, device=features.device)
        return logits, loss


class AsyncPrefetcherTests(unittest.TestCase):
    def test_prefetcher_returns_all_batches_from_loader(self):
        batches = [
            (
                torch.full((2, 3), float(i), dtype=torch.float32),
                torch.full((2, 1, 4), float(i), dtype=torch.float32),
            )
            for i in range(5)
        ]
        loader = _MockLoader(batches)
        prefetcher = AsyncPrefetcher(loader, device=torch.device("cpu"), non_blocking=False, pin_memory=False)

        got = list(prefetcher)

        self.assertEqual(len(got), len(batches))
        for expected, actual in zip(batches, got):
            x_expected, y_expected = expected
            x_actual, y_actual = actual
            self.assertTrue(torch.equal(x_expected, x_actual))
            self.assertTrue(torch.equal(y_expected, y_actual))


class EstimateLossTests(unittest.TestCase):
    def test_estimate_loss_consumes_only_eval_iters_batches(self):
        train_loader = _CountingLoader(num_batches=5)
        val_loader = _CountingLoader(num_batches=5)
        args = SimpleNamespace(eval_iters=2)

        estimate_loss(
            model=_ToyModel(),
            train_eval_loader=train_loader,
            val_loader=val_loader,
            args=args,
            ctx=nullcontext(),
            device=torch.device("cpu"),
        )

        self.assertEqual(train_loader.yield_count, 2)
        self.assertEqual(val_loader.yield_count, 2)


if __name__ == "__main__":
    unittest.main()
