import unittest

import numpy as np
import torch

from src.metrics import EqualErrorRate, compute_eer


class EERTest(unittest.TestCase):
    def test_perfect_separation(self):
        eer, threshold = compute_eer(np.array([2.0, 3.0]), np.array([-2.0, -1.0]))
        self.assertEqual(eer, 0.0)
        self.assertTrue(np.isfinite(threshold))

    def test_epoch_metric_returns_percentage(self):
        metric = EqualErrorRate()
        metric.update(
            scores=torch.tensor([2.0, -2.0, 3.0, -1.0]),
            labels=torch.tensor([1, 0, 1, 0]),
        )
        self.assertEqual(metric.compute(), 0.0)

    def test_non_finite_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_eer(np.array([1.0, np.nan]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
