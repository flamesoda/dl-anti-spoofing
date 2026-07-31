import unittest

import torch
from torch import nn

from src.datasets import BalancedClassSampler
from src.model import MaxFeatureMap, STCLightCNN


class LCNNTest(unittest.TestCase):
    def test_balanced_sampler_balances_every_batch(self):
        labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
        sampler = BalancedClassSampler(labels=labels, batch_size=4)
        sampled = list(sampler)
        for start in range(0, len(sampled), 4):
            batch_labels = labels[sampled[start : start + 4]]
            self.assertEqual(int((batch_labels == 0).sum()), 2)
            self.assertEqual(int((batch_labels == 1).sum()), 2)

    def test_mfm_pairs_channel_halves(self):
        inputs = torch.tensor([[[[1.0]], [[4.0]], [[3.0]], [[2.0]]]])
        output = MaxFeatureMap(dim=1)(inputs)
        expected = torch.tensor([[[[3.0]], [[4.0]]]])
        torch.testing.assert_close(output, expected)

    def test_output_contract_and_dropout_order(self):
        model = STCLightCNN().eval()
        self.assertIsInstance(model.dropout, nn.Dropout)
        self.assertIsInstance(model.final_batch_norm, nn.BatchNorm1d)
        with torch.no_grad():
            output = model(torch.zeros(1, 1, 863, 600))
        self.assertEqual(tuple(output["logits"].shape), (1, 2))
        self.assertEqual(tuple(output["scores"].shape), (1,))
        torch.testing.assert_close(
            output["scores"], output["logits"][:, 1] - output["logits"][:, 0]
        )


if __name__ == "__main__":
    unittest.main()
