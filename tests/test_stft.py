import unittest

import torch

from src.transforms import LogPowerSpectrogram


class STFTTest(unittest.TestCase):
    def test_shape_and_finiteness(self):
        transform = LogPowerSpectrogram()
        waveform = torch.zeros(16000)
        output = transform(waveform)
        self.assertEqual(tuple(output.shape), (1, 863, 600))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(transform.hop_length, 130)


if __name__ == "__main__":
    unittest.main()
