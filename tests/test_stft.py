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
        expected_floor = torch.tensor(transform.power_floor).log()
        torch.testing.assert_close(output, torch.full_like(output, expected_floor))

    def test_short_waveform_is_repeated_cyclically(self):
        transform = LogPowerSpectrogram(
            sample_rate=10,
            n_fft=4,
            win_length=4,
            hop_seconds=0.2,
            max_frames=5,
            short_input_mode="repeat",
        )
        waveform = torch.tensor([1.0, 2.0, 3.0])
        prepared = transform._prepare_short_waveform(waveform)
        expected = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0])
        torch.testing.assert_close(prepared, expected)

    def test_empty_waveform_is_rejected(self):
        transform = LogPowerSpectrogram()
        with self.assertRaises(ValueError):
            transform(torch.empty(0))


if __name__ == "__main__":
    unittest.main()
