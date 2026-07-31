import torch
from torch import nn
from torch.nn import functional as F


class LogPowerSpectrogram(nn.Module):
    """STC-style FFT front end for ASVspoof 2019 LA recordings.

    The transform uses a 1724-sample Blackman window and a hop of 0.0081
    seconds. It returns a fixed-size ``[1, frequency, time]`` log-power
    spectrogram. Long recordings are cropped and short recordings are repeated
    cyclically in the waveform domain before the STFT is computed. Repetition
    avoids introducing a conspicuous block of literal zeros after the log.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=1724,
        win_length=1724,
        hop_seconds=0.0081,
        max_frames=600,
        crop_mode="start",
        short_input_mode="repeat",
        center=True,
        power_floor=1e-12,
    ):
        super().__init__()
        if crop_mode not in {"start", "center", "random"}:
            raise ValueError("crop_mode must be 'start', 'center', or 'random'")
        if short_input_mode not in {"repeat", "zero"}:
            raise ValueError("short_input_mode must be 'repeat' or 'zero'")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = int(round(hop_seconds * sample_rate))
        self.max_frames = max_frames
        self.crop_mode = crop_mode
        self.short_input_mode = short_input_mode
        self.center = center
        self.power_floor = power_floor

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("win_length cannot be larger than n_fft")
        if self.power_floor <= 0:
            raise ValueError("power_floor must be positive")

        self.register_buffer(
            "window",
            torch.blackman_window(win_length, periodic=True),
            persistent=False,
        )

    @property
    def n_frequency_bins(self):
        return self.n_fft // 2 + 1

    @property
    def minimum_waveform_samples(self):
        """Minimum waveform length that produces ``max_frames`` STFT frames."""
        if self.center:
            # With centered STFT, PyTorch pads n_fft // 2 samples on each side,
            # giving 1 + floor(n_samples / hop_length) frames. Reflect padding
            # additionally requires an input longer than n_fft // 2.
            frame_length = (self.max_frames - 1) * self.hop_length
            return max(frame_length, self.n_fft // 2 + 1)
        return self.n_fft + (self.max_frames - 1) * self.hop_length

    def _prepare_short_waveform(self, waveform):
        target_length = self.minimum_waveform_samples
        current_length = waveform.shape[-1]
        if current_length == 0:
            raise ValueError("Cannot compute an STFT for an empty waveform")
        if current_length >= target_length:
            return waveform

        if self.short_input_mode == "repeat":
            repeats = (target_length + current_length - 1) // current_length
            return waveform.repeat(repeats)[:target_length]
        return F.pad(waveform, (0, target_length - current_length))

    def _crop(self, features):
        n_frames = features.shape[-1]
        if n_frames <= self.max_frames:
            return features

        if self.crop_mode == "random" and self.training:
            start = torch.randint(0, n_frames - self.max_frames + 1, size=(1,)).item()
        elif self.crop_mode == "center":
            start = (n_frames - self.max_frames) // 2
        else:
            start = 0
        return features[..., start : start + self.max_frames]

    def forward(self, waveform):
        """Convert a mono waveform to a fixed-size log-power spectrogram."""
        if waveform.ndim == 2:
            if waveform.shape[0] != 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
        if waveform.ndim != 1:
            raise ValueError(
                f"Expected waveform with shape [time] or [channels, time], got "
                f"{tuple(waveform.shape)}"
            )
        waveform = self._prepare_short_waveform(waveform)

        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=self.center,
            return_complex=True,
        )
        features = spectrum.abs().square().clamp_min(self.power_floor).log()
        features = self._crop(features)

        expected_shape = (self.n_frequency_bins, self.max_frames)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected STFT shape {tuple(features.shape)}; "
                f"expected {expected_shape}"
            )
        return features.unsqueeze(0)

    def extra_repr(self):
        hop_seconds = self.hop_length / self.sample_rate
        return (
            f"sample_rate={self.sample_rate}, n_fft={self.n_fft}, "
            f"win_length={self.win_length}, hop_length={self.hop_length} "
            f"({hop_seconds:.6f}s), max_frames={self.max_frames}, "
            f"crop_mode='{self.crop_mode}', short_input_mode="
            f"'{self.short_input_mode}'"
        )
