import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    """MFM 2/1 activation from Light CNN.

    Pairs the first and second halves of the channel (or feature) dimension
    and keeps their element-wise maximum.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        n_features = inputs.shape[self.dim]
        if n_features % 2 != 0:
            raise ValueError(
                f"MFM requires an even dimension, got {n_features} on dim {self.dim}"
            )
        first, second = torch.chunk(inputs, chunks=2, dim=self.dim)
        return torch.maximum(first, second)


class ConvMFM(nn.Sequential):
    """Convolution producing two competitors per output feature map."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels * 2,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            MaxFeatureMap(dim=1),
        )


class STCLightCNN(nn.Module):
    """Light CNN countermeasure following the STC ASVspoof 2019 paper.

    The convolutional topology follows Table 1 of Lavrentyeva et al. The
    paper's inconsistent printed tensor sizes are resolved with same-padding
    convolutions and four 2x2 pooling operations, giving 53x37 feature maps
    for the prescribed 863x600 FFT input.
    """

    def __init__(
        self,
        input_frequency_bins=863,
        input_frames=600,
        dropout=0.75,
        n_classes=2,
    ):
        super().__init__()
        self.input_frequency_bins = input_frequency_bins
        self.input_frames = input_frames

        self.features = nn.Sequential(
            # Conv_1, MFM_2, MaxPool_3
            ConvMFM(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv_4 .. BatchNorm_10
            ConvMFM(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            ConvMFM(32, 48, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            # Conv_11 .. MaxPool_16
            ConvMFM(48, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            ConvMFM(48, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv_17 .. MaxPool_28
            ConvMFM(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            ConvMFM(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            ConvMFM(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            ConvMFM(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        pooled_frequency = input_frequency_bins // 16
        pooled_frames = input_frames // 16
        if pooled_frequency <= 0 or pooled_frames <= 0:
            raise ValueError("LCNN input is too small for four pooling operations")
        flattened_features = 32 * pooled_frequency * pooled_frames

        self.fc1 = nn.Linear(flattened_features, 160)
        self.mfm_fc1 = MaxFeatureMap(dim=1)
        # The homework explicitly requires dropout before the final BatchNorm.
        self.dropout = nn.Dropout(p=dropout)
        self.final_batch_norm = nn.BatchNorm1d(80)
        self.fc2 = nn.Linear(80, n_classes)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, spectrogram, **batch):
        expected = (1, self.input_frequency_bins, self.input_frames)
        if tuple(spectrogram.shape[1:]) != expected:
            raise ValueError(
                f"Expected LCNN input [batch, {expected[0]}, {expected[1]}, "
                f"{expected[2]}], got {tuple(spectrogram.shape)}"
            )

        features = self.features(spectrogram)
        embedding = self.fc1(features.flatten(start_dim=1))
        embedding = self.mfm_fc1(embedding)
        embedding = self.dropout(embedding)
        embedding = self.final_batch_norm(embedding)
        logits = self.fc2(embedding)

        # Labels use spoof=0, bonafide=1. A larger score therefore supports
        # the bona fide hypothesis as required by the official EER code.
        scores = logits[:, 1] - logits[:, 0]
        return {"logits": logits, "scores": scores, "embeddings": embedding}

    def __str__(self):
        result = super().__str__()
        all_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return (
            f"{result}\nAll parameters: {all_parameters}\n"
            f"Trainable parameters: {trainable_parameters}"
        )
