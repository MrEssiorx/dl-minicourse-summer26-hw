import torch
from torch import nn

from src.model.mfm import MFM1d, MFM2d


class LCNN(nn.Module):
    """
    LightCNN countermeasure, following Table 1 of the Speech Technology
    Center paper (https://arxiv.org/abs/1904.05576).

    The convolutional trunk alternates convolutions with MFM 2/1
    activations, MaxPool, and BatchNorm exactly as listed in Table 1.
    The classifier head is FC(160) - MFM - Dropout - BatchNorm - FC(2).
    """

    def __init__(self, n_freq=257, n_time=750, n_class=2, dropout=0.75):
        """
        Args:
            n_freq (int): number of frequency bins of the input spectrogram
                (n_fft // 2 + 1).
            n_time (int): number of time frames of the input spectrogram.
            n_class (int): number of output classes.
            dropout (float): dropout probability in the classifier head.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),  # Conv_1
            MFM2d(),  # MFM_2 -> 32
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool_3
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),  # Conv_4
            MFM2d(),  # MFM_5 -> 32
            nn.BatchNorm2d(32),  # BatchNorm_6
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),  # Conv_7
            MFM2d(),  # MFM_8 -> 48
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool_9
            nn.BatchNorm2d(48),  # BatchNorm_10
            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),  # Conv_11
            MFM2d(),  # MFM_12 -> 48
            nn.BatchNorm2d(48),  # BatchNorm_13
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),  # Conv_14
            MFM2d(),  # MFM_15 -> 64
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool_16
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),  # Conv_17
            MFM2d(),  # MFM_18 -> 64
            nn.BatchNorm2d(64),  # BatchNorm_19
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Conv_20
            MFM2d(),  # MFM_21 -> 32
            nn.BatchNorm2d(32),  # BatchNorm_22
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),  # Conv_23
            MFM2d(),  # MFM_24 -> 32
            nn.BatchNorm2d(32),  # BatchNorm_25
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv_26
            MFM2d(),  # MFM_27 -> 32
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool_28
        )

        n_flatten = self._flatten_size(n_freq, n_time)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 160),  # FC_29
            MFM1d(),  # MFM_30 -> 80
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),  # BatchNorm_31
            nn.Linear(80, n_class),  # FC_32
        )

    def _flatten_size(self, n_freq, n_time):
        """
        Compute the flattened feature size after the convolutional trunk by
        passing a dummy tensor through it.

        Args:
            n_freq (int): number of frequency bins.
            n_time (int): number of time frames.
        Returns:
            n_flatten (int): number of features after flattening.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_freq, n_time)
            return self.features(dummy).flatten(1).shape[1]

    def forward(self, spectrogram, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram of shape (N, 1, F, T).
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.classifier(self.features(spectrogram))}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
