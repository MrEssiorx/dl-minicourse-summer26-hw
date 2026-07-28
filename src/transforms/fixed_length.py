import torch
from torch import nn
from torch.nn import functional as F


class FixedLengthCrop(nn.Module):
    """
    Fixes the time dimension of a spectrogram to a constant number of frames.

    Follows the trim-pad strategy from https://arxiv.org/abs/2103.11326:
    a random contiguous chunk of num_frames is taken if the input is longer,
    zero-padding on the right is used if it is shorter. Applied on the same
    partitions (train/dev/eval), matching the reference implementation.
    """

    def __init__(self, num_frames=750):
        """
        Args:
            num_frames (int): fixed number of frames along the time axis.
        """
        super().__init__()
        self.num_frames = num_frames

    def forward(self, spectrogram):
        """
        Args:
            spectrogram (Tensor): tensor of shape (..., T)
        Returns:
            spectrogram (Tensor): tensor of shape (..., num_frames).
        """
        total_frames = spectrogram.shape[-1]

        if total_frames >= self.num_frames:
            start = torch.randint(0, total_frames - self.num_frames + 1, (1,)).item()
            spectrogram = spectrogram[..., start : start + self.num_frames]
        else:
            pad_amount = self.num_frames - total_frames
            spectrogram = F.pad(spectrogram, (0, pad_amount))

        return spectrogram
