import torch
from torch import nn


class MFM2d(nn.Module):
    """
    Max-Feature-Map activation (MFM 2/1) for convolutional feature maps.

    Splits the channel dimension of a (N, C, H, W) tensor into two equal
    halves and takes their element-wise maximum, halving the number of
    channels. C must be even.
    """

    def forward(self, x):
        """
        Args:
            x (Tensor): tensor of shape (N, C, H, W), C even.
        Returns:
            x (Tensor): tensor of shape (N, C // 2, H, W).
        """
        a, b = torch.split(x, x.shape[1] // 2, dim=1)
        return torch.max(a, b)


class MFM1d(nn.Module):
    """
    Max-Feature-Map activation (MFM 2/1) for fully-connected features.

    Splits the feature dimension of a (N, F) tensor into two equal halves
    and takes their element-wise maximum, halving the number of features.
    F must be even.
    """

    def forward(self, x):
        """
        Args:
            x (Tensor): tensor of shape (N, F), F even.
        Returns:
            x (Tensor): tensor of shape (N, F // 2).
        """
        a, b = torch.split(x, x.shape[1] // 2, dim=1)
        return torch.max(a, b)
