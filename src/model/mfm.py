import torch
from torch import nn


class MFM2d(nn.Module):
    """
    Max-Feature-Map activation (MFM 2/1) for convolutional feature maps.
    """

    def forward(self, x):
        """
        Args:
            x (Tensor): tensor of shape (N, C, H, W), C even.
        Returns:
            x (Tensor): tensor of shape (N, C // 2, H, W).
        """
        a, b = torch.split(x, x.shape[-3] // 2, dim=-3)
        return torch.max(a, b)


class MFM1d(nn.Module):
    """
    Max-Feature-Map activation (MFM 2/1) for fully-connected features.
    """

    def forward(self, x):
        """
        Args:
            x (Tensor): tensor of shape (N, F), F even.
        Returns:
            x (Tensor): tensor of shape (N, F // 2).
        """
        a, b = torch.split(x, x.shape[-1] // 2, dim=-1)
        return torch.max(a, b)
