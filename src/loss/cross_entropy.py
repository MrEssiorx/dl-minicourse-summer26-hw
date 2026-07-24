import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for bonafide/spoof classification.

    The comparative study (https://arxiv.org/abs/2103.11326) reports that
    the gain of margin-based softmax over plain cross-entropy on ASVspoof2019
    LA is trivial and not statistically significant, so plain cross-entropy
    is used.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): model output of shape (N, n_class).
            labels (Tensor): ground-truth labels of shape (N,).
        Returns:
            losses (dict): dict containing the 'loss' key.
        """
        return {"loss": self.loss(logits, labels)}
