import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss
    """

    def __init__(self, weight=None):
        """
        Args:
            weight (list[float] | None): per-class weights,
                ordered by label index (0: spoof, 1: bonafide).
                None disables weighting.
        """
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor)
            labels (Tensor)
        Returns:
            losses (dict): dict containing the 'loss' key.
        """
        return {"loss": self.loss(logits, labels)}
