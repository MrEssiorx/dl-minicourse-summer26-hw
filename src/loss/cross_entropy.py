import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for bonafide/spoof classification.

    The comparative study (https://arxiv.org/abs/2103.11326) reports that
    margin-based softmax is comparable to plain cross-entropy on ASVspoof2019
    LA and that the random-seed variance can exceed the difference between
    losses, so plain cross-entropy is used.

    ASVspoof2019 LA is class-imbalanced (spoof outnumbers bonafide roughly
    9:1), so optional per-class weights can be supplied to counter it.
    """

    def __init__(self, weight=None):
        """
        Args:
            weight (list[float] | None): per-class weights passed to
                nn.CrossEntropyLoss, ordered by label index (0: spoof,
                1: bonafide). None disables weighting.
        """
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): model output of shape (N, n_class).
            labels (Tensor): ground-truth labels of shape (N,).
        Returns:
            losses (dict): dict containing the 'loss' key.
        """
        return {"loss": self.loss(logits, labels)}
