"""Loss for the single-factor BCE versus BCE + soft Dice experiment."""

import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    """Mean pixel BCE + mean per-image soft Dice loss (equal coefficients).

    Dice is computed on sigmoid probabilities, independently for each patch,
    then averaged over the batch. Smoothing also defines background-only patches.
    No threshold is applied here; threshold 0.5 is used only for evaluation.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Expected matching binary N x 1 x H x W tensors")
        probabilities = torch.sigmoid(logits).flatten(1)
        targets_flat = targets.flatten(1)
        intersection = (probabilities * targets_flat).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return self.bce(logits, targets) + (1 - dice).mean()
