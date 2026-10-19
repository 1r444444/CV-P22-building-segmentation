import torch
from torch import nn


class BCEDiceLoss(nn.Module):
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
