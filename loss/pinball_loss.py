import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    def __init__(self, alpha=0.1):
        """It learns the alpha quantile, not 1 - alpha quantile."""
        super(PinballLoss, self).__init__()
        self.alpha = alpha
        assert 0 < self.alpha < 1, "Alpha must be in (0, 1)."

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        loss = torch.max((self.alpha) * errors, (self.alpha - 1) * errors)
        return torch.mean(loss)