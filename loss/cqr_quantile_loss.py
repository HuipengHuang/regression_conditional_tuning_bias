import torch.nn as nn
from .pinball_loss import PinballLoss
class DoubleQuantileLoss(nn.Module):
    def __init__(self, lower_quantile, upper_quantile):
        super().__init__()
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_quantile_loss = PinballLoss(self.lower_quantile)
        self.upper_quantile_loss = PinballLoss(self.upper_quantile)

    def forward(self, y_pred, y_true):
        lower_loss = self.lower_quantile_loss(y_pred[:, 0], y_true)
        upper_loss = self.upper_quantile_loss(y_pred[:, 1], y_true)

        return (lower_loss + upper_loss) / 2