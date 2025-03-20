from .base_loss import BaseLoss
import torch

class CAdapterLoss(BaseLoss):
    def __init__(self, args, predictor):
        super().__init__()
        self.predictor = predictor
        self.T = args.T

    def forward(self, logits, target):
        prob = torch.softmax(logits, dim=1)
        score = self.predictor.score_function(prob)
        target_score = torch.gather(score, dim=1, index=target.unsqueeze(1))
        loss = torch.sigmoid((target_score.unsqueeze(0) - score) / self.T).sum()
        return loss / logits.size(0) / logits.size(1)

