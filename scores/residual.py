from .base_score import BaseScore
import torch


class Residual_score(BaseScore):
    def __init__(self,):
        super().__init__()

    def __call__(self, logits, labels):
        return torch.abs(logits - labels)

    def generate_intervals(self, predicts_batch, q_hat):
        return torch.stack((predicts_batch - q_hat, predicts_batch + q_hat), dim=-1)



