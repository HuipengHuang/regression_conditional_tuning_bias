from .thr import THR
from .raps import RAPS
from .saps import SAPS
import torch

class WeightedScore():
    def __init__(self, ):
        super().__init__()
        self.thr_score_function = THR()
        self.raps_score_function = RAPS(random=True, weight=1, size_regularization=10)
        self.saps_score_function = SAPS(random=True, weight=1)

    def __call__(self, weight, prob):
        thr_score = self.thr_score_function(prob)
        raps_score = self.raps_score_function(prob)
        saps_score = self.saps_score_function(prob)
        score = thr_score * weight[:, :, 0] + raps_score * weight[:, :, 1] + saps_score * weight[:, :, 2]
        return score

    def compute_target_score(self, weight, prob, target):
        """It will return a tensor with dimension 1"""
        score = self(weight, prob)
        return score[torch.arange(target.size(0)), target]