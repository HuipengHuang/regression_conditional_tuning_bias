from .base_loss import BaseLoss
import torch
import torch.nn.functional as F
import torch.nn as nn

class MyLoss():
    def __init__(self, args, predictor):
        self.alpha = args.alpha
        self.predictor = predictor
        self.batch_size = args.batch_size
        if args.temperature is None:
            self.T = 1e-1
        else:
            self.T = args.temperature

        if args.size_loss_weight is None:
            raise ValueError("Please specify a size_loss_weight")
        else:
            self.size_loss_weight = args.size_loss_weight

        if args.tau is None:
            raise ValueError("Please specify a tau.")
        else:
            self.tau = args.tau

    def forward(self,  weight, logits, target) -> torch.Tensor:
        shuffled_indices = torch.randperm(logits.size(0))
        shuffled_logit = logits[shuffled_indices]
        shuffled_target = target[shuffled_indices]
        shuffled_weight = weight[shuffled_indices]

        pred_size = int(shuffled_target.size(0) * 0.5)
        pred_logit, cal_logit = shuffled_logit[:pred_size], shuffled_logit[pred_size:]
        pred_target, cal_target = shuffled_target[:pred_size], shuffled_target[pred_size:]
        pred_weight, cal_weight = shuffled_weight[:pred_size], shuffled_weight[pred_size:]

        threshold = self.predictor.calibrate_batch_logit(cal_weight, cal_logit, cal_target, self.alpha)
        pred_prob = torch.softmax(pred_logit, dim=-1)
        pred_score = self.predictor.score_function(pred_weight, pred_prob)

        smooth_pred = torch.sigmoid((threshold - pred_score) / self.T)

        size_loss = self.compute_size_loss(smooth_pred)

        loss = torch.log(size_loss + 1e-8)
        return loss

    def compute_size_loss(self, smooth_pred) -> torch.Tensor:
        size_loss = torch.maximum(torch.sum(smooth_pred,dim=-1) - self.tau, torch.tensor([0], device=smooth_pred.device)).mean()
        return size_loss

class MyAdapterLoss():
    def __init__(self, args, predictor):
        super().__init__()
        self.predictor = predictor
        if args.temperature is None:
            self.T = 1e-1
        else:
            self.T = args.temperature

    def forward(self, weight, logits, target):
        prob = torch.softmax(logits, dim=-1)
        score = self.predictor.score_function(weight, prob)
        target_score = torch.gather(score, dim=1, index=target.unsqueeze(1))
        loss = torch.sigmoid((target_score.unsqueeze(0) - score) / self.T)
        print("---")
        print(target_score)
        print(score)
        loss = loss.mean()
        return loss
