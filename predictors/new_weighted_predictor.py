import abc
from scores.utils import get_score
import torch
import torch.nn as nn
import math


class New_Weighted_Predictor:
    def __init__(self, args, net, weight, adapter_net=None):
        self.score_function = get_score(args)
        if adapter_net:
            self.combined_net = nn.Sequential(net, adapter_net)
        else:
            self.combined_net = net
        self.weight = weight
        self.threshold = None
        self.alpha = args.alpha
        self.device = next(net.parameters()).device
        self.args = args

    def calibrate(self, cal_loader, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        self.combined_net.eval()
        with torch.no_grad():
            if alpha is None:
                    alpha = self.alpha
            cal_score = torch.tensor([], device=self.device)
            for data, target in cal_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.combined_net(data)
                prob = torch.softmax(logits, dim=1)

                weight = torch.softmax(self.weight, dim=-1)
                weight = torch.stack([weight for i in range(data.shape[0])], dim=0)

                batch_score = self.score_function.compute_target_score(weight, prob, target)

                cal_score = torch.cat((cal_score, batch_score), 0)
            N = cal_score.shape[0]
            threshold = torch.quantile(cal_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)
            self.threshold = threshold
            return threshold

    def calibrate_batch_logit(self, weight, logits, target, alpha):
        """Design for conformal training, which needs to compute threshold in every batch"""
        prob = torch.softmax(logits, dim=-1)
        batch_score = self.score_function.compute_target_score(weight, prob, target)
        N = target.shape[0]
        return torch.quantile(batch_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

    def evaluate(self, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        if self.threshold is None:
            raise ValueError("Threshold score is None. Please do calibration first.")
        self.combined_net.eval()
        with torch.no_grad():
            total_accuracy = 0
            total_coverage = 0
            total_prediction_set_size = 0

            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                logit = self.combined_net(data)
                prob = torch.softmax(logit, dim=-1)
                weight = self.weight
                weight = torch.softmax(weight, dim=-1)
                weight = torch.stack([weight for i in range(data.shape[0])], dim=0)

                prediction = torch.argmax(prob, dim=-1)
                total_accuracy += (prediction == target).sum().item()

                batch_score = self.score_function(weight, prob)
                prediction_set = (batch_score <= self.threshold)
                total_coverage += prediction_set[torch.arange(target.shape[0]), target].sum().item()
                total_prediction_set_size += prediction_set.sum().item()

            accuracy = total_accuracy / len(test_loader.dataset)
            coverage = total_coverage / len(test_loader.dataset)
            avg_set_size = total_prediction_set_size / len(test_loader.dataset)
            result_dict = {f"{self.args.score}_Top1Accuracy": accuracy,
                           f"{self.args.score}_AverageSetSize": avg_set_size,
                           f"{self.args.score}_Coverage": coverage}
            return result_dict

