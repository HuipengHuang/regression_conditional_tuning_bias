import numpy as np
from scores.utils import get_score
import torch
import math


class Predictor:
    def __init__(self, args, net):
        self.score_function = get_score(args)

        self.net = net
        self.threshold = None
        self.alpha = args.alpha
        self.device = next(net.parameters()).device
        self.args = args

    def calibrate(self, cal_loader, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        self.net.eval()
        with torch.no_grad():
            if alpha is None:
                alpha = self.alpha
            cal_score = torch.tensor([], device=self.device)
            for data, y_true in cal_loader:
                data = data.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.net(data)

                batch_score = self.score_function(y_pred, y_true)

                cal_score = torch.cat((cal_score, batch_score), 0)

            N = cal_score.shape[0]
            threshold = torch.quantile(cal_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

            self.threshold = threshold
            return threshold

    def evaluate(self, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        self.net.eval()
        if self.threshold is not None:
            with torch.no_grad():
                total_coverage = 0
                total_prediction_width = 0
                total_samples = 0

                for data, y_true in test_loader:
                    data, y_true = data.to(self.device), y_true.to(self.device)
                    batch_size = y_true.shape[0]
                    total_samples += batch_size

                    y_pred = self.net(data)

                    conf_interval = self.score_function.generate_intervals(y_pred, self.threshold)

                    total_coverage += torch.sum((y_true >= conf_interval[:, 0]).to(torch.int) *
                                                (y_true <= conf_interval[:, 1]).to(torch.int))

                    total_prediction_width += torch.sum(conf_interval[:, 1] - conf_interval[:, 0])

                coverage = total_coverage / total_samples
                avg_width = total_prediction_width / total_samples

                result_dict = {
                    f"AverageWidth": avg_width,
                    f"Coverage": coverage,
                }
        else:
            raise NotImplementedError


        return result_dict

