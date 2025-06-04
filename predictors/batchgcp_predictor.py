from .predictor import Predictor
import torch
import math
import torch.optim as optim
from models.net.length_optimization_net import SimpleNN

class BatchGcpPredictor(Predictor):
    def __init__(self, args, net):
        super().__init__(args, net)
        self.theta_tensor = None
        self.theta_marginal = None


    def calibrate(self, cal_loader, alpha=None):
        X_cal = torch.tensor([], device="cuda")
        S_cal = torch.tensor([], device="cuda")
        for data, y_true in cal_loader:
            data = data.to("cuda")
            y_true = y_true.to("cuda")

            X_cal = torch.cat((X_cal, data), 0)
            y_pred = self.net(data)
            S_cal = torch.cat((S_cal, self.score_function(y_pred, y_true)), 0)

        split_idx = int(0.7 * X_cal.shape[0])

        S_cal = S_cal.clone().detach().requires_grad_(False)

        X_cal = X_cal.to("cuda")
        S_cal = S_cal.to("cuda")

        # Pinball Loss
        def pinball_loss(q, s, tau):
            return torch.where(s >= q, tau * (s - q), (1 - tau) * (q - s))

        # Definitions and initializations
        theta_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True, device="cuda")
        theta_marginal = torch.tensor([1.0], requires_grad=True, device="cuda")

        self.theta_tensor = theta_tensor
        self.theta_marginal = theta_marginal

        optimizer = optim.Adam([theta_tensor, theta_marginal], lr=0.01)

        for epoch in range(5000):
            optimizer.zero_grad()

            # Simplified example operation
            sum_theta = torch.sum(theta_tensor * X_cal[:, :theta_tensor.shape[0]] + theta_marginal, axis=1)

            loss = pinball_loss(sum_theta, S_cal, 1 - self.alpha).mean()

            loss.backward()
            optimizer.step()

    def evaluate(self, test_loader):
        self.net.eval()

        total_coverage = 0
        total_width = 0
        total_samples = 0
        with torch.no_grad():
            for data, y_true in test_loader:
                data = data.to("cuda")
                y_true = y_true.to("cuda")
                y_pred = self.net(data)
                batch_score = self.score_function(y_pred, y_true)

                total_samples += data.shape[0]

                total_coverage += torch.sum(
                    batch_score <= torch.sum(self.theta_tensor * data[:, :self.theta_tensor.shape[0]] + self.theta_marginal, axis=1)
                )

                total_width += 2 * torch.sum(self.theta_tensor * data[:, :self.theta_tensor.shape[0]] + self.theta_marginal)

            coverage = total_coverage / total_samples
            avg_width = total_width / total_samples

            result_dict = {
                f"AverageWidth": avg_width,
                f"Coverage": coverage,
            }
        return result_dict