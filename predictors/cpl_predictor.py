from .predictor import Predictor
import torch
import math
import torch.optim as optim
from models.net.length_optimization_net import SimpleNN

class CPLPredictor(Predictor):
    def __init__(self, args, net):
        super().__init__(args, net)

    def maximize_for_h(self, optimizer_h, X, S, h, lambda_tensor, lambda_marginal, alpha, sigma=0.1):
        h.train()
        h_x = h(X)
        indicator_approx = 0.5 * (1 + torch.erf((-S + h_x) / (sigma * torch.sqrt(torch.tensor(2, device="cuda")))))
        sum_lambda = torch.sum(lambda_tensor * X[:, :lambda_tensor.shape[0]] + lambda_marginal, axis=1)
        product = sum_lambda * (indicator_approx - (1.0 - alpha))
        h_x_positive = torch.clamp(h_x, min=0)
        loss_h = - torch.mean(product - h_x_positive)
        optimizer_h.zero_grad()
        loss_h.backward()
        optimizer_h.step()

        return loss_h.item()

    def minimize_for_f(self, optimizer_lambda, X, S, h, lambda_tensor, lambda_marginal, alpha, sigma=0.1):
        optimizer_lambda.zero_grad()
        h_x = h(X)
        indicator_approx = (S <= h_x).float()

        sum_lambda = torch.sum(lambda_tensor * X[:, :lambda_tensor.shape[0]] + lambda_marginal, axis=1)
        product = sum_lambda * (indicator_approx - (1.0 - alpha))
        h_x_positive = torch.clamp(h_x, min=0)
        loss_lambda = torch.mean(product - h_x_positive)
        loss_lambda.backward()
        optimizer_lambda.step()

        return lambda_tensor, lambda_marginal, loss_lambda.item()

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
        # Split the tensors
        X1 = X_cal[:split_idx]
        S1 = S_cal[:split_idx]
        X2 = X_cal[split_idx:]
        S2 = S_cal[split_idx:]

        lambda_tensor = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], requires_grad=True,
                                     device="cuda")
        lambda_marginal = torch.tensor([5.0], requires_grad=True, device="cuda")

        optimizer_lambda = optim.Adam([lambda_tensor] + [lambda_marginal], lr=1)

        for t in range(60):
            if t % 1000 == 15:
                optimizer_lambda = optim.Adam([lambda_tensor] + [lambda_marginal], lr=0.5)
            if t % 1000 == 30:
                optimizer_lambda = optim.Adam([lambda_tensor] + [lambda_marginal], lr=0.01)
            h = SimpleNN(n_binary=10, n_continuous=self.args.in_shape - 10).to("cuda")

            self.h = h
            optimizer_h = optim.Adam(self.net.parameters(), lr=0.01)

            for epoch in range(2000):
                loss_h = self.maximize_for_h(optimizer_h, X1, S1, h, lambda_tensor, lambda_marginal, alpha=self.alpha,
                                             sigma=0.1)
                if (epoch + 1) % 1500 == 0:
                    print(f"{t + 1} / 60 {epoch + 1} / 2000 Loss: {loss_h}")

                if epoch % 1000 == 500:
                    optimizer_h = optim.Adam(self.net.parameters(), lr=0.001)
                if epoch % 2000 == 1000:
                    optimizer_h = optim.Adam(self.net.parameters(), lr=0.0002)

            for epoch in range(1):
                lambda_tensor, lambda_marginal, loss_lambda = self.minimize_for_f(optimizer_lambda, X2, S2, h,
                                                                                  lambda_tensor,
                                                                                  lambda_marginal, alpha=self.alpha,
                                                                                  sigma=0.1)
    def evaluate(self, test_loader):
        self.net.eval()
        self.h.eval()

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
                threshold = self.h(data)
                total_coverage += (batch_score <= threshold).sum()
                total_width += 2 * torch.sum(threshold)

            coverage = total_coverage / total_samples
            avg_width = total_width / total_samples

            result_dict = {
                f"AverageWidth": avg_width,
                f"Coverage": coverage,
            }
        return result_dict