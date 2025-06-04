from .naive_model import NaiveModel
from scores.utils import get_score
import torch
import torch.optim as optim


class BatchGcp_model(NaiveModel):
    def __init__(self, net, args):
        super().__init__(net, args)
        self.alpha = args.alpha
        self.score_function = get_score(args)

    def forward(self, x):
        return self.net(x)

    def tune(self, tune_loader):
        def pinball_loss(q, s, tau):
            return torch.where(s >= q, tau * (s - q), (1 - tau) * (q - s))

        X_cal = torch.tensor([], device="cuda")
        S_cal = torch.tensor([], device="cuda")
        for data, y_true in tune_loader:
            data = data.to("cuda")
            y_true = y_true.to("cuda")

            X_cal = torch.cat((X_cal, data), 0)
            y_pred = self.net(data)
            S_cal = torch.cat((S_cal, self.score_function(y_pred, y_true)), 0)

        split_idx = int(0.7 * X_cal.shape[0])

        S_cal = S_cal.clone().detach().requires_grad_(False)
        # Definitions and initializations
        theta_tensor = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True).to("cuda")
        theta_marginal = torch.tensor([1.0], requires_grad=True).to("cuda")

        optimizer = optim.Adam([theta_tensor, theta_marginal], lr=0.01)

        for epoch in range(5000):
            optimizer.zero_grad()

            # Simplified example operation
            sum_theta = torch.sum(theta_tensor * X_cal[:, :theta_tensor.shape[0]] + theta_marginal, axis=1)

            loss = pinball_loss(sum_theta, S_cal, 0.9).mean()

            loss.backward()
            optimizer.step()


