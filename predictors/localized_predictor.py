from scores.utils import get_score
import torch
import torch.nn as nn


class LocalizedPredictor:
    def __init__(self, args, net, adapter_net=None):
        self.score_function = get_score(args)
        if adapter_net:
            self.combined_net = nn.Sequential(net, adapter_net)
        else:
            self.combined_net = net
        self.threshold = None
        self.alpha = args.alpha
        self.device = next(net.parameters()).device
        self.args = args

        # Localization-specific attributes
        self.kernel = self.gaussian_kernel  # Kernel function for similarity
        self.cal_feature = None  # Store calibration features
        self.cal_score = None  # Store calibration scores
        self.bandwidth = args.bandwidth if hasattr(args, 'bandwidth') else 1.0  # Kernel bandwidth
        self.H = None
        self.Q = None

    def compute_cal_score(self, cal_loader):
        cal_score = torch.tensor([], device=self.device)
        feature = []
        for data, target in cal_loader:
            data, target = data.to(self.device), target.to(self.device)
            logits = self.combined_net(data)
            feature.append(self.combined_net.get_features(data))
            prob = torch.softmax(logits, dim=-1)

            batch_score = self.score_function.compute_target_score(prob, target)
            cal_score = torch.cat([cal_score, batch_score], dim=0)
        feature = torch.cat(feature, dim=0)
        cal_score, index = torch.sort(cal_score, dim=0, descending=False)
        self.cal_score = cal_score
        feature = feature[index]
        self.cal_feature = feature

        n = len(self.cal_feature)
        features = self.cal_feature

        diff = features.unsqueeze(1) - features.unsqueeze(0)  # [5000, 5000, D]
        norms = torch.norm(diff, dim=-1)  # [5000, 5000]
        H = torch.exp(-norms ** 2 / (2 * self.bandwidth ** 2))  # [5000, 5000]
        # Pad with zeros for +2 dimensions
        self.H = torch.zeros((n + 2, n + 2), device=self.device)
        self.H[1:-1, 1:-1] = H

    def gaussian_kernel(self, x1, x2):
        """Gaussian kernel for measuring similarity between features"""
        diff = x1 - x2
        return torch.exp(-torch.norm(diff) ** 2 / (2 * self.bandwidth ** 2))

    def get_weight(self, test_feature):
        test_feature = test_feature.squeeze(0)  # [feature_dim]

        cal_features = self.cal_feature  # [5000, feature_dim]
        diff = cal_features - test_feature.unsqueeze(0)  # [5000, feature_dim]
        similarities = torch.exp(-torch.norm(diff, dim=1) ** 2 / (2 * self.bandwidth ** 2))  # [5000]
        similarities = torch.cat([similarities, torch.tensor([0.5], device=self.device)], dim=0)

        self.H[1:, -1] = similarities
        self.H[-1, 1:] = similarities

        self.Q = torch.cumsum(self.H, dim=-1)

    def calibrate_instance(self, data, target, alpha):
        # Forward pass (unchanged)
        data = data.unsqueeze(dim=0)
        logits = self.combined_net(data)
        test_feature = self.combined_net.get_features(data)
        self.get_weight(test_feature)

        n = self.cal_score.shape[0]

        # Vectorized computations for theta_p, theta, theta_hat
        Q_diag = torch.diagonal(self.Q, offset=-1)[:n + 1]  # Q[i,i-1] for i=1..n+1
        Q_rowsum = self.Q[:n + 1, n]  # Q[i,n] for i=1..n+1
        H_lastcol = self.H[:n + 1, n + 1]  # H[i,n+1] for i=1..n+1

        theta_p = (Q_diag + H_lastcol) / (Q_rowsum + H_lastcol)
        theta = Q_diag / (Q_rowsum + H_lastcol)
        theta_hat = self.Q[n + 1, :n + 1] / torch.sum(self.H[n + 1, :])

        # Vectorized condition checks
        mask_A1 = theta_p < theta_hat
        mask_A2 = theta_hat <= theta
        mask_A3 = (~mask_A1) & (~mask_A2)

        # Precompute sorted lists for binary search
        theta_A1_sorted = torch.sort(theta_p[mask_A1]).values
        theta_A2_sorted = torch.sort(theta[mask_A2]).values
        A3_indices = torch.where(mask_A3)[0]  # Already 0-based indices

        # Compute S_k efficiently
        theta_hat_expanded = theta_hat.unsqueeze(1)
        S_k = (
                      torch.searchsorted(theta_A1_sorted, theta_hat_expanded).float() +
                      torch.searchsorted(theta_A2_sorted, theta_hat_expanded).float() +
                      torch.searchsorted(A3_indices.float(),
                                         torch.arange(n + 1, device=self.device).unsqueeze(1)).float()
              ) / (n + 1)

        # Find optimal k
        valid_k = torch.where(S_k.squeeze() > (1 - alpha))[0]
        optimal_k = valid_k[0] - 1 if len(valid_k) > 0 else n - 1

        # Final calculations (unchanged)
        threshold = self.cal_score[optimal_k]
        prob = torch.softmax(logits, dim=-1)
        acc = (torch.argmax(prob) == target).to(torch.int)
        score = self.score_function(prob)[0]
        prediction_set_size = torch.sum(score <= threshold).item()
        coverage = 1 if score[target] < threshold else 0

        return prediction_set_size, coverage, acc


    def calibrate(self, cal_loader, alpha=None):
        self.combined_net.eval()
        with torch.no_grad():
            self.compute_cal_score(cal_loader)

    def evaluate(self, test_loader):
        """Make localized predictions"""
        if self.cal_feature is None or self.cal_score is None:
            raise ValueError("Must calibrate first")

        self.combined_net.eval()
        with torch.no_grad():
            total_accuracy = 0
            total_coverage = 0
            total_set_size = 0

            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                for i in range(data.shape[0]):
                    prediction_set_size, coverage, acc = self.calibrate_instance(data[i], target[i], alpha=self.alpha)
                    total_set_size += prediction_set_size
                    total_coverage += coverage
                    total_accuracy += acc

            total_samples = len(test_loader.dataset)
            accuracy = total_accuracy / total_samples
            coverage = total_coverage / total_samples
            avg_set_size = total_set_size / total_samples

            return {
                f"{self.args.score}_Top1Accuracy": accuracy,
                f"{self.args.score}_Coverage": coverage,
                f"{self.args.score}_AverageSetSize": avg_set_size
            }


# Usage example:
"""
args = ... # Your args object with alpha, score, etc.
net = ...  # Your neural network
predictor = LocalizedPredictor(args, net)
predictor.calibrate(calibration_loader)
results = predictor.predict(test_loader)
"""