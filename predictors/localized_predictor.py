import math

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
        self.v_hat = None
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
        self.v_hat = torch.cat([torch.tensor([-math.inf], device=self.device), cal_score, torch.tensor([math.inf], device=self.device)], dim=0)
        feature = feature[index]
        self.cal_feature = feature

        n = len(self.cal_feature)
        features = self.cal_feature
        H = torch.zeros((n, n), device=self.device)

        # Process in chunks to reduce memory
        chunk_size = 500  # Adjust based on GPU memory
        for i in range(0, n, chunk_size):
            for j in range(0, n, chunk_size):
                chunk_diff = features[i:i + chunk_size].unsqueeze(1) - features[j:j + chunk_size].unsqueeze(0)  # [chunk, chunk, D]
                chunk_norms = torch.norm(chunk_diff / 100, dim=-1)  # [chunk, chunk]
                H[i:i + chunk_size, j:j + chunk_size] = torch.exp(-chunk_norms ** 2 / (2 * self.bandwidth ** 2))

        self.H = torch.zeros((n + 2, n + 2), device=self.device)
        self.H[1:-1, 1:-1] = H

    def gaussian_kernel(self, x1, x2):
        """Gaussian kernel for measuring similarity between features"""
        diff = x1 - x2
        return torch.exp(-torch.norm(diff) ** 2 / (2 * self.bandwidth ** 2))

    def get_weight(self, test_feature):
        test_feature = test_feature.squeeze(0)  # [feature_dim]

        cal_features = self.cal_feature  # [5000, feature_dim]
        diff = cal_features - test_feature.unsqueeze(0) # [5000, feature_dim]
        similarities = torch.exp(-torch.norm(diff / 100, dim=1) ** 2 / (2 * self.bandwidth ** 2))  # [5000]
        similarities = torch.cat([similarities, torch.tensor([1], device=self.device)], dim=0)

        self.H[1:, -1] = similarities
        self.H[-1, 1:] = similarities

        self.Q = torch.cumsum(self.H, dim=-1)

    def calibrate_instance(self, data, target, alpha):
        # Forward pass
        data = data.unsqueeze(dim=0)
        logits = self.combined_net(data)
        test_feature = self.combined_net.get_features(data)
        self.get_weight(test_feature)

        n = self.cal_score.shape[0]

        # Vectorized computations
        Q_diag = torch.diagonal(self.Q, offset=-1)[1 : n+2]  # Q[i,i-1] for i=1..n+1
        Q_rowsum = self.Q[1:n+2, n]  # Q[i,n] for i=1..n+1
        H_lastcol = self.H[1:n+2, n + 1]  # H[i,n+1] for i=1..n+1

        theta_p = (Q_diag + H_lastcol) / (Q_rowsum + H_lastcol)
        theta = Q_diag / (Q_rowsum + H_lastcol)
        theta_hat = self.Q[n + 1, :n + 1] / torch.sum(self.H[n + 1, :])

        # Masks
        mask_A1 = theta_p < theta_hat
        mask_A2 = theta_hat <= theta
        mask_A3 = (~mask_A1) & (~mask_A2)

        # Sorted values for binary search
        theta_A1 = theta_p[mask_A1]
        theta_A2 = theta[mask_A2]
        theta_A3 = torch.where(mask_A3)[0] + 1
        L1, L2, L3 = theta_A1.shape[0], theta_A2.shape[0], theta_A3.shape[0]

        S_k = [0,]
        theta_hat = torch.cat([torch.tensor([0], device=self.device), theta_hat], dim=0)
        theta_A1 = torch.cat([torch.tensor([0], device=self.device), theta_A1], dim=0)
        theta_A2 = torch.cat([torch.tensor([0], device=self.device), theta_A2], dim=0)
        theta_A3 = torch.cat([torch.tensor([0], device=self.device), theta_A3], dim=0)
        for k in range(1, n + 2):
            c1, c2, c3 = 0, 0, 0
            while c1 < L1 and theta_A1[c1+1] < theta_hat[k]:
                c1 += 1
            while c2 < L2 and theta_A2[c2+1] < theta_hat[k]:
                c2 += 1
            while c3 < L3 and theta_A3[c3+1] < k - 1:
                c3 += 1
            S_k.append((c1 + c2 + c3))

        S_k = torch.tensor(S_k, device=self.device) / (n + 1)
        optimal_k = S_k[S_k < (1 - alpha)].shape[0] - 1

        threshold = self.v_hat[optimal_k]
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
            instance_coverage_gap = 0

            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                for i in range(data.shape[0]):
                    prediction_set_size, coverage, acc = self.calibrate_instance(data[i], target[i], alpha=self.alpha)
                    total_set_size += prediction_set_size
                    total_coverage += coverage
                    total_accuracy += acc
                    instance_coverage_gap += abs(coverage - (1 - self.alpha))
            total_samples = len(test_loader.dataset)
            accuracy = total_accuracy / total_samples
            coverage = total_coverage / total_samples
            avg_set_size = total_set_size / total_samples
            instance_coverage_gap = instance_coverage_gap / total_samples

            return {
                f"{self.args.score}_Top1Accuracy": accuracy,
                f"{self.args.score}_Coverage": coverage,
                f"{self.args.score}_AverageSetSize": avg_set_size,
                f"{self.args.score}_instance_coverage_gap": instance_coverage_gap
            }
