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
        feature = torch.tensor([], device=self.device)
        for data, target in cal_loader:
            data, target = data.to(self.device), target.to(self.device)
            logits = self.combined_net(data)
            feature = torch.cat([feature, self.combined_net.get_features(data)])
            prob = torch.softmax(logits, dim=-1)

            batch_score = self.score_function.compute_target_score(prob, target)
            cal_score = torch.cat([cal_score, batch_score], dim=0)
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
        n = len(self.cal_feature)

        # Vectorized kernel computation (100x faster)
        cal_features = self.cal_feature  # [5000, feature_dim]
        diff = cal_features - test_feature.unsqueeze(0)  # [5000, feature_dim]
        similarities = torch.exp(-torch.norm(diff, dim=1) ** 2 / (2 * self.bandwidth ** 2))  # [5000]

        # Update H matrix efficiently
        self.H[:-1, -1] = similarities
        self.H[-1, :-1] = similarities
        self.H[-1, -1] = 0.5  # Self-similarity

        self.Q = torch.cumsum(self.H, dim=-1)


    def calibrate_instance(self, data, target, alpha):
        data = data.unsqueeze(dim=0)
        logits = self.combined_net(data)
        test_feature = self.combined_net.get_features(data)
        self.get_weight(test_feature)
        n = self.cal_score.shape[0]
        theta_p = torch.zeros(size=(n + 2,), device=self.device)
        theta = torch.zeros(size=(n + 2,), device=self.device)
        theta_hat = torch.zeros(size=(n + 2,), device=self.device)
        A_1, A_2, A_3 = [], [], []

        for i in range(1, n + 2):
            theta_p[i] = (self.Q[i, i - 1] + self.H[i, n+1]) / (self.Q[i, n] + self.H[i, n+1])
            theta[i] = self.Q[i, i - 1] / (self.Q[i, n] + self.H[i, n+1])
            theta_hat[i] = self.Q[n+1, i] / (torch.sum(self.H[n+1, ]))
            if theta_p[i] < theta_hat[i]:
                A_1.append(i)
            if theta_hat[i] <= theta[i]:
                A_2.append(i)
            if theta_p[i] >= theta_hat[i] and theta_hat[i] > theta[i]:
                A_3.append(i)
        theta_A_1 = [theta_p[i] for i in A_1]
        theta_A_2 = [theta[i] for i in A_2]
        theta_A_3 = [i - 1 for i in A_3]
        L1, L2, L3 = len(A_1), len(A_2), len(A_3)
        S_k = []
        for k in range(1, n+2):
            c1, c2, c3 = 0, 0, 0
            while c1 < L1 and theta_A_1[c1] < theta_hat[k]:
                c1 += 1
            while c2 < L2 and theta_A_2[c2] < theta_hat[k]:
                c2 += 1
            while c3 < L3 and theta_A_3[c3] < k-1:
                c3 += 1
            S_k.append((c1+c2+c3)/(n+1))
        optimal_k = -1
        for i in range(n+1):
            if S_k[i] > 1 - alpha:
                optimal_k = i - 1
                break

        threshold = self.cal_score[optimal_k]
        prob = torch.softmax(logits, dim=-1)
        acc = (torch.argmax(prob) == target).to(torch.int)
        score = self.score_function(prob)[0]
        prediction_set_size = torch.sum(score <= threshold).item()
        coverage = 1 if score[target] < threshold else 0
        return prediction_set_size, coverage, acc


    def calibrate(self, cal_loader, alpha=None):
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