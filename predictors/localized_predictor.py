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
        self.kernel = self.gaussian_kernel
        self.cal_feature = None
        self.cal_score = None
        self.bandwidth = args.bandwidth if hasattr(args, 'bandwidth') else 1.0
        self.H = None
        self.Q = None
        self.chunk_size = 1000  # Adjustable based on memory constraints

    def gaussian_kernel(self, x1, x2):
        """Gaussian kernel for measuring similarity between features"""
        diff = x1 - x2
        return torch.exp(-torch.norm(diff) ** 2 / (2 * self.bandwidth ** 2))

    def compute_cal_score(self, cal_loader):
        """Compute calibration scores and features incrementally"""
        n_samples = len(cal_loader.dataset)
        feature_dim = self._get_feature_dim(cal_loader)  # Infer feature dimension
        self.cal_score = torch.zeros(n_samples, device=self.device)
        self.cal_feature = torch.zeros(n_samples, feature_dim, device=self.device)

        idx = 0
        with torch.no_grad():
            for data, target in cal_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]
                logits = self.combined_net(data)
                batch_features = self.combined_net.get_features(data)
                prob = torch.softmax(logits, dim=-1)
                batch_score = self.score_function.compute_target_score(prob, target)

                # Store in pre-allocated tensors
                self.cal_score[idx:idx + batch_size] = batch_score
                self.cal_feature[idx:idx + batch_size] = batch_features
                idx += batch_size

                # Clear temporary tensors
                del logits, batch_features, prob, batch_score
                torch.cuda.empty_cache()  # Free memory explicitly

        # Sort after collecting all data
        self.cal_score, index = torch.sort(self.cal_score, dim=0, descending=False)
        self.cal_feature = self.cal_feature[index]

        # Compute H in chunks
        self._compute_H_chunked()

    def _get_feature_dim(self, cal_loader):
        """Infer feature dimension from the first batch"""
        with torch.no_grad():
            data, _ = next(iter(cal_loader))
            data = data.to(self.device)
            features = self.combined_net.get_features(data)
            return features.shape[1]  # Assuming features are [batch_size, feature_dim]

    def _compute_H_chunked(self):
        """Compute H matrix in chunks to avoid OOM"""
        n = self.cal_score.shape[0]
        self.H = torch.zeros(n + 2, n + 2, device=self.device)

        for i in range(0, n, self.chunk_size):
            i_end = min(i + self.chunk_size, n)
            for j in range(i, n, self.chunk_size):
                j_end = min(j + self.chunk_size, n)
                for ii in range(i, i_end):
                    for jj in range(max(ii, j), j_end):
                        self.H[ii + 1, jj + 1] = self.kernel(self.cal_feature[ii], self.cal_feature[jj])
                        self.H[jj + 1, ii + 1] = self.H[ii + 1, jj + 1]
                torch.cuda.empty_cache()  # Free memory after each chunk

    def get_weight(self, test_feature):
        """Compute weights and Q incrementally"""
        n = self.cal_score.shape[0]
        # Update H with test_feature
        with torch.no_grad():
            for i in range(n):
                self.H[i + 1, n + 1] = self.kernel(self.cal_feature[i], test_feature)
                self.H[n + 1, i + 1] = self.H[i + 1, n + 1]
            # Compute Q row by row
            self.Q = torch.zeros_like(self.H)
            for i in range(n + 2):
                self.Q[i] = torch.cumsum(self.H[i], dim=-1)
            torch.cuda.empty_cache()

    def calibrate_instance(self, data, target, alpha):
        """Calibrate a single instance"""
        with torch.no_grad():
            data = data.unsqueeze(dim=0).to(self.device)
            logits = self.combined_net(data)
            test_feature = self.combined_net.get_features(data)
            self.get_weight(test_feature)

            n = self.cal_score.shape[0]
            theta_p = torch.zeros(n + 2, device=self.device)
            theta = torch.zeros(n + 2, device=self.device)
            theta_hat = torch.zeros(n + 2, device=self.device)
            A_1, A_2, A_3 = [], [], []

            for i in range(1, n + 2):
                theta_p[i] = (self.Q[i, i - 1] + self.H[i, n + 1]) / (self.Q[i, n] + self.H[i, n + 1])
                theta[i] = self.Q[i, i - 1] / (self.Q[i, n] + self.H[i, n + 1])
                theta_hat[i] = self.Q[n + 1, i] / torch.sum(self.H[n + 1])
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

            for k in range(1, n + 2):
                c1 = sum(1 for t in theta_A_1 if t < theta_hat[k])
                c2 = sum(1 for t in theta_A_2 if t < theta_hat[k])
                c3 = sum(1 for t in theta_A_3 if t < k - 1)
                S_k.append((c1 + c2 + c3) / (n + 1))

            optimal_k = next((i - 1 for i in range(n + 1) if S_k[i] > 1 - alpha), -1)
            threshold = self.cal_score[optimal_k]
            prob = torch.softmax(logits, dim=-1)
            acc = (torch.argmax(prob) == target).to(torch.int)
            score = self.score_function(prob)[0]
            prediction_set_size = torch.sum(score <= threshold).item()
            coverage = 1 if score[target] < threshold else 0

            return prediction_set_size, coverage, acc

    def calibrate(self, cal_loader, alpha=None):
        """Calibrate the predictor"""
        self.compute_cal_score(cal_loader)

    def evaluate(self, test_loader):
        """Evaluate on test data"""
        if self.cal_feature is None or self.cal_score is None:
            raise ValueError("Must calibrate first")

        self.combined_net.eval()
        with torch.no_grad():
            total_accuracy = 0
            total_coverage = 0
            total_set_size = 0

            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                for i in range(data.shape[0]):
                    set_size, cov, acc = self.calibrate_instance(data[i], target[i], self.alpha)
                    total_set_size += set_size
                    total_coverage += cov
                    total_accuracy += acc

            total_samples = len(test_loader.dataset)
            return {
                f"{self.args.score}_Top1Accuracy": total_accuracy / total_samples,
                f"{self.args.score}_Coverage": total_coverage / total_samples,
                f"{self.args.score}_AverageSetSize": total_set_size / total_samples
            }


# Usage example:
"""
args = ... # Your args object with alpha, score, etc.
net = ...  # Your neural network
predictor = LocalizedPredictor(args, net)
cal_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=32, shuffle=False)
predictor.calibrate(cal_loader)
results = predictor.evaluate(test_loader)
"""