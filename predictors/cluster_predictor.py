import numpy as np
from sklearn.cluster import KMeans
from .utils import get_quantile_threshold, get_clustering_parameters
from scores.utils import get_score
import torch
import math
from dataset.utils import split_dataloader
from torch.utils.data import DataLoader

class ClusterPredictor:
    def __init__(self, args, net):
        self.score_function = get_score(args)
        self.net = net
        self.alpha = args.alpha
        self.device = next(net.parameters()).device
        self.args = args
        self.k = int(args.k) if args.k is not None else None
        self.num_classes = args.num_classes
        self.class_threshold = None
        self.class2cluster = None
        self.cluster2class = None

    def calibrate(self, cal_loader, alpha=None):
        """Input calibration dataloader.
        Compute scores for all the calibration data and take the (1 - alpha) quantile.
        If args.null_qhat == 'standard', we compute the qhat for standard conformal and use that as the default value
        """

        self.net.eval()
        with torch.no_grad():
            if alpha is None:
                alpha = self.alpha
            cal_scores, cal_targets = self.cluster_class(cal_loader, alpha)
            self.class_threshold = torch.zeros(size=(self.num_classes,), device=self.device)

            for i in range(self.k):
                class_list = self.cluster2class[i]
                if (i == self.k - 1) and (self.args.null_qhat == 'standard'):
                    cluster_score = cal_scores
                else:
                    mask = torch.zeros(size=(cal_targets.shape[0],), device=self.device)
                    for class_id in class_list:
                        mask = (mask) + (cal_targets == class_id)
                    cluster_score = cal_scores[mask > 0]
                N = cluster_score.shape[0]

                cluster_quantile = torch.quantile(cluster_score, math.ceil((1 - alpha) * (N + 1)) / N)
                self.class_threshold[torch.tensor(self.cluster2class[i], device=self.device, dtype=torch.long)] += cluster_quantile

    def cluster_class(self, cal_loader, alpha):
        n_threshold = get_quantile_threshold(alpha if alpha < 0.1 else 0.1)

        T = [0.5, 0.6, 0.7, 0.8, 0.9]
        if (1 - alpha) not in T:
            T.append(1 - alpha)

        with torch.no_grad():
            all_targets = []
            all_scores = []

            for data, target in cal_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                logits = self.net(data)
                prob = torch.softmax(logits, dim=1)
                target_score = self.score_function.compute_target_score(prob, target)
                all_targets.append(target)
                all_scores.append(target_score)

            all_targets = torch.cat(all_targets)
            all_scores = torch.cat(all_scores)

            num_per_class = torch.tensor([(all_targets == i).sum() for i in range(self.num_classes)])
            n_min = min(num_per_class)
            n_min = max(n_min, n_threshold)
            num_remaining_classes = torch.sum(num_per_class >= n_min)

            if self.k is None:
                n_clustering, self.k = get_clustering_parameters(num_remaining_classes, n_min)

                cluster_scores, cal_scores = all_scores[:n_clustering], all_scores[n_clustering:]
                cluster_targets, cal_targets = all_targets[:n_clustering], all_targets[n_clustering:]
            else:
                n_clustering = int(0.5 * len(cal_loader.dataset))
                cluster_scores, cal_scores = all_scores[:n_clustering], all_scores[n_clustering:]
                cluster_targets, cal_targets = all_targets[:n_clustering], all_targets[n_clustering:]

            class2cluster = {i: 0 for i in range(self.num_classes)}
            cluster2class = {i: [] for i in range(self.k + 1)}
            class_quantile_score = torch.tensor([], device=self.device)
            class_cts = []
            class_idx_list = []
            for class_idx in range(self.num_classes):
                mask = (cluster_targets == class_idx)

                scores = cluster_scores[mask]
                if len(scores) <= n_threshold:
                    class2cluster[class_idx] = self.k
                    cluster2class[self.k].append(class_idx)
                    continue

                class_idx_list.append(class_idx)
                class_quantile_score = torch.cat((class_quantile_score, torch.zeros(size=(1, len(T)), device=self.device)), dim=0)
                class_cts.append(scores.shape[0])
                for j, t in enumerate(T):
                    q = math.ceil(t * (len(scores) + 1)) / len(scores)
                    class_quantile_score[-1, j] = torch.quantile(scores, q)

            class_quantiles_np = class_quantile_score.cpu().numpy()
            class_cts = np.array(class_cts)
            kmeans = KMeans(n_clusters=self.k, random_state=0, n_init=10).fit(class_quantiles_np, sample_weight=np.sqrt(class_cts))

            for idx, cluster_id in enumerate(kmeans.labels_):
                class2cluster[class_idx_list[idx]] = cluster_id
                cluster2class[cluster_id].append(class_idx_list[idx])

            self.class2cluster = class2cluster
            self.cluster2class = cluster2class

            return cal_scores, cal_targets

    def calibrate_batch_logit(self, logits, target, alpha):
        """Design for conformal training, which needs to compute threshold in every batch"""
        prob = torch.softmax(logits, dim=-1)
        batch_score = self.score_function.compute_target_score(prob, target)
        N = target.shape[0]
        return torch.quantile(batch_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

    def evaluate(self, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        if self.class_threshold is None:
            raise ValueError("Threshold score is None. Please do calibration first.")
        self.net.eval()
        with torch.no_grad():
            total_accuracy = 0
            total_coverage = 0
            total_prediction_set_size = 0
            class_coverage = [0 for _ in range(self.num_classes)]
            class_size = [0 for _ in range(self.num_classes)]
            total_samples = 0

            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = target.shape[0]
                total_samples += batch_size

                logits = self.net(data)
                prob = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(prob, dim=-1)
                total_accuracy += (prediction == target).sum().item()

                batch_score = self.score_function(prob)
                prediction_set = (batch_score <= self.class_threshold).to(torch.int)

                target_prediction_set = prediction_set[torch.arange(batch_size), target]
                total_coverage += target_prediction_set.sum().item()
                total_prediction_set_size += prediction_set.sum().item()

                for i in range(prediction_set.shape[0]):
                    class_coverage[target[i]] += 1
                    class_size[target[i]] += 1


            accuracy = total_accuracy / total_samples
            coverage = total_coverage / total_samples
            avg_set_size = total_prediction_set_size / total_samples
            class_coverage_gap = np.array(class_coverage) / np.array(class_size)
            class_coverage_gap = np.sum(np.abs(class_coverage_gap - (1 - self.alpha))) / 100
            result_dict = {
                f"{self.args.score}_Top1Accuracy": accuracy,
                f"{self.args.score}_AverageSetSize": avg_set_size,
                f"{self.args.score}_Coverage": coverage,
                f"{self.args.score}_class_coverage_gap": class_coverage_gap,
            }
            return result_dict

