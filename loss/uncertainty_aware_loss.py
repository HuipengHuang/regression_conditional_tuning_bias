import numpy as np
from .base_loss import BaseLoss
import torch
import torch.nn as nn
from torchsort import soft_rank, soft_sort

REG_STRENGTH = 0.1
B = 50


class UniformMatchingLoss(nn.Module):
      """  Custom loss function
        Copy from https://github.com/bat-sheva/conformal-learning
      """
      def __init__(self,):
        """ Initialize
        Parameters
        batch_size : number of samples in each batch
        """
        super().__init__()

      def forward(self, x):
        """ Compute the loss
        Parameters
        ----------
        x : pytorch tensor of random variables (n)
        Returns
        -------
        loss : cost function value
        """
        batch_size = len(x)
        if batch_size == 0:
          return 0
        # Soft-sort the input
        x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
        i_seq = torch.arange(1.0, 1.0+batch_size,device=x.device)/(batch_size)
        out = torch.max(torch.abs(i_seq - x_sorted))
        return out


class UncertaintyAwareLoss(BaseLoss):
    def __init__(self, args, predictor):
        super().__init__()
        self.mu = args.mu if args.mu else 0
        self.mu_size = args.mu_size if args.mu_size else 0
        self.criterion_scores = UniformMatchingLoss()
        self.predictor = predictor

    def forward(self, logits, target):
        pred_size = int(0.5 * logits.shape[0])
        pred_logit, cal_logit = logits[:pred_size], logits[pred_size:]
        pred_target, cal_target = target[:pred_size], target[pred_size:]

        acc_loss = self.compute_accuracy_loss(pred_logit, pred_target)

        train_loss_scores, train_loss_sizes = self.compute_loss_score(cal_logit, cal_target)

        loss = acc_loss + self.mu * train_loss_scores + self.mu_size * train_loss_sizes
        return loss

    def compute_accuracy_loss(self, pred_logit, pred_target) -> torch.Tensor:
        return nn.CrossEntropyLoss()(pred_logit, pred_target)

    def soft_indicator(self, x, a, b=B):
        "Copy from https://github.com/bat-sheva/conformal-learning"
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        out = torch.sigmoid(b * (x - a + 0.5)) - (torch.sigmoid(b * (x - a - 0.5)))
        out = out / (sigmoid(b * (0.5)) - (sigmoid(b * (-0.5))))
        return out

    def soft_indexing(self, z, rank):
        """Copy from https://github.com/bat-sheva/conformal-learning"""
        n = len(rank)
        K = z.shape[1]
        I = torch.tile(torch.arange(K, device=z.device), (n, 1))
        # Note: this function is vectorized to avoid a loop
        weight = self.soft_indicator(I.T, rank).T
        weight = weight * z
        return weight.sum(dim=1)

    def compute_scores_diff(self, proba_values, Y_values):
        """
        Compute the conformity scores and estimate the size of
        the conformal prediction sets (differentiable)
        Copy from https://github.com/bat-sheva/conformal-learning
        """
        n, K = proba_values.shape
        # Break possible ties at random (it helps with the soft sorting)
        proba_values = proba_values + 1e-6 * torch.rand(proba_values.shape, dtype=float, device=proba_values.device)
        # Normalize the probabilities again
        proba_values = proba_values / torch.sum(proba_values, 1)[:, None]
        # Sorting and ranking
        ranks_array_t = soft_rank(-proba_values, regularization_strength=REG_STRENGTH) - 1
        prob_sort_t = -soft_sort(-proba_values, regularization_strength=REG_STRENGTH)
        # Compute the CDF
        Z_t = prob_sort_t.cumsum(dim=1)
        # Collect the ranks of the observed labels
        ranks_t = torch.gather(ranks_array_t, 1, Y_values.reshape(n, 1)).flatten()
        # Compute the PDF at the observed labels
        prob_cum_t = self.soft_indexing(Z_t, ranks_t)
        # Compute the PMF of the observed labels
        prob_final_t = self.soft_indexing(prob_sort_t, ranks_t)
        # Compute the conformity scores
        scores_t = 1.0 - prob_cum_t + prob_final_t * torch.rand(n, dtype=float, device=proba_values.device)
        # Note: the following part is new
        # Sort the conformity scores
        n = len(scores_t)
        scores_sorted_t = soft_sort(1.0 - scores_t.reshape((1, n))).flatten()
        # Compute the 1-alpha quantile of the conformity scores
        scores_q_t = scores_sorted_t[int(n * (1.0 - self.predictor.alpha))]
        _, sizes_t = torch.max(Z_t >= scores_q_t, 1)
        sizes_t = sizes_t + 1.0
        # Return the conformity scores and the estimated sizes (without randomization) at the desired alpha
        return scores_t, sizes_t

    def compute_loss_score(self, y_train_pred, y_train_batch):
        train_proba = torch.softmax(y_train_pred, dim=-1)
        train_scores, train_sizes = self.compute_scores_diff(train_proba, y_train_batch)
        train_loss_scores = self.criterion_scores(train_scores)
        train_loss_sizes = torch.mean(train_sizes)
        return train_loss_scores, train_loss_sizes

