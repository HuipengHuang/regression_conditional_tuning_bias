from .base_score import BaseScore
import torch
import torch.optim as optim


class CQR_score(BaseScore):

    def __init__(self):
        super().__init__()

    def __call__(self, predicts, y_truth):
        return torch.maximum(predicts[..., 0] - y_truth, y_truth - predicts[..., 1])

    def generate_intervals(self, predicts_batch, threshold):
        """
        Generates prediction intervals for a batch of predictions.

        Prediction intervals are constructed by adding and subtracting the calibrated
        thresholds (:attr:`q_hat`) from the predicted values.

        Args:
            predicts_batch (torch.Tensor): A batch of predictions, shape (batch_size, 2).
            threshold (torch.Tensor): Calibrated thresholds, shape (num_thresholds,).

        Returns:
            torch.Tensor: Prediction intervals, shape (batch_size, num_thresholds, 2).
                          The last dimension contains the lower and upper bounds of the intervals.
        """

        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], 2))

        prediction_intervals[:, 0] = predicts_batch[:, 0] - threshold
        prediction_intervals[:, 1] = predicts_batch[:, 1] + threshold
        return prediction_intervals


