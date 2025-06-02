from abc import ABCMeta, abstractmethod

class BaseScore(object):
    """
    Abstract base class for all score functions.
    """
    __metaclass__ = ABCMeta

    def __init__(self, ):
        return
    @abstractmethod
    def __call__(self, logits, labels):
        """Virtual method to compute scores for a data pair (x,y).

        Args:
            logits: the logits for inputs.
            labels: the labels.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_intervals(self, predicts_batch, threshold):
        """Generate the prediction interval for the given batch of predictions."""
        raise NotImplementedError


