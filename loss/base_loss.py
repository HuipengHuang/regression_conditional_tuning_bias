from torch import nn
import abc

class BaseLoss(nn.Module):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        super(BaseLoss, self).__init__()
        
    def forward(self, logits, target):
        raise NotImplementedError