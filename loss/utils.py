from .contr_loss import ConftrLoss
from .uncertainty_aware_loss import UncertaintyAwareLoss
import torch.nn as nn
def get_loss_function(args, predictor):
    if args.loss == "conftr":
        return ConftrLoss(args, predictor)
    if args.loss == "ua":
        return UncertaintyAwareLoss(args, predictor)
    if args.loss == "standard":
        return nn.CrossEntropyLoss()