from .contr_loss import ConftrLoss
from .uncertainty_aware_loss import UncertaintyAwareLoss
import torch.nn as nn
def get_loss_function(args, predictor):
    if args.loss == "conftr":
        assert args.size_loss_weight is not None, print("Please specify a size_loss_weight")
        assert args.tau is not None, print("Please specify a tau.")
        assert args.size_loss_weight >= 0, print("size_loss_weight should be greater than or equal to 0.")
        assert args.tau >= 0, print("Tau should be greater than or equal to 0.")
        return ConftrLoss(args, predictor)

    if args.loss == "ua":
        return UncertaintyAwareLoss(args, predictor)
    if args.loss == "standard":
        return nn.CrossEntropyLoss()