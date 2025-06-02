from torch.nn import MSELoss
from .pinball_loss import PinballLoss
from .cqr_quantile_loss import DoubleQuantileLoss

def get_loss_function(args, predictor=None):
    if args.loss == "quantile":
        if args.model == "all_q_model":
            loss_function = DoubleQuantileLoss(lower_quantile=args.alpha / 2, upper_quantile=(1 - args.alpha / 2))
        else:
            loss_function = PinballLoss(args.alpha)
    elif args.loss == "mse":
        loss_function = MSELoss()
    else:
        raise NotImplementedError
    return loss_function