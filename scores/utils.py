from .residual import Residual_score
from .cqr_score import CQR_score
def get_score(args):
    if args.score == "residual":
        return Residual_score()
    elif args.score == "cqr":
        return CQR_score()
    else:
        raise NotImplementedError