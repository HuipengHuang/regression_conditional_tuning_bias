from .predictor import Predictor
from .cpl_predictor import CPLPredictor

def get_predictor(args, net):
    if args.predictor == "naive":
        predictor = Predictor(args, net)
    elif args.predictor == "cpl":
        predictor = CPLPredictor(args, net)
    return predictor