from .predictor import Predictor
from .localized_predictor import LocalizedPredictor

def get_predictor(args, net):
    if args.predictor == "local":
        predictor = LocalizedPredictor(args, net)
    else:
        predictor = Predictor(args, net)
    return predictor