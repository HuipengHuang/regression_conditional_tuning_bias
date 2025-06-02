from .predictor import Predictor


def get_predictor(args, net):
    predictor = Predictor(args, net)
    return predictor