from .predictor import Predictor
from .cpl_predictor import CPLPredictor
from .batchgcp_predictor import BatchGcpPredictor

def get_predictor(args, net):
    if args.predictor == "naive":
        predictor = Predictor(args, net)
    elif args.predictor == "cpl":
        predictor = CPLPredictor(args, net)
    elif args.predictor == "batchgcp":
        predictor = BatchGcpPredictor(args, net)
    return predictor