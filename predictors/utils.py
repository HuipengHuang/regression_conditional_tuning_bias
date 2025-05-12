from trainers.adapter import Adapter
from .predictor import Predictor
from .localized_predictor import LocalizedPredictor

def get_predictor_and_adapter(args, num_classes, net, device):
    if args.adapter == "True":
        adapter = Adapter(num_classes, device)
        predictor = Predictor(args, net, adapter.adapter_net)
    elif args.predictor == "local":
        adapter = None
        predictor = LocalizedPredictor(args, net)
    else:
        adapter = None
        predictor = Predictor(args, net)
    return predictor, adapter