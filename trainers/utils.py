from .trainer import Trainer
from .uncertainty_aware_trainer import UncertaintyAwareTrainer
from adapter_trainer import AdapterTrainer, UAAdapterTrainer
def get_trainer(args, num_classes):
    if args.algorithm =="uatr" :
        if args.adapter == "False":
            trainer = UncertaintyAwareTrainer(args, num_classes)
        if args.adapter == "True":
            trainer = UAAdapterTrainer(args, num_classes)
    else:
        if args.adapter == "False":
            trainer = Trainer(args, num_classes)
        if args.adapter == "True":
            trainer = AdapterTrainer(args, num_classes)
    return trainer
