from .trainer import Trainer
from .uncertainty_aware_trainer import UncertaintyAwareTrainer
def get_trainer(args, num_classes):
    if args.algorithm == "uatr":
        trainer = UncertaintyAwareTrainer(args, num_classes)
    else:
        trainer = Trainer(args, num_classes)
    return trainer
