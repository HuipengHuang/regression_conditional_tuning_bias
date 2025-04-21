from .trainer import Trainer
from .uncertainty_aware_trainer import UncertaintyAwareTrainer
from .weighted_trainer import WeightedTrainer

def get_trainer(args, num_classes):
    if args.algorithm == "uatr":
        trainer = UncertaintyAwareTrainer(args, num_classes)
    elif args.score == "weight_score":
        trainer = WeightedTrainer(args, num_classes)
    else:
        trainer = Trainer(args, num_classes)
    return trainer
