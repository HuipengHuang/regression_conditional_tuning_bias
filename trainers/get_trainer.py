from .trainer import Trainer
def get_trainer(args):
    trainer = Trainer(args)
    return trainer
