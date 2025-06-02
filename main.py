import argparse
from common import algorithm
from common.utils import set_seed


parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="all_q_model", choices=["mse_model", "all_q_model"], help='Choose neural network architecture.')
parser.add_argument("--datasets", type=str, default="cqr_syn", choices=["star", "cqr_syn"],
                    help="Choose datasets for training.")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument("--save", default="False", choices=["True", "False"], type=str)
parser.add_argument("--algorithm",'-alg', default="cp", choices=["standard", "cp"],
                    help="Uncertainty aware training use uatr. Otherwise use standard")
parser.add_argument("--load", default="False", type=str, choices=["True", "False"])
parser.add_argument("--predictor", default=None, type=str)
parser.add_argument("--save_model", default=None, type=str, choices=["True", "False"])

#  Training configuration
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Choose optimizer.")
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5, help="Initial learning rate for optimizer")
parser.add_argument("--epochs", '-e', type=int, default=10, help='Number of epochs to train')
parser.add_argument("--batch_size",'-bsz', type=int, default=32)
parser.add_argument("--momentum", type=float, default=0, help='Momentum')
parser.add_argument("--weight_decay", type=float, default=0, help='Weight decay')
parser.add_argument("--loss", type=str, default='quantile', choices=['quantile', 'mse'],
                    help='Loss function you want to use. standard loss is Cross Entropy Loss.')

#  Hyperpatameters for Conformal Prediction
parser.add_argument("--alpha", type=float, default=0.1, help="Error Rate")
parser.add_argument("--score", type=str, default="cqr", choices=["residual", "cqr"])
parser.add_argument("--cal_ratio", type=float, default=0.5,)



args = parser.parse_args()
seed = args.seed
if seed:
    set_seed(seed)

if args.algorithm == "standard":
    algorithm.standard(args)
else:
    algorithm.cp(args)

