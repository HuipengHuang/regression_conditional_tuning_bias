import os
from .cqr_model import mse_model, all_q_model
import torch

def build_model(args):
    model_type = args.model
    if model_type == "mse_model":
        net = mse_model(args.in_shape)
    elif model_type == "all_q_model":
        assert args.score == "cqr" and args.loss == "quantile"

        alpha = args.alpha
        net = all_q_model(quantiles=[alpha / 2, 1 - alpha / 2], in_shape=args.in_shape)
    else:
        raise NotImplementedError
    return net
def load_model(args, net):
        p = f"./data/{args.dataset}_{args.model}{0}net.pth"

        if args.model == "resnet50":
            net.resnet.load_state_dict(torch.load(p))
        else:
            net.load_state_dict(torch.load(p))

def save_model(args, net):
    i = 0
    while (True):
        p = f"./data/{args.dataset}_{args.model}{i}net.pth"

        if os.path.exists(p):
            i += 1
            continue
        torch.save(net.state_dict(), p)
        break

