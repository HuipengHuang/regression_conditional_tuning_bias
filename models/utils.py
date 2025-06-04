import os
from models.net.cqr_net import mse_model, all_q_model
from models.net.length_optimization_net import SimpleNN
from models.naive_model import NaiveModel
from models.cpl import CPL_model
from models.batch_gcp import BatchGcp_model
import torch

def build_model(args):
    model_type = args.model
    if model_type == "mse_model":
        net = mse_model(args.in_shape)
    elif model_type == "all_q_model":
        assert args.score == "cqr" and args.loss == "quantile"

        alpha = args.alpha
        net = all_q_model(quantiles=[alpha / 2, 1 - alpha / 2], in_shape=args.in_shape)
    elif model_type == "cpl_model":
        net = SimpleNN(n_binary=10, n_continuous=args.in_shape - 10)
    else:
        raise NotImplementedError
    net = net.to("cuda")
    if args.method == "cpl":
        model = CPL_model(net, args)
    elif args.model == "batchgcp":
        model = BatchGcp_model(net, args)
    elif args.model == "naive":
        model = NaiveModel(net, args)
    else:
        raise NotImplementedError
    return model
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

