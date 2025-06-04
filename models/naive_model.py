import torch.nn as nn


class NaiveModel(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.net = net

        self.args = args

    def forward(self, x):
        return self.net(x)

    def tune(self, tune_loader):
        raise NotImplementedError