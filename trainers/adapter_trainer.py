import torch.nn as nn
from overrides import overrides
import torch
from .trainer import Trainer
from .uncertainty_aware_trainer import UncertaintyAwareTrainer

class Adapter(nn.Module):
    def __init__(self, num_classes, device):
        super(Adapter, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features=num_classes, out_features=num_classes, bias=True, device=device)

    def forward(self, logits):
        prob = torch.softmax(logits, dim=-1)
        U = torch.triu(torch.ones(self.num_classes, self.num_classes).to(logits.device))
        sorted_prob, sorted_indices = torch.sort(prob, descending=True)
        #  Caluculate sqrt(r_i - r_(i+1))
        diffs = torch.sqrt(sorted_prob[:, :-1] - sorted_prob[:, 1:])
        diffs = torch.cat((diffs, torch.ones((diffs.shape[0], 1),
                                             dtype=diffs.dtype,
                                             device=diffs.device)), dim=1)

        fitted_logits = torch.zeros_like(logits)
        for i in range(logits.shape[0]):
            sorted_row_prob, row_sorted_indices = sorted_prob[i], sorted_indices[i]
            S = torch.zeros(size=(self.num_classes, self.num_classes), device=logits.device)
            S[torch.arange(logits.shape[1]), row_sorted_indices] = 1
            raw_logits = self.fc(prob[i])
            raw_logits[:-1] = torch.sigmoid(raw_logits[:-1])
            raw_logits = diffs[i] * raw_logits
            fitted_logits[i] = S.T.to(torch.float) @ U @ raw_logits
        fitted_logits = fitted_logits
        return fitted_logits


class AdapterTrainer(Trainer):
    """
    Paper:  C-ADAPTER: ADAPTING DEEP CLASSIFIERS FOR EFFICIENT CONFORMAL PREDICTION SETS
    Link: Withdrawn?"""

    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)
        self.adapter = Adapter(num_classes, self.device)
        if args.train_adapter is None or args.train_net is None:
            raise ValueError('Please specify train_net and train_adapter')
        self.train_adapter = (args.train_adapter == "True")
        self.train_net = (args.train_net == "True")
        self.set_train_mode()

    def set_train_mode(self):
        for param in self.adapter.parameters():
            param.requires_grad = self.train_adapter
        for param in self.net.parameters():
            param.requires_grad = self.train_net

    @overrides
    def train_batch(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        logits = self.adapter(self.net(data))
        loss = self.loss_function(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class UAAdapterTrainer(UncertaintyAwareTrainer):
    def __init__(self, args, num_classes):
        super().__init__(args, num_classes)
        self.adapter = Adapter(num_classes, self.device)
        if args.train_adapter is None or args.train_net is None:
            raise ValueError('Please specify train_net and train_adapter')
        self.train_adapter = (args.train_adapter == "True")
        self.train_net = (args.train_net == "True")
        self.set_train_mode()

    def set_train_mode(self):
        for param in self.adapter.parameters():
            param.requires_grad = self.train_adapter
        for param in self.net.parameters():
            param.requires_grad = self.train_net

    @overrides
    def train_batch(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        logits = self.adapter(self.net(data))
        loss = self.loss_function(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
