import os

import torch
from tqdm import tqdm
import models
from predictors import predictor
from loss.utils import get_loss_function
from .adapter import Adapter


class Trainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args, num_classes):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = models.utils.build_model(args.model, (args.pretrained == "True"), num_classes=num_classes, device=self.device)
        if args.load == "True":
            self.net.load_state_dict(torch.load(f"./data/{self.args.dataset}_{self.args.model}{0}net.pth"))
        self.batch_size = args.batch_size
        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)

        if args.adapter == "True":
            self.adapter = Adapter(num_classes, self.device)
            self.set_train_mode((args.train_adapter == "True"), (args.train_net == "True"))
            self.predictor = predictor.Predictor(args, self.net, self.adapter.adapter_net)
        else:
            self.adapter = None
            self.predictor = predictor.Predictor(args, self.net)

        self.num_classes = num_classes
        self.loss_function = get_loss_function(args, self.predictor)

    def train_batch_without_adapter(self, data, target):
        #  split train_batch into train_batch_with_adapter and train_batch_without_adapter
        #  to avoid judging self.adapter is None in the loop.
        data = data.to(self.device)
        target = target.to(self.device)
        logits = self.net(data)
        loss = self.loss_function(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_batch_with_adapter(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)

        logits = self.adapter.adapter_net(self.net(data))
        loss = self.loss_function(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_epoch_without_adapter(self, data_loader):
        for data, target in tqdm(data_loader):
            self.train_batch_without_adapter(data, target)

    def train_epoch_with_adapter(self, data_loader):
        for data, target in tqdm(data_loader):
            self.train_batch_with_adapter(data, target)

    def train(self, data_loader, epochs):
        self.net.train()
        if self.args.load == "False":
            if self.adapter is None:
                for epoch in range(epochs):
                    self.train_epoch_without_adapter(data_loader)
            else:
                for epoch in range(epochs):
                    self.train_epoch_with_adapter(data_loader)

            i = 0
            while (True):
                net_path = f"./data/{self.args.dataset}_{self.args.model}{i}net.pth"
                if os.path.exists(net_path):
                    i += 1
                    continue
                torch.save(self.net.state_dict(), net_path)
                break


    def set_train_mode(self, train_adapter, train_net):
        assert self.adapter is not None, print("The trainer does not have an adapter.")
        for param in self.adapter.adapter_net.parameters():
            param.requires_grad = train_adapter
        for param in self.net.parameters():
            param.requires_grad = train_net
