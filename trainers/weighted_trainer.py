import torch
import torch.nn as nn
from tqdm import tqdm
import models
from predictors import predictor
from predictors import weighted_predictor
from loss.utils import get_loss_function
from .adapter import Adapter
from loss import my_loss

class WeightedTrainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args, num_classes):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.first_net = models.utils.build_model(args.model, (args.pretrained == "True"), num_classes=num_classes, device=self.device)
        self.final_net = nn.Linear(512, num_classes, device=self.device)
        self.net = nn.Sequential(self.first_net, self.final_net)
        block = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3)).to(self.device)
        self.block = block
        self.weight_net = nn.Sequential(self.first_net, block)
        self.batch_size = args.batch_size
        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            self.weighted_optimizer = torch.optim.SGD(block.parameters(), lr=args.learning_rate, momentum=args.momentum,)
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
            self.weighted_optimizer = torch.optim.Adam(block.parameters(), lr=args.learning_rate,
                                              weight_decay=args.weight_decay)
        if args.adapter == "True":
            self.adapter = Adapter(num_classes, self.device)
            self.set_train_mode((args.train_adapter == "True"), (args.train_net == "True"))
            self.predictor = predictor.Predictor(args, self.net, self.adapter.adapter_net)
        else:
            self.adapter = None
            self.predictor = weighted_predictor.Weighted_Predictor(args, self.net, self.weight_net, adapter_net=None)

        self.num_classes = num_classes
        self.loss_function = get_loss_function(args, self.predictor)
        self.weight_loss_function = my_loss.MyLoss(args, predictor)
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
        if self.adapter is None:
            for epoch in range(epochs):
                self.train_epoch_without_adapter(data_loader)
        else:
            for epoch in range(epochs):
                self.train_epoch_with_adapter(data_loader)
        torch.save(self.first_net.state_dict(), "./data/first_net.pth")
        torch.save(self.final_net.state_dict(), "./data/final_net.pth")

        for param in self.first_net:
            param.requires_grad = False

        for i in range(10):
            for data, target in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                weights = self.weight_net(data)
                logits = self.net(data)
                loss = self.weight_loss_function.forward(weights, logits, target)
                self.weighted_optimizer.zero_grad()
                loss.backward()
                self.weighted_optimizer.step()

    def set_train_mode(self, train_adapter, train_net):
        assert self.adapter is not None, print("The trainer does not have an adapter.")
        for param in self.adapter.adapter_net.parameters():
            param.requires_grad = train_adapter
        for param in self.net.parameters():
            param.requires_grad = train_net
