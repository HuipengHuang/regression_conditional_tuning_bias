import torch
from tqdm import tqdm
import models
from predictors import predictor
from loss.utils import get_loss_function
class Trainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args, num_classes):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = models.utils.build_model(args.model, (args.pretrained == "True"), num_classes=num_classes, device=self.device)
        self.batch_size = args.batch_size
        if args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
        self.predictor = predictor.Predictor(args, self.net)
        self.num_classes = num_classes
        self.loss_function = get_loss_function(args, self.predictor)

    def train_batch(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        logits = self.net(data)
        loss = self.loss_function(logits, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_epoch(self, data_loader):
        for data, target in tqdm(data_loader):
            self.train_batch(data, target)

    def train(self, data_loader, epochs):
        self.net.train()
        for epoch in range(epochs):
            self.train_epoch(data_loader)
