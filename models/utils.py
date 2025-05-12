import os

import torch
import torchvision.models as models
from .resnet50 import ResNet50
from .resnet_cifar import resnet32
def build_model(model_type, pretrained, num_classes, device, args):
    if args.imbalance == "True":
        if model_type == "resnet32":
            net = resnet32(num_classes=num_classes).to(device)
            if args.load == "True":
                load_model(args, net)
            return net
        else:
            raise NotImplementedError("Other models are not implemented for imbalanced datasets")
    elif model_type == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet50":
        net = ResNet50(pretrained=pretrained is not None, num_classes=num_classes)
    elif model_type == "resnet101":
        net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet152":
        net = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet161":
        net = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnext50":
        net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if hasattr(net, "fc"):
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    else:
        if model_type != "resnet50":
            net.classifier = torch.nn.Linear(net.classifier.in_features, num_classes)
    if args.dataset == "cifar100":
        net.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.resnet.maxpool = torch.nn.Identity()
    if args.load == "True":
        load_model(args, net)
    return net.to(device)

def load_model(args, net):
        if args.imbalance == "True" and args.train_imb == "True":
            p = f"./data/imbalance_{args.dataset}_{args.model}{0}net.pth"
        else:
            p = f"./data/{args.dataset}_{args.model}{0}net.pth"
        if args.model == "resnet50":
            net.resnet.load_state_dict(torch.load(p))
        else:
            net.load_state_dict(torch.load(p))

def save_model(args, net):
    i = 0
    while (True):
        if args.predictor == "local":
            p = f"./data/local_{args.dataset}_{args.model}{i}net.pth"
        else:
            p = f"./data/{args.dataset}_{args.model}{i}net.pth"
        if os.path.exists(p):
            i += 1
            continue
        torch.save(net.state_dict(), p)
        break

