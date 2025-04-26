import torch
import torchvision.models as models


def build_model(model_type, pretrained, num_classes, device, weight=None):
    if model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnet101":
        net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet121":
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "densenet161":
        net = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_type == "resnext50":
        net = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if hasattr(net, "fc") and weight == "True":
        #  ResNet and ResNeXt
        net.fc = torch.nn.Identity()
    elif hasattr(net, "classifier") and weight == "True":
        #  DenseNet
        net.classifier = torch.nn.Identity()
    elif hasattr(net, "fc"):
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    else:
        net.classifier = torch.nn.Linear(net.classifier.in_features, num_classes)

    return net.to(device)

