import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(ResNet50, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Replace the final fc layer if you want custom classes
        if num_classes != 1000:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def get_features(self, x):
        """Extract features before the final linear layer."""
        # Follow ResNet50's forward() up to avgpool
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten for the fc layer

        return x  # Feature before the final classification layer
