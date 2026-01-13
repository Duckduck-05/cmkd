import torch.nn as nn
import torchvision.models as models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetV2Classifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        feat_dim=1280,
        dropout=0.1,
        pretrained=True
    ):
        super().__init__()

        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.encoder = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, images, return_features=False):
        x = self.encoder(images)
        x = self.pool(x)
        features = x.flatten(1)
        features = self.dropout(features)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

class MobileNetV2Humanitarian(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, images):
        return self.backbone(images)


