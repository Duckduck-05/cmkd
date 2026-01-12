import torch.nn as nn
import torchvision.models as models


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



