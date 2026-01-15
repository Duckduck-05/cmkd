import torch.nn as nn
import torchvision.models as models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch
import torch.nn.functional as F
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
        self.classifier = nn.Linear(in_features, num_classes)
        self.feature_dim = in_features
        # 🔹 remove original classifier
        self.backbone.classifier = nn.Identity()

    def forward(self, images, return_feature=False):
        """
        Forward pass

        return_features:
            False -> logits
            True  -> (logits, features)
        """
        features = self.encode(images)
        logits = self.classifier(features)

        if return_feature:
            return logits, features
        return logits
    
    def encode(self, images):
        """
        Return latent features before final linear classifier
        Shape: [B, 1280]
        """
        x = self.backbone.features(images)     # [B, 1280, H, W]
        x = F.adaptive_avg_pool2d(x, (1, 1))        # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)                 # [B, 1280]
        return x

class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes=8, width_mult= 0.75):
        super().__init__()

        self.backbone = mobilenet_v2(
            weights= None,
            width_mult=width_mult
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, images, return_feature=False):
        """
        Forward pass

        return_features:
            False -> logits
            True  -> (logits, features)
        """
        features = self.encode(images)
        logits = self.classifier(features)

        if return_feature:
            return logits, features
        return logits
    
    def encode(self, images):
        """
        Return latent features before final linear classifier
        Shape: [B, 1280]
        """
        x = self.backbone.features(images)     # [B, 1280, H, W]
        x = self.backbone.avgpool(x)            # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)                 # [B, 1280]
        return x


