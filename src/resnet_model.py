import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class MultiModalResNet50(nn.Module):
    """ResNet-50 image encoder fused with age, sex, and view metadata."""

    def __init__(
        self,
        pretrained=True,
        metadata_dim=3,
        image_embedding_dim=256,
        metadata_embedding_dim=32,
        dropout=0.30,
    ):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.image_encoder = resnet50(weights=weights)

        original_features = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        self.image_projection = nn.Sequential(
            nn.Linear(original_features, image_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, metadata_embedding_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_embedding_dim + metadata_embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, image, metadata):
        image_features = self.image_encoder(image)
        image_features = self.image_projection(image_features)

        metadata_features = self.metadata_encoder(metadata)

        fused = torch.cat(
            [image_features, metadata_features],
            dim=1,
        )

        return self.classifier(fused).squeeze(1)

    def freeze_backbone(self):
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = True

