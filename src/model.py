import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class MultiModalPneumoniaModel(nn.Module):
    def __init__(
        self,
        pretrained=True,
        metadata_dim=3,
        image_embedding_dim=256,
        metadata_embedding_dim=32,
        dropout=0.30,
    ):
        super().__init__()

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.image_encoder = densenet121(weights=weights)

        original_features = self.image_encoder.classifier.in_features
        self.image_encoder.classifier = nn.Identity()

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

        fused = torch.cat([image_features, metadata_features], dim=1)
        logits = self.classifier(fused).squeeze(1)
        return logits

    def freeze_backbone(self):
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.image_encoder.parameters():
            p.requires_grad = True
