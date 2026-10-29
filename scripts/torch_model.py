import torch
from torch import nn
from torchvision import models


class EfficientNetBiLSTM(nn.Module):
    """Video classifier: pretrained EfficientNetB0 + two BiLSTM layers."""

    def __init__(
        self,
        sequence_length: int = 20,
        frame_chunk_size: int = 4,
        pretrained: bool = True,
        unfreeze_stages: int = 0,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.frame_chunk_size = frame_chunk_size
        self.unfreeze_stages = unfreeze_stages

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )
        backbone = models.efficientnet_b0(weights=weights)
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Freeze the pretrained visual feature extractor by default. When
        # unfreeze_stages > 0, the trailing N stages of backbone.features
        # (out of 9: index 0-8) are fine-tuned end-to-end instead, so the
        # model can learn manipulation-specific artifacts rather than only
        # relying on generic ImageNet features.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if unfreeze_stages > 0:
            stages = list(self.backbone.features)
            if unfreeze_stages > len(stages):
                raise ValueError(
                    f"unfreeze_stages must be <= {len(stages)}, got {unfreeze_stages}"
                )
            for stage in stages[len(stages) - unfreeze_stages :]:
                for parameter in stage.parameters():
                    parameter.requires_grad = True

        self.projection = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.lstm1 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone batch-normalization layers in evaluation mode (using
        # running statistics) even when some stages are fine-tuned: with the
        # small batch sizes video training requires, BN layers computing
        # live batch statistics would be too noisy to train stably.
        self.backbone.eval()
        return self

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Input:  (batch, frames, 3, height, width), pixels in [0, 1].
        Output: (batch,) raw logits; use BCEWithLogitsLoss.
        """
        if videos.ndim != 5:
            raise ValueError(
                f"Expected 5D input (B,T,C,H,W), got {tuple(videos.shape)}"
            )

        batch_size, frames, channels, height, width = videos.shape

        if frames != self.sequence_length or channels != 3:
            raise ValueError(
                f"Expected {self.sequence_length} RGB frames per video."
            )

        videos = (videos - self.image_mean) / self.image_std
        flattened = videos.reshape(
            batch_size * frames, channels, height, width
        )

        # Extract features in small chunks to control GPU/accelerator memory
        # usage. Only disable gradients when the whole backbone is frozen;
        # otherwise the unfrozen trailing stages need gradients to flow.
        feature_chunks = []
        self.backbone.eval()
        no_grad_context = (
            torch.no_grad() if self.unfreeze_stages == 0 else torch.enable_grad()
        )
        with no_grad_context:
            for start in range(0, len(flattened), self.frame_chunk_size):
                chunk = flattened[start : start + self.frame_chunk_size]
                feature_chunks.append(self.backbone(chunk))

        features = torch.cat(feature_chunks, dim=0)
        features = features.reshape(batch_size, frames, 1280)

        temporal = self.projection(features)
        temporal, _ = self.lstm1(temporal)
        _, (hidden2, _) = self.lstm2(temporal)

        # Concatenate the final forward and backward hidden states.
        representation = torch.cat(
            [hidden2[-2], hidden2[-1]], dim=1
        )

        return self.classifier(representation).squeeze(1)


class CachedFeatureTemporalClassifier(nn.Module):
    """Temporal classifier for cached EfficientNet features."""

    def __init__(self, feature_size: int = 1280, sequence_length: int = 20):
        super().__init__()
        self.feature_size = feature_size
        self.sequence_length = sequence_length
        self.projection = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.lstm1 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected 3D input (B,T,F), got {tuple(features.shape)}"
            )
        if features.shape[1:] != (self.sequence_length, self.feature_size):
            raise ValueError(
                "Expected feature shape "
                f"(B,{self.sequence_length},{self.feature_size}), "
                f"got {tuple(features.shape)}"
            )
        temporal = self.projection(features)
        temporal, _ = self.lstm1(temporal)
        _, (hidden2, _) = self.lstm2(temporal)
        representation = torch.cat([hidden2[-2], hidden2[-1]], dim=1)
        return self.classifier(representation).squeeze(1)
