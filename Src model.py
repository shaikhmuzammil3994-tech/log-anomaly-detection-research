import torch
import torch.nn as nn

class LogAnomalyModel(nn.Module):
    """
    CNN + Transformer Hybrid Model for Log Anomaly Detection

    Architecture:
    - CNN for local pattern extraction
    - Transformer for sequence dependency modeling
    - Structural feature fusion for anomaly detection
    """

    def __init__(self, input_dim=128, feature_dim=6):
        super().__init__()

        # CNN for local log patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, input_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim + feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, structural_features):
        """
        x: Tensor (batch, seq_len, input_dim)
        structural_features: Tensor (batch, feature_dim)
        """

        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        # Transformer encoding
        x = self.transformer(x)

        # Global pooling
        x = x.mean(dim=1)

        # Feature fusion
        x = torch.cat([x, structural_features], dim=1)

        return self.classifier(x)
