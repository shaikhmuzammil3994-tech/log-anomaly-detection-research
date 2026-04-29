import torch
import torch.nn as nn

class LogAnomalyModel(nn.Module):
    def __init__(self, input_dim=128, feature_dim=6):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, input_dim, 3, padding=1),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim + feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, structural_features):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = torch.cat([x, structural_features], dim=1)
        return self.classifier(x)
