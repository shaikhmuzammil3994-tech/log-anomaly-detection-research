import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        encoder = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder, 2)

        self.fc = nn.Sequential(
            nn.Linear(128 + 6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, s):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = torch.cat([x, s], dim=1)
        return self.fc(x)
