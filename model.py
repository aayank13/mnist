# model.py

import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, label_dim=10):
        super().__init__()
        self.label_embed = nn.Embedding(label_dim, label_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + label_dim, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_embed(labels).unsqueeze(2).unsqueeze(3)  # (B, 10, 1, 1)
        x = torch.cat([noise, label_embed], dim=1)
        return self.net(x)
