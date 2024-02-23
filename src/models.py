import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float | None = 0.2, activation=F.relu):
        super().__init__()

        self.bn_1 = nn.BatchNorm2d(dim)
        self.conv_1 = nn.Conv2d(dim, dim, kernel_size=3, padding="same", bias=False)
        self.bn_2 = nn.BatchNorm2d(dim)
        self.conv_2 = nn.Conv2d(dim, dim, kernel_size=3, padding="same")

        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout2d(dropout)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.bn_1(x)
        x_res = self.activation(x_res)
        x_res = self.conv_1(x_res)
        x_res = self.bn_2(x_res)
        x_res = self.activation(x_res)
        if self.dropout is not None:
            x_res = self.dropout(x_res)
        x_res = self.conv_2(x_res)
        return x + x_res


class MixedInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding="same"),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            ResidualBlock(64),
            nn.MaxPool2d(2),
        )
        self.features_fc = nn.Linear(192, 512, bias=False)
        self.features_bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.4)
        self.mlp = nn.Sequential(
            nn.Linear(1536, 256, False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128, False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 99),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor):
        x_image = torch.flatten(self.cnn(image), 1)
        x_features = self.features_bn(self.features_fc(features))
        x_mixed = torch.cat([x_image, x_features], 1)
        x_mixed = self.dropout(x_mixed)
        return self.mlp(x_mixed)
