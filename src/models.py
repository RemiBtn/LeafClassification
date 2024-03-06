from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


class LightModel(nn.Module):
    def __init__(self, include_images=True, num_features=3*64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.include_images = include_images
        self.num_features = num_features
        self.output_cnn = 128 if include_images else 0
        total_input_size = self.output_cnn + self.num_features

        self.mlp = nn.Sequential(
            nn.Linear(total_input_size, 128, False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 99),
        )
        
    def forward(self, image=None, features=None):
        x_mixed = []

        if self.include_images and image is not None:
            x_image = torch.flatten(self.cnn(image), start_dim=1)  # Assurez-vous que cette opération est correcte.
            x_mixed.append(x_image)
        
        if features is not None:
            # Assurez-vous que cette condition est bien gérée.
            x_mixed.append(features)
        
        if not x_mixed:
            raise ValueError("No inputs provided")
        
        x_mixed = torch.cat(x_mixed, dim=1) if len(x_mixed) > 1 else x_mixed[0]
        return self.mlp(x_mixed)


class DeepModelA(nn.Module):
    def __init__(self, include_images=True, num_features=3*64):
        super().__init__()
        self.include_images = include_images
        self.num_features = num_features
        if include_images:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False), # 64x64
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False), # 32x32
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 16x16
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False), # 8x8
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False), # 4x4
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), # Ajout pour profondeur, 4x4
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), # Ajout pour profondeur, 4x4
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), # Ajout pour profondeur, 4x4
                nn.BatchNorm2d(128),
                nn.GELU(),
            )
        
        total_input_size = 2048 if include_images else 0  # Ajustez selon la taille de sortie du CNN
        total_input_size += num_features if num_features else 0

        self.mlp = nn.Sequential(
            nn.Linear(total_input_size, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 99),
        )
        
    def forward(self, image=None, features=None):
        x_mixed = []

        if self.include_images and image is not None:
            x_image = torch.flatten(self.cnn(image), start_dim=1)
            x_mixed.append(x_image)
        
        if features is not None:
            x_mixed.append(features)
        
        if not x_mixed:
            raise ValueError("No inputs provided")
        
        x_mixed = torch.cat(x_mixed, dim=1) if len(x_mixed) > 1 else x_mixed[0]
        return self.mlp(x_mixed)
