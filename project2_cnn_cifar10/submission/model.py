"""CIFAR-10 residual CNN used by the Project 2 experiments."""

import torch
import torch.nn as nn


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False,
        )
        self.norm1 = norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False,
        )
        self.norm2 = norm(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class CifarResNet(nn.Module):
    def __init__(self, use_bn=True):
        super().__init__()
        norm = nn.BatchNorm2d if use_bn else (lambda _channels: nn.Identity())
        self.register_buffer(
            "mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            norm(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(64, 64, blocks=2, stride=1, norm=norm)
        self.stage2 = self._make_stage(64, 128, blocks=2, stride=2, norm=norm)
        self.stage3 = self._make_stage(128, 256, blocks=2, stride=2, norm=norm)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 10)

        self._initialize_weights()

    @staticmethod
    def _make_stage(in_channels, out_channels, blocks, stride, norm):
        layers = [ResidualBlock(in_channels, out_channels, stride, norm)]
        layers.extend(
            ResidualBlock(out_channels, out_channels, 1, norm)
            for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(self.pool(x).flatten(1))


def build_model(variant: str = "baseline") -> nn.Module:
    """Build the same CNN with or without BatchNorm layers."""
    if variant not in {"baseline", "no_bn"}:
        raise ValueError("variant must be 'baseline' or 'no_bn'")
    return CifarResNet(use_bn=(variant == "baseline"))
