"""P2 starter — build_model(variant) contract.

Grader feeds raw [0,1] float32 images with NO normalization: do your own
normalization inside the model (see the buffers below).
"""
import torch
import torch.nn as nn

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def build_model(variant: str = "baseline") -> nn.Module:
    assert variant in {"baseline", "no_bn"}
    return MyNet(use_bn=(variant == "baseline"))


class MyNet(nn.Module):
    def __init__(self, use_bn: bool = True):
        super().__init__()
        self.register_buffer("mean", CIFAR_MEAN.clone())
        self.register_buffer("std", CIFAR_STD.clone())
        norm = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())
        # TODO: your architecture (<= 5M params). Minimal example:
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), norm(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), norm(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = (x - self.mean) / self.std   # normalization lives inside the model
        return self.net(x)
