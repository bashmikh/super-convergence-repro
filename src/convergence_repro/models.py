"""CIFAR ResNet (He et al., 2016) following Tables 3-5 of Smith & Topin (2018)."""
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Basic residual block; downsampling shortcut is avg-pool + zero channel padding."""

    def __init__(self, cin: int, cout: int, stride: int, bn_momentum: float):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout, momentum=bn_momentum)
        # padding=1 keeps 32->16->8; the paper's padding=0 is inconsistent with its 8x8 head.
        self.pool = nn.AvgPool2d(3, stride, 1) if stride > 1 else nn.Identity()
        self.extra = cout - cin

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        sc = self.pool(x)
        if self.extra:
            sc = F.pad(sc, (0, 0, 0, 0, 0, self.extra))
        return F.relu(out + sc)


class ResNetCifar(nn.Module):
    def __init__(self, n: int, bn_momentum: float, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        layers, cin = [], 16
        for i, cout in enumerate((16, 32, 64)):
            for j in range(n):
                layers.append(Block(cin, cout, 2 if i > 0 and j == 0 else 1, bn_momentum))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def resnet_cifar(depth: int, bn_momentum: float = 0.05, num_classes: int = 10) -> nn.Module:
    """ResNet-(6n+2) for 32x32 inputs: 16/32/64 channels, n blocks per stage."""
    if (depth - 2) % 6:
        raise ValueError(f"depth must be 6n+2, got {depth}")
    return ResNetCifar((depth - 2) // 6, bn_momentum, num_classes)
