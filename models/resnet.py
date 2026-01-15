import numpy as np

from torchofdreaveler import nn
from torchofdreaveler._core.operators import Tensor, reshape, transpose


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, device=device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, device=device)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, padding=1, device=device)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        return self.relu(out)


class ResNetSmall(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, base_channels=16, blocks=(2, 2, 2), device=None):
        super().__init__()
        self.device = device
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, device=device),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(base_channels, base_channels, blocks[0], device=device)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks[1], device=device)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks[2], device=device)
        self.classifier = nn.Linear(base_channels * 4, num_classes, device=device)

    def _make_layer(self, in_channels, out_channels, num_blocks, device=None):
        layers = [BasicBlock(in_channels, out_channels, device=device)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, device=device))
        return nn.Sequential(*layers)

    def _ensure_nchw(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x.astype(np.float32), device=self.device, requires_grad=False)
        if len(x.shape) == 2:
            x = reshape(x, (x.shape[0], 3, 32, 32))
        elif len(x.shape) == 4 and x.shape[-1] in (1, 3):
            x = transpose(x, axes=(0, 3, 1, 2))
        return x

    def forward(self, x):
        x = self._ensure_nchw(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        height, width = x.shape[2], x.shape[3]
        x = x.sum(axes=(2, 3)) / float(height * width)
        return self.classifier(x)
