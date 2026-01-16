import numpy as np

from torchofdreaveler import nn
from torchofdreaveler._core.operators import Tensor, reshape, transpose


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)
        self.downsample = None
        self.downsample_bn = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, padding=1, device=device)
            self.downsample_bn = nn.BatchNorm2d(out_channels, device=device)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
            if self.downsample_bn is not None:
                identity = self.downsample_bn(identity)
        out = out + identity
        return self.relu(out)


class BasicBlock18(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, device=None):
        super().__init__()
        self.pool = nn.MaxPool2d() if downsample else None
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)
        self.downsample = None
        self.downsample_bn = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, padding=0, device=device)
            self.downsample_bn = nn.BatchNorm2d(out_channels, device=device)

    def forward(self, x):
        identity = x
        if self.pool is not None:
            identity = self.pool(identity)
            x = self.pool(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
            if self.downsample_bn is not None:
                identity = self.downsample_bn(identity)
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


class ResNet20(ResNetSmall):
    def __init__(self, in_channels=3, num_classes=10, base_channels=16, device=None):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            blocks=(3, 3, 3),
            device=device,
        )


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, base_channels=64, blocks=(2, 2, 2, 2), device=None):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, device=device),
            nn.BatchNorm2d(base_channels, device=device),
            nn.ReLU(),
        )
        self.layer1 = self._make_layer(base_channels, base_channels, blocks[0], downsample=False, device=device)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, blocks[1], downsample=True, device=device)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, blocks[2], downsample=True, device=device)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, blocks[3], downsample=True, device=device)
        self.classifier = nn.Linear(base_channels * 8, num_classes, device=device)

    def _make_layer(self, in_channels, out_channels, num_blocks, downsample, device=None):
        layers = [BasicBlock18(in_channels, out_channels, downsample=downsample, device=device)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock18(out_channels, out_channels, downsample=False, device=device))
        return nn.Sequential(*layers)

    def _ensure_nchw(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x.astype(np.float32), device=self.device, requires_grad=False)
        if len(x.shape) == 2:
            channels = self.in_channels
            spatial = int(np.sqrt(x.shape[1] / channels))
            if spatial * spatial * channels != x.shape[1]:
                raise ValueError(f"Cannot infer spatial size from shape {x.shape}")
            x = reshape(x, (x.shape[0], channels, spatial, spatial))
        elif len(x.shape) == 4 and x.shape[-1] in (1, 3):
            x = transpose(x, axes=(0, 3, 1, 2))
        return x

    def forward(self, x):
        x = self._ensure_nchw(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        height, width = x.shape[2], x.shape[3]
        x = x.sum(axes=(2, 3)) / float(height * width)
        return self.classifier(x)
