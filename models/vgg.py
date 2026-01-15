import numpy as np
import torch,torchvision
from torchofdreaveler import nn
from torchofdreaveler._core.operators import Tensor, reshape, transpose


class VGG11(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, device=None):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, device=device),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Conv2d(64, 128, 3, padding=1, device=device),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Conv2d(128, 256, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, device=device),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Conv2d(256, 512, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, device=device),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Conv2d(512, 512, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, device=device),
            nn.ReLU(),
            nn.MaxPool2d(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes, device=device),
        )

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
        x = self.features(x)
        return self.classifier(x)
