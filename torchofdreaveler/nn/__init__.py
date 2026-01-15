import numpy as np

from torchofdreaveler._core.device import cpu
from torchofdreaveler._core.operators import Tensor, matmul, relu, conv2d, maxpool2d, reshape, softmax_cross_entropy, batchnorm2d, broadcast_to
from . import functional


class Module:
    def __init__(self):
        self.training = True
        self._modules = {}
        self._parameters = {}

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            modules = self.__dict__.setdefault("_modules", {})
            modules[name] = value
        elif isinstance(value, Tensor):
            params = self.__dict__.setdefault("_parameters", {})
            params[name] = value
        object.__setattr__(self, name, value)

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def parameters(self):
        params = []
        seen = set()
        self._collect_parameters(params, seen)
        return params

    def named_parameters(self, prefix=""):
        params = []
        self._collect_named_parameters(params, prefix)
        return params

    def _collect_parameters(self, params, seen):
        for p in self._parameters.values():
            if id(p) in seen:
                continue
            seen.add(id(p))
            params.append(p)
        for m in self._modules.values():
            m._collect_parameters(params, seen)

    def _collect_named_parameters(self, params, prefix):
        for name, p in self._parameters.items():
            params.append((f"{prefix}{name}", p))
        for name, m in self._modules.items():
            m._collect_named_parameters(params, f"{prefix}{name}.")

    def to(self, device):
        for name, p in list(self._parameters.items()):
            new_p = p.to(device)
            setattr(self, name, new_p)
        for m in self._modules.values():
            m.to(device)
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        device = cpu() if device is None else device
        std = 1.0 / np.sqrt(in_features)
        weight = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.weight = Tensor(weight, device=device, requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), device=device, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        out = matmul(x, self.weight)
        if self.bias is not None:
            bias = reshape(self.bias, (1, self.bias.shape[0]))
            out = out + broadcast_to(bias, out.shape)
        return out


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, device=None):
        super().__init__()
        device = cpu() if device is None else device
        if isinstance(kernel_size, int):
            k_h, k_w = kernel_size, kernel_size
        else:
            k_h, k_w = kernel_size
        weight = np.random.randn(out_channels, in_channels, k_h, k_w).astype(np.float32) * 0.05
        self.weight = Tensor(weight, device=device, requires_grad=True)
        self.bias = None
        if bias:
            b = np.zeros(out_channels, dtype=np.float32)
            self.bias = Tensor(b, device=device, requires_grad=True)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        out = conv2d(x, self.weight, padding=self.padding, stride=self.stride)
        if self.bias is not None:
            bias = reshape(self.bias, (1, self.bias.shape[0], 1, 1))
            out = out + broadcast_to(bias, out.shape)
        return out


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class MaxPool2d(Module):
    def forward(self, x):
        return maxpool2d(x)


class Flatten(Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        shape = x.shape
        start_dim = self.start_dim
        if start_dim < 0:
            start_dim += len(shape)
        flat = 1
        for d in shape[start_dim:]:
            flat *= d
        new_shape = list(shape[:start_dim]) + [flat]
        return reshape(x, tuple(new_shape))


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for idx, m in enumerate(modules):
            setattr(self, str(idx), m)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class CrossEntropyLoss(Module):
    def forward(self, logits, targets):
        if not isinstance(targets, Tensor):
            targets = Tensor(np.asarray(targets, dtype=np.float32), device=logits.device, requires_grad=False)
        return softmax_cross_entropy(logits, targets)


class BatchNorm2d(Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5, device=None):
        super().__init__()
        device = cpu() if device is None else device
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), device=device, requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), device=device, requires_grad=True)
        self.running_mean = Tensor(np.zeros(num_features, dtype=np.float32), device=device, requires_grad=False)
        self.running_var = Tensor(np.ones(num_features, dtype=np.float32), device=device, requires_grad=False)
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        return batchnorm2d(
            x,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )


__all__ = [
    "Module",
    "Linear",
    "Conv2d",
    "ReLU",
    "MaxPool2d",
    "Flatten",
    "Sequential",
    "CrossEntropyLoss",
    "BatchNorm2d",
    "functional",
]
