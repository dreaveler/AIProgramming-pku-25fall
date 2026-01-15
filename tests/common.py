import os

import numpy as np
import torch,torchvision
import torchofdreaveler as mytorch

try:
    from torchvision import datasets
except ImportError:  # pragma: no cover
    datasets = None


MNIST_ROOT = os.path.join(os.path.dirname(__file__), "..", "datasets", "mnist")
DEFAULT_DEVICE = mytorch.Device.gpu


def numpy_to_tensor(array: np.ndarray, device=DEFAULT_DEVICE):
    """Convert numpy array to mytorch Tensor on the requested device."""
    contiguous = np.ascontiguousarray(array.astype(np.float32))
    return mytorch.Tensor(contiguous, device=device)


def tensor_to_numpy(t: mytorch.Tensor) -> np.ndarray:
    """Return a standalone numpy copy of a (possibly GPU) Tensor."""
    return np.array(t.cpu().numpy, copy=True)


def random_numpy(shape):
    return np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def load_mnist_numpy(train=True, limit=1024):
    """Download (if needed) MNIST and return normalized numpy arrays."""
    if datasets is None:
        raise RuntimeError("torchvision is required to load MNIST.")
    ds = datasets.MNIST(root=MNIST_ROOT, train=train, download=True)
    limit = len(ds.data) if limit is None else min(limit, len(ds.data))
    images = ds.data[:limit].numpy().astype(np.float32) / 255.0
    images = np.expand_dims(images, 1)  # (N, 1, 28, 28)
    labels = ds.targets[:limit].numpy().astype(np.int64)
    return images, labels


def mnist_tensors(limit=128, device=DEFAULT_DEVICE):
    images, labels = load_mnist_numpy(train=True, limit=limit)
    image_tensor = numpy_to_tensor(images, device=device)
    label_tensor = numpy_to_tensor(labels.astype(np.float32), device=device)
    return image_tensor, label_tensor
