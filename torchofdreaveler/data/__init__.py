import numpy as np

import os

try:
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from torchvision import datasets as tv_datasets
    from torchvision import transforms as tv_transforms
    from PIL import Image
except ImportError:  # pragma: no cover
    torch = None
    TorchDataLoader = None
    tv_datasets = None
    tv_transforms = None
    Image = None

from torchofdreaveler._core.device import cpu
from torchofdreaveler._core.operators import Tensor


class TorchvisionDataset:
    def __init__(self, dataset, normalize=False, flatten=False, mean=None, std=None):
        self.dataset = dataset
        self.normalize = normalize
        self.flatten = flatten
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if hasattr(x, "numpy"):
            x = x.numpy()
        x = np.asarray(x, dtype=np.float32)
        if self.normalize:
            x = x / 255.0 if x.max() > 1.0 else x
        if self.mean is not None and self.std is not None:
            if x.ndim == 3:
                mean = self.mean.reshape(-1, 1, 1)
                std = self.std.reshape(-1, 1, 1)
            else:
                mean = self.mean
                std = self.std
            x = (x - mean) / std
        if self.flatten:
            x = x.reshape(-1)
        return x, int(y)


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, device=None,
                 drop_last=False, num_workers=0, pin_memory=False):
        if TorchDataLoader is None:
            raise RuntimeError("torch and torchvision are required for DataLoader")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = cpu() if device is None else device
        self._loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def __iter__(self):
        for xb, yb in self._loader:
            if hasattr(xb, "numpy"):
                xb = xb.numpy()
            if hasattr(yb, "numpy"):
                yb = yb.numpy()
            xb = np.asarray(xb, dtype=np.float32)
            yb = np.asarray(yb, dtype=np.int64)
            yield Tensor(xb, device=self.device, requires_grad=False), yb

    def __len__(self):
        return len(self._loader)


class TinyImageNetDataset:
    def __init__(self, root, split="train", transform=None):
        if Image is None:
            raise RuntimeError("Pillow is required for Tiny ImageNet")
        self.root = root
        self.split = split
        self.transform = transform
        self.wnids = self._read_wnids(root)
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.samples = self._build_samples()

    def _read_wnids(self, root):
        wnids_path = os.path.join(root, "wnids.txt")
        if os.path.exists(wnids_path):
            with open(wnids_path, "r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]
        train_root = os.path.join(root, "train")
        if os.path.isdir(train_root):
            return sorted([name for name in os.listdir(train_root)
                           if os.path.isdir(os.path.join(train_root, name))])
        return []

    def _build_samples(self):
        split = self.split.lower()
        if split == "train":
            return self._samples_from_split_dir(os.path.join(self.root, "train"))
        if split in ("val", "valid", "validation"):
            val_root = os.path.join(self.root, "val")
            val_images = os.path.join(val_root, "images")
            val_ann = os.path.join(val_root, "val_annotations.txt")
            if os.path.isdir(val_images) and os.path.exists(val_ann):
                return self._samples_from_val_annotations(val_images, val_ann)
            return self._samples_from_split_dir(val_root)
        raise ValueError(f"Unsupported split: {self.split}")

    def _samples_from_split_dir(self, split_root):
        samples = []
        for wnid in self.wnids:
            images_dir = os.path.join(split_root, wnid, "images")
            if not os.path.isdir(images_dir):
                images_dir = os.path.join(split_root, wnid)
            if not os.path.isdir(images_dir):
                continue
            for filename in sorted(os.listdir(images_dir)):
                if filename.lower().endswith((".jpeg", ".jpg", ".png")):
                    path = os.path.join(images_dir, filename)
                    samples.append((path, self.class_to_idx[wnid]))
        return samples

    def _samples_from_val_annotations(self, images_dir, annotations_path):
        samples = []
        with open(annotations_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename, wnid = parts[0], parts[1]
                if wnid not in self.class_to_idx:
                    continue
                path = os.path.join(images_dir, filename)
                if os.path.exists(path):
                    samples.append((path, self.class_to_idx[wnid]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CachedArrayDataset:
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = int(self.labels[idx])
        if self.transform is not None:
            if Image is not None and not isinstance(x, Image.Image):
                x = Image.fromarray(np.asarray(x))
            x = self.transform(x)
        return x, y


def _build_transform(image_size=None, augment=False):
    if tv_transforms is None:
        return None
    transforms = []
    if image_size is not None:
        size = (image_size, image_size) if isinstance(image_size, int) else image_size
    else:
        size = None
    if augment:
        if size is not None:
            transforms.append(tv_transforms.RandomCrop(size, padding=4))
        transforms.append(tv_transforms.RandomHorizontalFlip())
    if size is not None and not augment:
        transforms.append(tv_transforms.Resize(size))
    transforms.append(tv_transforms.ToTensor())
    return tv_transforms.Compose(transforms)


def mnist(root="./datasets/mnist", train=True, download=True, flatten=False, augment=False,
          transform=None, normalize=False):
    if tv_datasets is None:
        raise RuntimeError("torchvision is required for MNIST")
    if transform is None:
        transform = _build_transform(image_size=None, augment=augment)
    dataset = tv_datasets.MNIST(root=root, train=train, download=download, transform=transform)
    return TorchvisionDataset(dataset, normalize=normalize, flatten=flatten)


def cifar10(root="./datasets/cifar10", train=True, download=True, flatten=False, augment=False,
            transform=None, normalize=False):
    if tv_datasets is None:
        raise RuntimeError("torchvision is required for CIFAR10")
    if transform is None:
        transform = _build_transform(image_size=32, augment=augment)
    dataset = tv_datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
    return TorchvisionDataset(dataset, normalize=normalize, flatten=flatten)


def imagenet(root="./datasets/imagenet", split="train", flatten=False, image_size=None, augment=False,
             transform=None, normalize=False):
    if tv_datasets is None:
        raise RuntimeError("torchvision is required for ImageNet")
    if not hasattr(tv_datasets, "ImageNet"):
        raise RuntimeError("torchvision ImageNet dataset is not available")
    if transform is None:
        transform = _build_transform(image_size=image_size, augment=augment)
    dataset = tv_datasets.ImageNet(root=root, split=split, transform=transform)
    return TorchvisionDataset(dataset, normalize=normalize, flatten=flatten)

def tiny_imagenet(root="./datasets/tiny-imagenet-200", split="train", flatten=False, image_size=None,
                  augment=False, transform=None, normalize=False):
    if transform is None:
        transform = _build_transform(image_size=image_size, augment=augment)
    dataset = TinyImageNetDataset(root=root, split=split, transform=transform)
    return TorchvisionDataset(dataset, normalize=normalize, flatten=flatten)


def cached_tiny_imagenet(cache_root="./datasets/tiny-imagenet-200/cache", split="train",
                         transform=None, normalize=False, flatten=False):
    images_path = os.path.join(cache_root, f"tiny_imagenet_{split}_images.npy")
    labels_path = os.path.join(cache_root, f"tiny_imagenet_{split}_labels.npy")
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise RuntimeError(f"Cached files not found: {images_path} / {labels_path}")
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    dataset = CachedArrayDataset(images, labels, transform=transform)
    return TorchvisionDataset(dataset, normalize=normalize, flatten=flatten)


__all__ = [
    "TorchvisionDataset",
    "DataLoader",
    "mnist",
    "cifar10",
    "imagenet",
    "tiny_imagenet",
    "cached_tiny_imagenet",
]
