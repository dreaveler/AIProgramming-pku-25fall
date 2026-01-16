import argparse
import os

import numpy as np
from PIL import Image
import torch,torchvision
from torchofdreaveler.data import TinyImageNetDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Cache Tiny ImageNet images/labels to .npy files.")
    parser.add_argument("--data-root", default="datasets/tiny-imagenet-200")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output-dir", default="datasets/tiny-imagenet-200/cache")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = TinyImageNetDataset(root=args.data_root, split=args.split, transform=None)
    num_samples = len(dataset)
    if num_samples == 0:
        raise RuntimeError(f"No samples found for split={args.split} in {args.data_root}")

    images_path = os.path.join(args.output_dir, f"tiny_imagenet_{args.split}_images.npy")
    labels_path = os.path.join(args.output_dir, f"tiny_imagenet_{args.split}_labels.npy")

    images = np.lib.format.open_memmap(
        images_path, mode="w+", dtype=np.uint8, shape=(num_samples, args.image_size, args.image_size, 3)
    )
    labels = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int64, shape=(num_samples,)
    )

    for idx in range(num_samples):
        img, label = dataset[idx]
        if args.image_size is not None:
            img = img.resize((args.image_size, args.image_size), Image.BILINEAR)
        images[idx] = np.asarray(img, dtype=np.uint8)
        labels[idx] = int(label)
        if args.log_every and (idx + 1) % args.log_every == 0:
            print(f"cached {idx + 1}/{num_samples} images...")

    images.flush()
    labels.flush()
    print(f"Done. Saved {images_path} and {labels_path}")


if __name__ == "__main__":
    main()
