import argparse

from torchvision import transforms

from torchofdreaveler import data, optim
from torchofdreaveler._core.device import gpu
from torchofdreaveler.training import fit

from models.resnet import ResNet20


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet20 on Tiny ImageNet.")
    parser.add_argument("--data-root", default="datasets/tiny-imagenet-200")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=["step", "cos", "none"], default="cos")
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoint-path", default="checkpoints/resnet20_tinyimagenet.pkl")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--log-dir", default="logdir/resnet20_tinyimagenet")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    return parser.parse_args()


def build_transforms(image_size, augment, normalize):
    train_transforms = []
    if augment:
        train_transforms.extend(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif image_size is not None:
        train_transforms.append(transforms.Resize(image_size))
    train_transforms.append(transforms.ToTensor())
    if normalize:
        train_transforms.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    test_transforms = []
    if image_size is not None:
        test_transforms.append(transforms.Resize(image_size))
    test_transforms.append(transforms.ToTensor())
    if normalize:
        test_transforms.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


def main():
    args = parse_args()
    device = gpu()

    train_tf, test_tf = build_transforms(args.image_size, args.augment, args.normalize)
    train_ds = data.tiny_imagenet(
        root=args.data_root,
        split="train",
        flatten=False,
        image_size=args.image_size,
        augment=False,
        transform=train_tf,
    )
    test_ds = data.tiny_imagenet(
        root=args.data_root,
        split="val",
        flatten=False,
        image_size=args.image_size,
        augment=False,
        transform=test_tf,
    )
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    model = ResNet20(in_channels=3, num_classes=200, device=device).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "cos":
        scheduler = optim.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    elif args.scheduler == "step":
        scheduler = optim.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    fit(
        model,
        train_loader,
        test_loader,
        optimizer,
        epochs=args.epochs,
        scheduler=scheduler,
        checkpoint_path=args.checkpoint_path,
        save_every=args.save_every,
        resume_from=args.resume_from,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
