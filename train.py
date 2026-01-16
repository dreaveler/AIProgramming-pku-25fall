import argparse
import torch,torchvision
from torchvision import transforms
from torchofdreaveler import data, optim
from torchofdreaveler._core.device import gpu

from models.resnet import ResNetSmall, ResNet20
from torchofdreaveler.training import fit


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNetSmall on CIFAR10.")
    parser.add_argument("--model", choices=["resnet", "resnet20", "vgg", "lenet"], default="resnet")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--scheduler", choices=["step", "cos", "none"], default="cos")
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--checkpoint-path", default="checkpoints/resnet_small.pkl")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--log-dir", default="logdir/resnet_small")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    device = gpu()
    train_transforms = [
        transforms.ToTensor(),
    ]
    if args.augment:
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + train_transforms
    if args.normalize:
        train_transforms.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        )
    test_transforms = [
        transforms.ToTensor(),
    ]
    if args.normalize:
        test_transforms.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        )
    train_ds = data.cifar10(train=True, flatten=False, augment=False,
                            transform=transforms.Compose(train_transforms))
    test_ds = data.cifar10(train=False, flatten=False, augment=False,
                           transform=transforms.Compose(test_transforms))
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

    if args.model == "resnet":
        model = ResNetSmall(in_channels=3, num_classes=10, device=device).to(device)
    elif args.model == "resnet20":
        model = ResNet20(in_channels=3, num_classes=10, device=device).to(device)
    elif args.model == "vgg":
        from models.vgg import VGG11
        model = VGG11(in_channels=3, num_classes=10, device=device).to(device)
    else:
        from models.lenet import LeNet
        model = LeNet(in_channels=3, num_classes=10, device=device, input_size=32).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
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
