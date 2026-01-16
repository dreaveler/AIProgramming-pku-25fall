import torch

import torchofdreaveler as mytorch
from torchofdreaveler._core.device import gpu


def main():
    dev = gpu()
    print("torchofdreaveler device:", dev)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count:", torch.cuda.device_count())
        idx = torch.cuda.current_device()
        print("torch.cuda.current_device:", idx)
        print("torch.cuda.get_device_name:", torch.cuda.get_device_name(idx))


if __name__ == "__main__":
    main()
