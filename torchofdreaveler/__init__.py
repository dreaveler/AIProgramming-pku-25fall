from .core import *

import importlib


def __getattr__(name):
    if name in ("nn", "optim", "data", "training"):
        return importlib.import_module(f"{__name__}.{name}")
    if name == "Tensor":
        from torchofdreaveler._core.operators import Tensor as _Tensor
        return _Tensor
    if name in ("no_grad", "set_grad_enabled", "is_grad_enabled"):
        from torchofdreaveler._core import no_grad, set_grad_enabled, is_grad_enabled
        return {"no_grad": no_grad, "set_grad_enabled": set_grad_enabled, "is_grad_enabled": is_grad_enabled}[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["nn", "optim", "data", "training", "Tensor", "no_grad", "set_grad_enabled", "is_grad_enabled"]
