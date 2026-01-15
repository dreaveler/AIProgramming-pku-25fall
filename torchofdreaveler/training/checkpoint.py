import os
import pickle

import numpy as np

import torchofdreaveler as mytorch
from torchofdreaveler._core.device import CUDADevice


def _core_device(dev):
    return mytorch.Device.gpu if isinstance(dev, CUDADevice) else mytorch.Device.cpu


def _model_state_dict(model):
    state = {}
    for name, param in model.named_parameters():
        state[name] = np.array(param.numpy(), copy=True)
    return state


def _tensor_to_numpy(tensor):
    if hasattr(tensor, "cpu") and callable(tensor.cpu):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        numpy_attr = tensor.numpy
        if callable(numpy_attr):
            return np.array(numpy_attr(), copy=True)
        return np.array(numpy_attr, copy=True)
    return np.array(tensor, copy=True)


def _load_model_state(model, state):
    for name, param in model.named_parameters():
        if name not in state:
            raise KeyError(f"Missing parameter in checkpoint: {name}")
        arr = np.asarray(state[name], dtype=np.float32)
        param.cached_data = mytorch.Tensor(arr, _core_device(param.device))


def _optimizer_state_dict(optimizer):
    if optimizer is None:
        return None
    state = {"type": optimizer.__class__.__name__, "lr": optimizer.lr}
    state["weight_decay"] = getattr(optimizer, "weight_decay", 0.0)
    if optimizer.__class__.__name__ == "SGD":
        state["momentum"] = getattr(optimizer, "momentum", 0.0)
        state["state"] = {}
        for p in optimizer.params:
            pid = id(p)
            entry = optimizer.state.get(pid)
            if entry and "v" in entry:
                state["state"][pid] = _tensor_to_numpy(entry["v"])
    elif optimizer.__class__.__name__ == "Adam":
        state["betas"] = (optimizer.beta1, optimizer.beta2)
        state["eps"] = optimizer.eps
        state["state"] = {}
        for p in optimizer.params:
            pid = id(p)
            entry = optimizer.state.get(pid)
            if entry is None:
                continue
            state["state"][pid] = {
                "m": _tensor_to_numpy(entry["m"]),
                "v": _tensor_to_numpy(entry["v"]),
                "t": entry["t"],
            }
    return state


def _load_optimizer_state(optimizer, state):
    if optimizer is None or state is None:
        return
    if state.get("type") != optimizer.__class__.__name__:
        raise ValueError("Optimizer type mismatch in checkpoint")
    optimizer.lr = state.get("lr", optimizer.lr)
    if hasattr(optimizer, "weight_decay"):
        optimizer.weight_decay = state.get("weight_decay", optimizer.weight_decay)
    if optimizer.__class__.__name__ == "SGD":
        optimizer.momentum = state.get("momentum", optimizer.momentum)
        optimizer.state = {}
        saved = state.get("state", {})
        for p in optimizer.params:
            pid = id(p)
            if pid not in saved:
                continue
            v = np.asarray(saved[pid], dtype=np.float32)
            optimizer.state[pid] = {"v": mytorch.Tensor(v, _core_device(p.device))}
    elif optimizer.__class__.__name__ == "Adam":
        optimizer.beta1, optimizer.beta2 = state.get("betas", (optimizer.beta1, optimizer.beta2))
        optimizer.eps = state.get("eps", optimizer.eps)
        optimizer.state = {}
        saved = state.get("state", {})
        for p in optimizer.params:
            pid = id(p)
            if pid not in saved:
                continue
            entry = saved[pid]
            optimizer.state[pid] = {
                "m": mytorch.Tensor(np.asarray(entry["m"], dtype=np.float32), _core_device(p.device)),
                "v": mytorch.Tensor(np.asarray(entry["v"], dtype=np.float32), _core_device(p.device)),
                "t": entry["t"],
            }


def save_checkpoint(path, model, optimizer=None, epoch=None, extra=None):
    payload = {
        "epoch": epoch,
        "model": _model_state_dict(model),
        "optimizer": _optimizer_state_dict(optimizer),
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def load_checkpoint(path, model, optimizer=None):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    _load_model_state(model, payload["model"])
    _load_optimizer_state(optimizer, payload.get("optimizer"))
    return payload.get("epoch", None), payload.get("extra", {})


__all__ = ["save_checkpoint", "load_checkpoint"]
