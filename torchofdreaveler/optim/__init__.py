import torchofdreaveler as mytorch
from torchofdreaveler._core.device import CUDADevice
from torchofdreaveler.optim.lr_scheduler import LRScheduler, StepLR, CosineAnnealingLR


class Optimizer:
    def __init__(self, params, lr=0.1, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        self.state = {}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.realize_cached_data()
            if self.momentum:
                if self.weight_decay:
                    g = mytorch.add(g, mytorch.mul_scalar(p.cached_data, self.weight_decay))
                state = self.state.get(id(p))
                if state is None:
                    state = {
                        "v": mytorch.Tensor.zeros(list(p.shape), p.cached_data.device),
                    }
                    self.state[id(p)] = state
                state["v"] = mytorch.add(
                    mytorch.mul_scalar(state["v"], self.momentum),
                    g,
                )
                update = state["v"]
            else:
                if self.weight_decay:
                    g = mytorch.add(g, mytorch.mul_scalar(p.cached_data, self.weight_decay))
                update = g
            p.cached_data = mytorch.add(p.cached_data, mytorch.mul_scalar(update, -self.lr))


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.state = {}

    def _core_device(self, dev):
        return mytorch.Device.gpu if isinstance(dev, CUDADevice) else mytorch.Device.cpu

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            state = self.state.get(id(p))
            if state is None:
                core_dev = self._core_device(p.device)
                state = {
                    "m": mytorch.Tensor.zeros(list(p.shape), core_dev),
                    "v": mytorch.Tensor.zeros(list(p.shape), core_dev),
                    "t": 0,
                }
                self.state[id(p)] = state

            g = p.grad.realize_cached_data()
            if self.weight_decay:
                g = mytorch.add(g, mytorch.mul_scalar(p.cached_data, self.weight_decay))
            state["t"] += 1
            state["m"] = mytorch.add(
                mytorch.mul_scalar(state["m"], self.beta1),
                mytorch.mul_scalar(g, 1 - self.beta1),
            )
            g_sq = mytorch.power_scalar(g, 2.0)
            state["v"] = mytorch.add(
                mytorch.mul_scalar(state["v"], self.beta2),
                mytorch.mul_scalar(g_sq, 1 - self.beta2),
            )

            m_hat = mytorch.div_scalar(state["m"], 1 - self.beta1 ** state["t"])
            v_hat = mytorch.div_scalar(state["v"], 1 - self.beta2 ** state["t"])
            denom = mytorch.add_scalar(mytorch.power_scalar(v_hat, 0.5), self.eps)
            step = mytorch.divide(m_hat, denom)
            p.cached_data = mytorch.add(p.cached_data, mytorch.mul_scalar(step, -self.lr))


__all__ = ["Optimizer", "SGD", "Adam", "LRScheduler", "StepLR", "CosineAnnealingLR"]
