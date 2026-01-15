import math


class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch

    def get_last_lr(self):
        return [self.optimizer.lr]

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super().__init__(optimizer, last_epoch=last_epoch)
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr

    def get_lr(self):
        if self.step_size <= 0:
            return self.optimizer.lr
        steps = (self.last_epoch + 1) // self.step_size
        return self.base_lr * (self.gamma ** steps)


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        super().__init__(optimizer, last_epoch=last_epoch)
        self.T_max = max(1, T_max)
        self.eta_min = eta_min
        self.base_lr = optimizer.lr

    def get_lr(self):
        cosine = 0.5 * (1.0 + math.cos(math.pi * (self.last_epoch + 1) / self.T_max))
        return self.eta_min + (self.base_lr - self.eta_min) * cosine


__all__ = ["LRScheduler", "StepLR", "CosineAnnealingLR"]
