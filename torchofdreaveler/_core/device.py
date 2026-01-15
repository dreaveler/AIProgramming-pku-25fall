"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们模仿PyTorch定义了一个数据运行框架Device
提供基础的运算接口
"""
import numpy as np
import torchofdreaveler as mytorch

class Device:
    """基类"""


class CPUDevice(Device):
    """CPU Device"""

    def __repr__(self):
        return "cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        return mytorch.Tensor.zeros(list(shape), mytorch.Device.cpu)

    def ones(self, *shape, dtype="float32"):
        return mytorch.Tensor.ones(list(shape), mytorch.Device.cpu)

    def randn(self, *shape):
        data = np.random.randn(*shape).astype("float32")
        return mytorch.Tensor(data, mytorch.Device.cpu)

    def rand(self, *shape):
        t = mytorch.Tensor.zeros(list(shape), mytorch.Device.cpu)
        t.random()
        return t

    def one_hot(self, n, i, dtype="float32"):
        data = np.eye(n, dtype=dtype)[i]
        return mytorch.Tensor(data, mytorch.Device.cpu)

    def empty(self, shape, dtype="float32"):
        return mytorch.Tensor(list(shape), mytorch.Device.cpu)

    def full(self, shape, fill_value, dtype="float32"):
        t = mytorch.Tensor.zeros(list(shape), mytorch.Device.cpu)
        if fill_value != 0:
            t = mytorch.add_scalar(t, float(fill_value))
        return t
    

class CUDADevice(Device):
    def __repr__(self):
        return "gpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CUDADevice)

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        return mytorch.Tensor.zeros(list(shape), mytorch.Device.gpu)

    def ones(self, *shape, dtype="float32"):
        return mytorch.Tensor.ones(list(shape), mytorch.Device.gpu)

    def randn(self, *shape):
        data = np.random.randn(*shape).astype("float32")
        return mytorch.Tensor(data, mytorch.Device.gpu)

    def rand(self, *shape):
        t = mytorch.Tensor.zeros(list(shape), mytorch.Device.gpu)
        t.random()
        return t

    def one_hot(self, n, i, dtype="float32"):
        data = np.eye(n, dtype=dtype)[i]
        return mytorch.Tensor(data, mytorch.Device.gpu)

    def empty(self, shape, dtype="float32"):
        return mytorch.Tensor(list(shape), mytorch.Device.gpu)

    def full(self, shape, fill_value, dtype="float32"):
        t = mytorch.Tensor.zeros(list(shape), mytorch.Device.gpu)
        if fill_value != 0:
            t = mytorch.add_scalar(t, float(fill_value))
        return t


def cpu():
    return CPUDevice()

def gpu():
    return CUDADevice()


def default_device():
    return gpu()


def all_devices():
    return [cpu(), gpu()]

