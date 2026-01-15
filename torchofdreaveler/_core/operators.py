"""
本文件我们给出一个基本完善的Tensor类
你可以将hw5的对应代码复制到这里
"""

import numpy as np
from typing import List, Optional, Tuple, Union
import torch,torchvision
import torchofdreaveler as mytorch
from .device import cpu, gpu, Device, CPUDevice, CUDADevice
from .basic_operator import Op, Value
from .autodiff import compute_gradient_of_variables

class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        def _wrap_device(core_dev):
            return gpu() if core_dev == mytorch.Device.gpu else cpu()

        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        elif isinstance(array, mytorch.Tensor):
            cached_data = array
            if device is None:
                device = _wrap_device(array.device)
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )
        self._device = device

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        core_dev = mytorch.Device.gpu if isinstance(device, CUDADevice) else mytorch.Device.cpu
        return mytorch.Tensor(np.array(numpy_array, dtype="float32"), core_dev)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return "float32"

    @property
    def device(self):
        if hasattr(self, "_device"):
            return self._device
        cached = self.realize_cached_data()
        self._device = gpu() if cached.device == mytorch.Device.gpu else cpu()
        return self._device

    def to(self, device):
        if device is None:
            return self
        if isinstance(device, str):
            dev = device.lower()
            if dev in ("cpu", "cpu()", "host"):
                device = cpu()
            elif dev in ("cuda", "gpu", "cuda()", "gpu()"):
                device = gpu()
            else:
                raise ValueError(f"Unknown device string: {device}")
        if device == self.device:
            return self
        data = self.numpy()
        return Tensor(data, device=device, dtype=self.dtype, requires_grad=self.requires_grad)

    def backward(self, out_grad=None):
        def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
            device = cpu() if device is None else device
            array = device.ones(*shape, dtype=dtype)
            if c != 1.0:
                array = mytorch.mul_scalar(array, float(c))
            return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
        def ones(*shape, device=None, dtype="float32", requires_grad=False):
            return constant(
                *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
            )
        out_grad = (
                    out_grad
                    if out_grad
                    else ones(*self.shape, dtype=self.dtype, device=self.device)
                )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()

        return data.cpu().numpy


    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return EWisePow()(self, other)
        else:
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def matmul(self, other):
        return MatMul()(self, other)

    def sum(self, axes=None):
        return Summation(axes)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def __neg__(self):
        return Negate()(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class EWiseAdd(TensorOp):
    def compute(self, a: mytorch.Tensor, b: mytorch.Tensor):
        return mytorch.add(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: mytorch.Tensor):
        return mytorch.add_scalar(a, float(self.scalar))

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: mytorch.Tensor, b: mytorch.Tensor):
        return mytorch.multiply(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: mytorch.Tensor):
        return mytorch.mul_scalar(a, float(self.scalar))

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: mytorch.Tensor) -> mytorch.Tensor:
        ## 请于此填写你的代码
        return mytorch.power_scalar(a, float(self.scalar))
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        a = node.inputs[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: mytorch.Tensor, b: mytorch.Tensor) -> mytorch.Tensor:
        return mytorch.power(a, b)

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """逐点相除"""

    def compute(self, a, b):
        ## 请于此填写你的代码
        return mytorch.divide(a, b)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        a,b = node.inputs
        grad_a = out_grad/b
        grad_b = -out_grad*a/(b*b)
        return grad_a,grad_b
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ## 请于此填写你的代码
        return mytorch.div_scalar(a, float(self.scalar))
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad*(1.0/self.scalar)
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ## 请于此填写你的代码
        if self.axes is None:
            perm = list(range(len(a.shape)))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            return mytorch.transpose(a, tuple(perm))
        if len(self.axes) == 2:
            perm = list(range(len(a.shape)))
            i, j = self.axes
            perm[i], perm[j] = perm[j], perm[i]
            return mytorch.transpose(a, tuple(perm))
        return mytorch.transpose(a, tuple(self.axes))
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        if self.axes is None:
            perm = list(range(len(out_grad.shape)))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            return transpose(out_grad, axes=tuple(perm))

        if len(self.axes) == 2:
            perm = list(range(len(out_grad.shape)))
            i, j = self.axes
            perm[i], perm[j] = perm[j], perm[i]
            return transpose(out_grad, axes=tuple(perm))

        inv_axes = [0] * len(self.axes)
        for idx, ax in enumerate(self.axes):
            inv_axes[ax] = idx
        return transpose(out_grad, axes=tuple(inv_axes))
        


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ## 请于此填写你的代码
        return mytorch.reshape(a, list(self.shape))
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return reshape(out_grad, node.inputs[0].shape)
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ## 请于此填写你的代码
        return mytorch.broadcast_to(a, list(self.shape))
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        x = node.inputs[0]
        grad = out_grad

        while len(grad.shape) > len(x.shape):
            grad = summation(grad, axes=(0,))

        axes = []
        for i, (in_dim, g_dim) in enumerate(zip(x.shape, grad.shape)):
            if in_dim == 1 and g_dim != 1:
                axes.append(i)
        for i in reversed(axes):
            grad = summation(grad, axes=(i,))

        return reshape(grad, x.shape)
        
        
        


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ## 请于此填写你的代码
        return mytorch.summation(a, self.axes)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        x = node.inputs[0]
        axes = self.axes
        in_shape = x.shape

        if axes is None:
            axes = tuple(range(len(in_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        else:
            axes = tuple(axes)
        axes = tuple(ax if ax >= 0 else ax + len(in_shape) for ax in axes)

        expected_out_rank = len(in_shape) - len(axes)
        if expected_out_rank == 0:
            target_shape = []
        else:
            target_shape = list(out_grad.shape)
        for ax in sorted(axes):
            target_shape.insert(ax, 1)
        grad = reshape(out_grad, tuple(target_shape))
        return broadcast_to(grad, in_shape)
            


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ## 请于此填写你的代码
        return mytorch.matmul(a, b)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        a,b = node.inputs
        def reduce_to_shape(grad, shape):
            while len(grad.shape) > len(shape):
                grad = summation(grad, axes=(0,))
            for i, (gd, sd) in enumerate(zip(grad.shape, shape)):
                if sd == 1 and gd != 1:
                    grad = summation(grad, axes=(i,))
            return reshape(grad, shape)
        grad_a = matmul(out_grad,transpose(b,axes=(-1,-2)))
        grad_b = matmul(transpose(a,axes=(-1,-2)),out_grad)
        grad_a = reduce_to_shape(grad_a, a.shape)
        grad_b = reduce_to_shape(grad_b, b.shape)
        return grad_a,grad_b
        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ## 请于此填写你的代码
        return mytorch.negate(a)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return -out_grad
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ## 请于此填写你的代码
        return mytorch.log(a)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad/node.inputs[0]
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ## 请于此填写你的代码
        return mytorch.exp(a)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad*exp(node.inputs[0])
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ## 请于此填写你的代码
        self.fn = mytorch.Relu()
        return self.fn.forward(a)
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return Tensor.make_const(self.fn.backward(out_grad.realize_cached_data()))
        

def relu(a):
    return ReLU()(a)


class SoftmaxCrossEntropyLoss(TensorOp):
    def compute(self, logits: mytorch.Tensor, labels: mytorch.Tensor):
        n = logits.shape[0]
        self.soutput = mytorch.Tensor.zeros(list(logits.shape), logits.device)
        self.labels = labels
        mytorch.softmax(logits, self.soutput)
        per_loss = mytorch.Tensor.zeros([n], logits.device)
        mytorch.cross_entropy_loss(self.soutput, labels, per_loss)
        loss = mytorch.div_scalar(mytorch.summation(per_loss, None), float(n))
        return loss

    def gradient(self, out_grad, node):
        logits, labels = node.inputs
        grad_input = mytorch.Tensor.zeros(list(logits.shape), logits.realize_cached_data().device)
        mytorch.softmax_cross_entropy_backward(self.soutput, labels.realize_cached_data(), grad_input)
        n = logits.shape[0]
        grad_input = mytorch.div_scalar(grad_input, float(n))
        if out_grad is not None:
            og = out_grad
            if og.shape != logits.shape:
                og = broadcast_to(og, logits.shape)
            grad_input = mytorch.multiply(grad_input, og.realize_cached_data())
        return Tensor.make_const(grad_input), Tensor.make_const(mytorch.Tensor.zeros(list(labels.shape), labels.realize_cached_data().device))


def softmax_cross_entropy(logits, labels):
    return SoftmaxCrossEntropyLoss()(logits, labels)


class BatchNorm2d(TensorOp):
    def __init__(self, running_mean, running_var, training=True, momentum=0.1, eps=1e-5):
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def compute(self, x: mytorch.Tensor, gamma: mytorch.Tensor, beta: mytorch.Tensor):
        n, c, h, w = x.shape
        out = mytorch.Tensor.zeros([n, c, h, w], x.device)
        self.mean = mytorch.Tensor.zeros([c], x.device)
        self.var = mytorch.Tensor.zeros([c], x.device)
        mytorch.batchnorm_forward(
            x, gamma, beta,
            out, self.mean, self.var,
            self.running_mean, self.running_var,
            self.training, self.momentum, self.eps,
        )
        return out

    def gradient(self, out_grad, node):
        x, gamma, beta = node.inputs
        grad_input = mytorch.Tensor.zeros(list(x.shape), x.realize_cached_data().device)
        grad_gamma = mytorch.Tensor.zeros(list(gamma.shape), gamma.realize_cached_data().device)
        grad_beta = mytorch.Tensor.zeros(list(beta.shape), beta.realize_cached_data().device)
        mytorch.batchnorm_backward(
            out_grad.realize_cached_data(),
            x.realize_cached_data(),
            gamma.realize_cached_data(),
            self.mean,
            self.var,
            grad_input,
            grad_gamma,
            grad_beta,
            self.eps,
        )
        return (
            Tensor.make_const(grad_input),
            Tensor.make_const(grad_gamma),
            Tensor.make_const(grad_beta),
        )


def batchnorm2d(x, gamma, beta, running_mean, running_var, training=True, momentum=0.1, eps=1e-5):
    return BatchNorm2d(
        running_mean=running_mean,
        running_var=running_var,
        training=training,
        momentum=momentum,
        eps=eps,
    )(x, gamma, beta)

class Conv2d(TensorOp):
    def __init__(self, padding=1, stride=1):
        self.padding = padding
        self.stride = stride

    def compute(self, img: mytorch.Tensor, kernel: mytorch.Tensor):
        n, _, h, w = img.shape
        cout = kernel.shape[0]
        out = mytorch.Tensor.zeros([n, cout, h, w], kernel.device)
        mytorch.convolve(img, kernel, out, padding=self.padding, stride=self.stride)
        return out

    def gradient(self, out_grad, node):
        img, kernel = node.inputs
        grad_input = mytorch.Tensor.zeros(list(img.shape), img.realize_cached_data().device)
        grad_kernel = mytorch.Tensor.zeros(list(kernel.shape), kernel.realize_cached_data().device)
        mytorch.convolve_backward(
            img.realize_cached_data(),
            kernel.realize_cached_data(),
            out_grad.realize_cached_data(),
            grad_input,
            grad_kernel,
        )
        return Tensor.make_const(grad_input), Tensor.make_const(grad_kernel)


def conv2d(img, kernel, padding=1, stride=1):
    return Conv2d(padding=padding, stride=stride)(img, kernel)


class MaxPool2d(TensorOp):
    def compute(self, x: mytorch.Tensor):
        n, c, h, w = x.shape
        out_h = h // 2
        out_w = w // 2
        out = mytorch.Tensor.zeros([n, c, out_h, out_w], x.device)
        self.mask = mytorch.Tensor.zeros([n, c, out_h, out_w], x.device)
        mytorch.maxpooling(x, out, self.mask)
        return out

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        grad_input = mytorch.Tensor.zeros(list(x.shape), x.realize_cached_data().device)
        mytorch.maxpooling_backward(
            out_grad.realize_cached_data(),
            self.mask,
            grad_input,
        )
        return Tensor.make_const(grad_input)


def maxpool2d(x):
    return MaxPool2d()(x)

