"""
本文件我们尝试实现一个Optimizer类，用于优化一个简单的双层Linear Network
本次作业主要的内容将会在opti_epoch内对于一个epoch的参数进行优化
分为SGD_epoch和Adam_epoch两个函数，分别对应SGD和Adam两种优化器
其余函数为辅助函数，也请一并填写
和大作业的要求一致，我们不对数据处理和读取做任何要求
因此你可以引入任何的库来帮你进行数据处理和读取
理论上我们也不需要依赖hw5的内容，如果你需要的话，你可以将hw5对应代码copy到对应位置
"""
from .autodiff import *
from .operators import *
from .device import CUDADevice
import numpy as np
import torch,torchvision
import torchofdreaveler as mytorch

def opti_epoch(X, y, weights, forward_fn, loss_fn, lr=0.1, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    if using_adam:
        Adam_epoch(X, y, weights, forward_fn, loss_fn, lr=lr, batch=batch, beta1=beta1, beta2=beta2)
    else:
        SGD_epoch(X, y, weights, forward_fn, loss_fn, lr=lr, batch=batch)

def SGD_epoch(X, y, weights, forward_fn, loss_fn, lr=0.1, batch=100):
    """ 
    SGD优化一个List of Weights
    本函数应该inplace地修改Weights矩阵来进行优化
    用学习率简单更新Weights

    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)

    for start in range(0, num_samples, batch):
        batch_idx = indices[start:start + batch]
        dev = weights[0].device
        X_batch = Tensor(X[batch_idx].astype(np.float32), device=dev, requires_grad=False)
        y_batch = y[batch_idx]
        batch_size = len(y_batch)

        logits = forward_fn(X_batch, weights)
        loss = loss_fn(logits, y_batch)

        # 计算梯度并更新参数
        for w in weights:
            w.grad = None
        loss.backward()
        for w in weights:
            if w.grad is None:
                continue
            grad = w.grad.realize_cached_data()
            w.cached_data = mytorch.add(w.cached_data, mytorch.mul_scalar(grad, -lr))

def Adam_epoch(X, y, weights, forward_fn, loss_fn, lr=0.1, batch=100, beta1=0.9, beta2=0.999):
    r""" 
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum

    Returns:
        None
    """
    eps = 1e-8
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)

    def _core_device(dev):
        return mytorch.Device.gpu if isinstance(dev, CUDADevice) else mytorch.Device.cpu

    for w in weights:
        core_dev = _core_device(w.device)
        if not hasattr(w, "adam_m"):
            w.adam_m = mytorch.Tensor.zeros(list(w.shape), core_dev)
        if not hasattr(w, "adam_v"):
            w.adam_v = mytorch.Tensor.zeros(list(w.shape), core_dev)
        if not hasattr(w, "adam_t"):
            w.adam_t = 0

    for start in range(0, num_samples, batch):
        batch_idx = indices[start:start + batch]
        dev = weights[0].device
        X_batch = Tensor(X[batch_idx].astype(np.float32), device=dev, requires_grad=False)
        y_batch = y[batch_idx]
        batch_size = len(y_batch)

        logits = forward_fn(X_batch, weights)
        loss = loss_fn(logits, y_batch)

        for w in weights:
            w.grad = None
        loss.backward()

        for w in weights:
            if w.grad is None:
                continue
            g = w.grad.realize_cached_data()
            w.adam_t += 1
            w.adam_m = mytorch.add(
                mytorch.mul_scalar(w.adam_m, beta1),
                mytorch.mul_scalar(g, 1 - beta1),
            )
            g_sq = mytorch.power_scalar(g, 2.0)
            w.adam_v = mytorch.add(
                mytorch.mul_scalar(w.adam_v, beta2),
                mytorch.mul_scalar(g_sq, 1 - beta2),
            )

            m_hat = mytorch.div_scalar(w.adam_m, 1 - beta1 ** w.adam_t)
            v_hat = mytorch.div_scalar(w.adam_v, 1 - beta2 ** w.adam_t)

            denom = mytorch.add_scalar(mytorch.power_scalar(v_hat, 0.5), eps)
            step = mytorch.divide(m_hat, denom)
            w.cached_data = mytorch.add(w.cached_data, mytorch.mul_scalar(step, -lr))


