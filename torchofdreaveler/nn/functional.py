from torchofdreaveler._core.operators import (
    relu,
    conv2d,
    maxpool2d,
    matmul,
    reshape,
    transpose,
    softmax_cross_entropy,
    batchnorm2d,
)


def linear(x, weight, bias=None):
    out = matmul(x, weight)
    if bias is not None:
        out = out + bias
    return out


def flatten(x, start_dim=1):
    shape = x.shape
    if start_dim < 0:
        start_dim += len(shape)
    flat = 1
    for d in shape[start_dim:]:
        flat *= d
    new_shape = list(shape[:start_dim]) + [flat]
    return reshape(x, tuple(new_shape))
