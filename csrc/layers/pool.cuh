#pragma once
#ifndef POOL_CUH
#define POOL_CUH

#include "../tensor.cuh"

namespace nn {
    void maxpooling(const Tensor& input, Tensor& output, Tensor& mask);
    void maxpooling_backward(const Tensor& grad_y, const Tensor& mask, Tensor& grad_x);
}

#endif
