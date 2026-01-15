#pragma once
#ifndef SOFTMAX_LOSS_CUH
#define SOFTMAX_LOSS_CUH

#include "../tensor.cuh"

namespace nn {
    void softmax(const Tensor& input, Tensor& output);
    void crossentropyloss(const Tensor& input, const Tensor& label, Tensor& output);
    void softmaxsel_backward(const Tensor& soutput, const Tensor& label, Tensor& grad_sinput);
}

#endif
