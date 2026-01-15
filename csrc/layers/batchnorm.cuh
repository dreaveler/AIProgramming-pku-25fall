#pragma once
#ifndef BATCHNORM_CUH
#define BATCHNORM_CUH

#include "../tensor.cuh"

namespace nn {
    void batchnorm_forward(const Tensor& input,
                           const Tensor& gamma,
                           const Tensor& beta,
                           Tensor& output,
                           Tensor& mean,
                           Tensor& var,
                           Tensor& running_mean,
                           Tensor& running_var,
                           bool training = true,
                           float momentum = 0.1f,
                           float eps = 1e-5f);

    void batchnorm_backward(const Tensor& grad_out,
                            const Tensor& input,
                            const Tensor& gamma,
                            const Tensor& mean,
                            const Tensor& var,
                            Tensor& grad_input,
                            Tensor& grad_gamma,
                            Tensor& grad_beta,
                            float eps = 1e-5f);
}

#endif
