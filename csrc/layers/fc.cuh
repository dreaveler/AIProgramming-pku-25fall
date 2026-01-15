#pragma once
#ifndef FC_CUH
#define FC_CUH

#include "../tensor.cuh"

namespace nn {
    void gemm_gpu(const Tensor& A, const Tensor& B, Tensor& C,
                  const float alpha = 1.0f, const float beta = 0.0f,
                  const bool A_trans = false, const bool B_trans = false);
    void FC_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);
    void FC_forward_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
                        Tensor& output, int batch_size, int in_c, int out_c);
    void FC_backward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                     const Tensor& grad_output, Tensor& grad_input, Tensor& grad_weights, Tensor& grad_bias);
    void FC_backward_cpu(const Tensor& input, const Tensor& weight, const Tensor& grad_output,
                         Tensor& grad_input, Tensor& grad_weights, Tensor& grad_bias,
                         int batch_size, int in_c, int out_c);
}

#endif
