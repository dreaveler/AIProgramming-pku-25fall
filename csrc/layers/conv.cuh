#pragma once
#ifndef CONV_CUH
#define CONV_CUH

#include "../tensor.cuh"

namespace nn {
    void img2col(const Tensor& img, Tensor& col, const int padding = 1, const int stride = 1,
                 const int k_h = 3, const int k_w = 3);
    void convolve(const Tensor& img, const Tensor& kernel, Tensor& output, const int padding = 1,
                  const int stride = 1);
    void convolve_backward(const Tensor& input, const Tensor& kernel, const Tensor& grad_y,
                           Tensor& grad_input, Tensor& grad_kernel,
                           const int padding = 1, const int stride = 1);
    void col2img(const Tensor& gard_col, Tensor& grad_img, const int padding = 1,
                 const int stride = 1, const int k_h = 3, const int k_w = 3);
}

#endif
