#include "pool.cuh"
#include "../common/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <cfloat>

__global__ void kernel_maxpooling(const float* input, float* output, int* mask,
                                  const int N, const int C, const int H, const int W,
                                  const int k_h = 2, const int k_w = 2,
                                  const int stride = 2, const int padding = 0) {
    int OH = H / 2, OW = W / 2;
    CUDAKERNELLOOP(i, N * C * OH * OW) {
        int n = i / (C * OH * OW);
        int c = (i % (C * OH * OW)) / (OH * OW);
        int ph = (i % (OH * OW)) / OW;
        int pw = i % OW;
        const int oh = ph * stride;
        const int ow = pw * stride;
        float max_val = -FLT_MAX;
        int max_idx = -1;
        for (int j = oh; j < oh + 2; j++) {
            for (int k = ow; k < ow + 2; k++) {
                const int input_idx = ((n * C + c) * H + j) * W + k;
                float val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }
        output[i] = max_val;
        mask[i] = max_idx;
    }
}

void nn::maxpooling(const Tensor& input, Tensor& output, Tensor& mask) {
    int N = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
    int OH = H / 2, OW = W / 2;
    int total_num = N * C * OH * OW;
    kernel_maxpooling<<<CudaGetBlocks(total_num), kCudaThreadsNum>>>(
        input.get_ptr(), output.get_ptr(), reinterpret_cast<int*>(mask.get_ptr()), N, C, H, W);
}

__global__ void kernel_maxpooling_backward(const float* grad_y, const int* mask, float* grad_input,
                                           const int N, const int C, const int H, const int W,
                                           const int OH, const int OW,
                                           const int k_h = 2, const int k_w = 2,
                                           const int stride = 2, const int padding = 0) {
    const int output_size = N * C * OH * OW;
    CUDAKERNELLOOP(index, output_size) {
        const int max_index = mask[index];
        const float grad = grad_y[index];
        if (max_index >= 0) {
            atomicAdd(&grad_input[max_index], grad);
        }
    }
}

void nn::maxpooling_backward(const Tensor& grad_y, const Tensor& mask, Tensor& grad_input) {
    int N = grad_input.shape[0], C = grad_input.shape[1], H = grad_input.shape[2], W = grad_input.shape[3];
    int OH = grad_y.shape[2], OW = grad_y.shape[3];
    int total_num = N * C * OH * OW;
    grad_input = Tensor::zeros(grad_input.shape, Device::gpu);
    kernel_maxpooling_backward<<<CudaGetBlocks(total_num), kCudaThreadsNum>>>(
        grad_y.get_ptr(), reinterpret_cast<const int*>(mask.get_ptr()),
        grad_input.get_ptr(), N, C, H, W, OH, OW);
}
