#include "batchnorm.cuh"
#include "../common/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <cmath>

__global__ void kernel_mean_var(const float* input, float* mean, float* var, int N, int C, int H, int W) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int hw = H * W;
    int m = N * hw;
    float sum = 0.0f;
    float sq = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float* ptr = input + (n * C + c) * hw;
        for (int i = 0; i < hw; ++i) {
            float v = ptr[i];
            sum += v;
            sq += v * v;
        }
    }
    float mu = sum / m;
    mean[c] = mu;
    var[c] = sq / m - mu * mu;
}

__global__ void kernel_bn_forward(const float* input,
                                  const float* gamma,
                                  const float* beta,
                                  const float* mean,
                                  const float* var,
                                  float* output,
                                  int N, int C, int H, int W,
                                  float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int hw = H * W;
    int c = (idx / hw) % C;
    float inv_std = rsqrtf(var[c] + eps);
    float x_hat = (input[idx] - mean[c]) * inv_std;
    output[idx] = x_hat * gamma[c] + beta[c];
}

__global__ void kernel_update_running(float* running_mean,
                                      float* running_var,
                                      const float* batch_mean,
                                      const float* batch_var,
                                      int C,
                                      float momentum) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * batch_mean[c];
    running_var[c] = momentum * running_var[c] + (1.0f - momentum) * batch_var[c];
}

__global__ void kernel_bn_grad_stats(const float* input,
                                     const float* grad_out,
                                     const float* mean,
                                     const float* var,
                                     float* grad_gamma,
                                     float* grad_beta,
                                     int N, int C, int H, int W,
                                     float eps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int hw = H * W;
    int m = N * hw;
    float gg = 0.0f;
    float gb = 0.0f;
    float inv_std = rsqrtf(var[c] + eps);
    for (int n = 0; n < N; ++n) {
        const float* in_ptr = input + (n * C + c) * hw;
        const float* go_ptr = grad_out + (n * C + c) * hw;
        for (int i = 0; i < hw; ++i) {
            float x_hat = (in_ptr[i] - mean[c]) * inv_std;
            float g = go_ptr[i];
            gg += g * x_hat;
            gb += g;
        }
    }
    grad_gamma[c] = gg;
    grad_beta[c] = gb;
}

__global__ void kernel_bn_backward(const float* input,
                                   const float* grad_out,
                                   const float* mean,
                                   const float* var,
                                   const float* gamma,
                                   const float* grad_gamma,
                                   const float* grad_beta,
                                   float* grad_input,
                                   int N, int C, int H, int W,
                                   float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int hw = H * W;
    int c = (idx / hw) % C;
    int m = N * hw;
    float inv_std = rsqrtf(var[c] + eps);
    float x_hat = (input[idx] - mean[c]) * inv_std;
    float g = grad_out[idx];
    float term = (m * g - grad_beta[c] - x_hat * grad_gamma[c]);
    grad_input[idx] = (gamma[c] * inv_std / m) * term;
}

void nn::batchnorm_forward(const Tensor& input,
                           const Tensor& gamma,
                           const Tensor& beta,
                           Tensor& output,
                           Tensor& mean,
                           Tensor& var,
                           Tensor& running_mean,
                           Tensor& running_var,
                           bool training,
                           float momentum,
                           float eps) {
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int total = N * C * H * W;

    int blocks_c = CudaGetBlocks(C);
    if (training) {
        kernel_mean_var<<<blocks_c, kCudaThreadsNum>>>(input.get_ptr(), mean.get_ptr(), var.get_ptr(), N, C, H, W);
        kernel_update_running<<<blocks_c, kCudaThreadsNum>>>(
            running_mean.get_ptr(), running_var.get_ptr(),
            mean.get_ptr(), var.get_ptr(), C, momentum);
    } else {
        cudaMemcpy(mean.get_ptr(), running_mean.get_ptr(), sizeof(float) * C, cudaMemcpyDeviceToDevice);
        cudaMemcpy(var.get_ptr(), running_var.get_ptr(), sizeof(float) * C, cudaMemcpyDeviceToDevice);
    }

    int blocks = CudaGetBlocks(total);
    kernel_bn_forward<<<blocks, kCudaThreadsNum>>>(
        input.get_ptr(), gamma.get_ptr(), beta.get_ptr(),
        mean.get_ptr(), var.get_ptr(),
        output.get_ptr(), N, C, H, W, eps);
}

void nn::batchnorm_backward(const Tensor& grad_out,
                            const Tensor& input,
                            const Tensor& gamma,
                            const Tensor& mean,
                            const Tensor& var,
                            Tensor& grad_input,
                            Tensor& grad_gamma,
                            Tensor& grad_beta,
                            float eps) {
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int total = N * C * H * W;

    int blocks_c = CudaGetBlocks(C);
    kernel_bn_grad_stats<<<blocks_c, kCudaThreadsNum>>>(
        input.get_ptr(), grad_out.get_ptr(),
        mean.get_ptr(), var.get_ptr(),
        grad_gamma.get_ptr(), grad_beta.get_ptr(),
        N, C, H, W, eps);

    int blocks = CudaGetBlocks(total);
    kernel_bn_backward<<<blocks, kCudaThreadsNum>>>(
        input.get_ptr(), grad_out.get_ptr(),
        mean.get_ptr(), var.get_ptr(),
        gamma.get_ptr(), grad_gamma.get_ptr(), grad_beta.get_ptr(),
        grad_input.get_ptr(), N, C, H, W, eps);
}
