#include "conv.cuh"
#include "fc.cuh"
#include "../common/cuda_utils.cuh"
#include "../utils.cuh"
#include <cuda_runtime.h>

__global__ void kernel_img2col(const float* img, float* col,
                               int N, int C, int H, int W,
                               int k_h = 3, int k_w = 3, int stride = 1, int padding = 1) {
    int H_out = H;
    int W_out = W;
    CUDAKERNELLOOP(i, (long long)N * H_out * W_out * C * k_h * k_w) {
        int kw = i % k_w;
        int kh = (i / k_w) % k_h;
        int c_in = (i / (k_w * k_h)) % C;
        long long patch_idx = i / (C * k_h * k_w);
        int w_out = patch_idx % W_out;
        int h_out = (patch_idx / W_out) % H_out;
        int n = patch_idx / (H_out * W_out);

        int h_in = h_out * stride - padding + kh;
        int w_in = w_out * stride - padding + kw;

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            long long img_idx = ((long long)(n * C + c_in) * H + h_in) * W + w_in;
            col[i] = img[img_idx];
        } else {
            col[i] = 0.0f;
        }
    }
}

void nn::img2col(const Tensor& img, Tensor& col, const int padding, const int stride,
                  const int k_h, const int k_w) {
    int N = img.shape[0], C = img.shape[1], H = img.shape[2], W = img.shape[3];
    long long total_num = (long long)N * C * H * W * k_h * k_w;
    kernel_img2col<<<CudaGetBlocks((int)total_num), kCudaThreadsNum>>>(
        img.get_ptr(), col.get_ptr(), N, C, H, W, k_h, k_w, stride, padding);
}

void nn::convolve(const Tensor& img, const Tensor& kernel, Tensor& output,
                   const int padding, const int stride) {
    int Cout = kernel.shape[0], Cin = kernel.shape[1], k_h = kernel.shape[2], k_w = kernel.shape[3];
    int N = img.shape[0], C = img.shape[1], H = img.shape[2], W = img.shape[3];
    Tensor col({N * H * W, C * k_h * k_w}, Device::gpu);
    Tensor flattened_kernel({Cin * k_h * k_w, Cout}, Device::gpu);
    flatten_kernel(kernel, flattened_kernel);
    img2col(img, col, padding, stride, k_h, k_w);
    Tensor ans_N({N * H * W, Cout}, Device::gpu);
    gemm_gpu(col, flattened_kernel, ans_N);
    reshape_col_to_image(ans_N, output);
}

void nn::convolve_backward(const Tensor& img, const Tensor& kernel, const Tensor& grad_y,
                            Tensor& grad_img, Tensor& grad_kernel,
                            const int padding, const int stride) {
    int Cout = kernel.shape[0], Cin = kernel.shape[1], k_h = kernel.shape[2], k_w = kernel.shape[3];
    int N = img.shape[0], C = img.shape[1], H = img.shape[2], W = img.shape[3];
    Tensor col = Tensor::zeros({N * H * W, C * k_h * k_w}, Device::gpu);
    img2col(img, col, padding, stride, k_h, k_w);

    Tensor grad_col = Tensor::zeros({N * H * W, C * k_h * k_w}, Device::gpu);
    grad_img = Tensor::zeros(grad_img.shape, grad_img.device);

    Tensor grad_y_reshaped = Tensor({N * H * W, Cout}, Device::gpu);
    pack_NCHW_rows(grad_y, grad_y_reshaped);

    Tensor grad_kernel_reshaped = Tensor({Cin * k_h * k_w, Cout}, Device::gpu);
    gemm_gpu(col, grad_y_reshaped, grad_kernel_reshaped, 1.0f, 0.0f, true, false);

    unflatten_KO_to_OIHW(grad_kernel_reshaped, grad_kernel);

    Tensor flattened_kernel = Tensor::zeros({Cin * k_h * k_w, Cout}, Device::gpu);
    flatten_kernel(kernel, flattened_kernel);
    gemm_gpu(grad_y_reshaped, flattened_kernel, grad_col, 1.0f, 0.0f, false, true);

    col2img(grad_col, grad_img, padding, stride, k_h, k_w);
}

__global__ void col2img_kernel(const float* grad_col, float* grad_img,
                               const int N, const int C, const int H, const int W,
                               const int stride = 1, const int padding = 1,
                               const int k_h = 3, const int k_w = 3) {
    int OH = H, OW = W;
    CUDAKERNELLOOP(i, (long long)N * OH * OW * C * k_h * k_w) {
        int kw = i % k_w;
        int kh = (i / k_w) % k_h;
        int c_in = (i / (k_w * k_h)) % C;
        long long patch_idx = i / (C * k_h * k_w);
        int w_out = patch_idx % OW;
        int h_out = (patch_idx / OW) % OH;
        int n = patch_idx / (OH * OW);

        int h_in = h_out * stride - padding + kh;
        int w_in = w_out * stride - padding + kw;

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            long long img_idx = ((long long)(n * C + c_in) * H + h_in) * W + w_in;
            atomicAdd(&grad_img[img_idx], grad_col[i]);
        }
    }
}

void nn::col2img(const Tensor& grad_col, Tensor& grad_img, const int padding,
                  const int stride, const int k_h, const int k_w) {
    int N = grad_img.shape[0], C = grad_img.shape[1], H = grad_img.shape[2], W = grad_img.shape[3];
    long long total_num = (long long)N * C * H * W * k_h * k_w;
    col2img_kernel<<<CudaGetBlocks((int)total_num), kCudaThreadsNum>>>(
        grad_col.get_ptr(), grad_img.get_ptr(), N, C, H, W, stride, padding, k_h, k_w);
}
