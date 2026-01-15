#include "fc.cuh"
#include "../common/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

void nn::gemm_gpu(const Tensor& A, const Tensor& B, Tensor& C,
                   const float alf, const float bet,
                   const bool A_trans, const bool B_trans) {
    const int A_rows = A.shape[0];
    const int A_cols = A.shape[1];
    const int B_rows = B.shape[0];
    const int B_cols = B.shape[1];
    const int C_rows = C.shape[0];
    const int C_cols = C.shape[1];

    const int m = C_rows;
    const int n = C_cols;
    const int k = A_trans ? A_rows : A_cols;

    const int lda = A_cols;
    const int ldb = B_cols;
    const int ldc = C_cols;

    const cublasOperation_t transA = A_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transB = B_trans ? CUBLAS_OP_T : CUBLAS_OP_N;

    const float* alpha = &alf;
    const float* beta = &bet;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, transB, transA,
                n, m, k,
                alpha,
                B.get_ptr(), ldb,
                A.get_ptr(), lda,
                beta,
                C.get_ptr(), ldc);
    cublasDestroy(handle);
}

void nn::FC_forward(const Tensor& input, const Tensor& weight, const Tensor& bias,
                     Tensor& output) {
    int batch_size = input.shape[0];
    gemm_gpu(input, weight, output);
    gemm_gpu(Tensor::ones({batch_size, 1}, Device::gpu), bias, output, 1.0f, 1.0f);
}

void nn::FC_forward_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
                         Tensor& output, int batch_size, int in_c, int out_c) {
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_c; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < in_c; ++k) {
                sum += input.get_ptr()[i * in_c + k] * weight.get_ptr()[k * out_c + j];
            }
            output.get_ptr()[i * out_c + j] = sum;
        }
    }

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_c; ++j) {
            output.get_ptr()[i * out_c + j] += bias.get_ptr()[j];
        }
    }
}

void nn::FC_backward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                      const Tensor& grad_output, Tensor& grad_input, Tensor& grad_weights, Tensor& grad_bias) {
    int batch_size = input.shape[0];
    gemm_gpu(grad_output, weight, grad_input, 1.0f, 0.0f, false, true);
    gemm_gpu(input, grad_output, grad_weights, 1.0f, 0.0f, true, false);
    gemm_gpu(Tensor::ones({batch_size, 1}, Device::gpu), grad_output, grad_bias, 1.0f, 0.0f, true, false);
}

void nn::FC_backward_cpu(const Tensor& input,
                          const Tensor& weight,
                          const Tensor& grad_output,
                          Tensor& grad_input,
                          Tensor& grad_weights,
                          Tensor& grad_bias,
                          int batch_size,
                          int in_c,
                          int out_c) {
    std::fill(grad_input.get_ptr(), grad_input.get_ptr() + batch_size * in_c, 0.f);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < in_c; ++i) {
            float sum = 0.f;
            for (int o = 0; o < out_c; ++o) {
                sum += grad_output.get_ptr()[b * out_c + o] * weight.get_ptr()[i * out_c + o];
            }
            grad_input.get_ptr()[b * in_c + i] = sum;
        }
    }

    std::fill(grad_weights.get_ptr(), grad_weights.get_ptr() + in_c * out_c, 0.f);
    for (int i = 0; i < in_c; ++i) {
        for (int o = 0; o < out_c; ++o) {
            float sum = 0.f;
            for (int b = 0; b < batch_size; ++b) {
                sum += input.get_ptr()[b * in_c + i] * grad_output.get_ptr()[b * out_c + o];
            }
            grad_weights.get_ptr()[i * out_c + o] = sum;
        }
    }

    std::fill(grad_bias.get_ptr(), grad_bias.get_ptr() + out_c, 0.f);
    for (int o = 0; o < out_c; ++o) {
        float sum = 0.f;
        for (int b = 0; b < batch_size; ++b) {
            sum += grad_output.get_ptr()[b * out_c + o];
        }
        grad_bias.get_ptr()[o] = sum;
    }
}
