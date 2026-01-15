#include "softmax_loss.cuh"
#include "../common/cuda_utils.cuh"
#include <cuda_runtime.h>

__global__ void kernel_softmax(const float* input, float* output, int N, int C) {
    CUDAKERNELLOOP(i, N) {
        const float* input_ptr = input + i * C;
        float* output_ptr = output + i * C;
        float max_val = input_ptr[0];
        for (int c = 1; c < C; c++) {
            float v = input_ptr[c];
            if (v > max_val) {
                max_val = v;
            }
        }
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += expf(input_ptr[c] - max_val);
        }
        for (int c = 0; c < C; c++) {
            output_ptr[c] = expf(input_ptr[c] - max_val) / sum;
        }
    }
}

void nn::softmax(const Tensor& input, Tensor& output) {
    int N = input.shape[0];
    int C = input.shape[1];
    kernel_softmax<<<CudaGetBlocks(N), kCudaThreadsNum>>>(input.get_ptr(), output.get_ptr(), N, C);
}

__global__ void kernel_crosseloss(const float* input, const float* label, float* output, int N, int C) {
    CUDAKERNELLOOP(i, N) {
        const float* input_ptr = input + i * C;
        int label_i = static_cast<int>(label[i]);
        output[i] = -log(input_ptr[label_i]);
    }
}

void nn::crossentropyloss(const Tensor& input, const Tensor& label, Tensor& output) {
    int N = input.shape[0];
    int C = input.shape[1];
    kernel_crosseloss<<<CudaGetBlocks(N), kCudaThreadsNum>>>(
        input.get_ptr(), label.get_ptr(), output.get_ptr(), N, C);
}

__global__ void kernel_smel_backward(const float* soutput, const float* label, float* grad_input, int N, int C) {
    CUDAKERNELLOOP(i, N * C) {
        int n = i / C;
        int c = i % C;

        int correct_class = static_cast<int>(label[n]);

        if (c == correct_class) {
            grad_input[i] = soutput[i] - 1.0f;
        } else {
            grad_input[i] = soutput[i];
        }
    }
}

void nn::softmaxsel_backward(const Tensor& soutput, const Tensor& label, Tensor& grad_sinput) {
    int N = soutput.shape[0];
    int C = soutput.shape[1];
    kernel_smel_backward<<<CudaGetBlocks(N * C), kCudaThreadsNum>>>(
        soutput.get_ptr(), label.get_ptr(), grad_sinput.get_ptr(), N, C);
}
