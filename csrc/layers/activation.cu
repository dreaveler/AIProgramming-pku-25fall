#include "activation.cuh"
#include "../common/cuda_utils.cuh"
#include <cuda_runtime.h>

__global__ void gpu_relu(const float* in, float* out, float* mask, int N) {
    CUDAKERNELLOOP(i, N) {
        float v = in[i];
        out[i] = v > 0 ? v : 0.0f;
        mask[i] = v > 0 ? 1.0f : 0.0f;
    }
}
__global__ void gpu_relu_backward(const float* grad_y, const float* mask, float* grad_x, int N) {
    CUDAKERNELLOOP(i, N) {
        grad_x[i] = grad_y[i] * mask[i];
    }
}

__global__ void gpu_sigmoid(const float* in, float* out, float* saved, int N) {
    CUDAKERNELLOOP(i, N) {
        float v = in[i];
        float s = 1.0f / (1.0f + expf(-v));
        out[i] = s;
        saved[i] = s;
    }
}
__global__ void gpu_sigmoid_backward(const float* grad_y, const float* y_saved, float* grad_x, int N) {
    CUDAKERNELLOOP(i, N) {
        float s = y_saved[i];
        grad_x[i] = grad_y[i] * s * (1.0f - s);
    }
}

Tensor cpu_relu(const Tensor& x, Tensor& saved) {
    Tensor out(x.shape, x.device);
    for (auto i = 0; i < x.N; i++) {
        out.get_ptr()[i] = x.get_ptr()[i] > 0 ? x.get_ptr()[i] : 0.f;
        saved.get_ptr()[i] = x.get_ptr()[i] > 0 ? 1.f : 0.f;
    }
    return out;
}

Tensor cpu_relu_backward(const Tensor& grad_y, const Tensor& saved) {
    Tensor grad_x(grad_y.shape, grad_y.device);
    for (auto i = 0; i < grad_y.N; i++) {
        grad_x.get_ptr()[i] = grad_y.get_ptr()[i] * saved.get_ptr()[i];
    }
    return grad_x;
}

Tensor cpu_sigmoid(const Tensor& x, Tensor& saved) {
    Tensor out(x.shape, x.device);
    for (auto i = 0; i < x.N; i++) {
        out.get_ptr()[i] = 1.0f / (1.0f + expf(-x.get_ptr()[i]));
        saved.get_ptr()[i] = out.get_ptr()[i];
    }
    return out;
}

Tensor cpu_sigmoid_backward(const Tensor& grad_y, const Tensor& saved) {
    Tensor grad_x(grad_y.shape, grad_y.device);
    for (auto i = 0; i < grad_y.N; i++) {
        grad_x.get_ptr()[i] = grad_y.get_ptr()[i] * saved.get_ptr()[i] * (1.0f - saved.get_ptr()[i]);
    }
    return grad_x;
}

Tensor Relu::forward(const Tensor& x) {
    if (x.device == Device::cpu) {
        saved = Tensor(x.shape, x.device);
        return cpu_relu(x, saved);
    } else if (x.device == Device::gpu) {
        Tensor out(x.shape, x.device);
        saved = Tensor(x.shape, x.device);
        gpu_relu<<<CudaGetBlocks(x.N), kCudaThreadsNum>>>(x.get_ptr(), out.get_ptr(), saved.get_ptr(), x.N);
        return out;
    } else {
        throw std::runtime_error("device is not defined");
    }
}

Tensor Relu::backward(const Tensor& grad_y) {
    if (grad_y.device == Device::cpu) {
        return cpu_relu_backward(grad_y, saved);
    } else if (grad_y.device == Device::gpu) {
        Tensor grad_x(grad_y.shape, grad_y.device);
        gpu_relu_backward<<<CudaGetBlocks(grad_y.N), kCudaThreadsNum>>>(
            grad_y.get_ptr(), saved.get_ptr(), grad_x.get_ptr(), grad_y.N);
        return grad_x;
    } else {
        throw std::runtime_error("device is not defined");
    }
}

Tensor Sigmoid::forward(const Tensor& x) {
    if (x.device == Device::cpu) {
        saved = Tensor(x.shape, x.device);
        return cpu_sigmoid(x, saved);
    } else if (x.device == Device::gpu) {
        Tensor out(x.shape, x.device);
        saved = Tensor(x.shape, x.device);
        gpu_sigmoid<<<CudaGetBlocks(x.N), kCudaThreadsNum>>>(x.get_ptr(), out.get_ptr(), saved.get_ptr(), x.N);
        return out;
    } else {
        throw std::runtime_error("device is not defined");
    }
}

Tensor Sigmoid::backward(const Tensor& grad_y) {
    if (grad_y.device == Device::cpu) {
        return cpu_sigmoid_backward(grad_y, saved);
    } else if (grad_y.device == Device::gpu) {
        Tensor grad_x(grad_y.shape, grad_y.device);
        gpu_sigmoid_backward<<<CudaGetBlocks(grad_y.N), kCudaThreadsNum>>>(
            grad_y.get_ptr(), saved.get_ptr(), grad_x.get_ptr(), grad_y.N);
        return grad_x;
    } else {
        throw std::runtime_error("device is not defined");
    }
}
