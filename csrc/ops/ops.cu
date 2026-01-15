#include "ops.cuh"
#include "../layers/nn.cuh"
#include "../common/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace {

std::vector<int> compute_strides(const std::vector<int>& shape) {
    int ndim = static_cast<int>(shape.size());
    std::vector<int> strides(ndim);
    int acc = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

std::vector<int> normalize_axes(const std::vector<int>& axes, int ndim) {
    if (axes.empty()) {
        std::vector<int> all;
        all.reserve(ndim);
        for (int i = 0; i < ndim; ++i) {
            all.push_back(i);
        }
        return all;
    }
    std::vector<int> norm;
    norm.reserve(axes.size());
    for (int ax : axes) {
        int a = ax;
        if (a < 0) {
            a += ndim;
        }
        if (a < 0 || a >= ndim) {
            throw std::runtime_error("axis out of range");
        }
        norm.push_back(a);
    }
    return norm;
}

struct DeviceIntArray {
    int* ptr{nullptr};
    int size{0};

    DeviceIntArray(const std::vector<int>& v) {
        size = static_cast<int>(v.size());
        if (size == 0) {
            ptr = nullptr;
            return;
        }
        cudaMalloc(&ptr, sizeof(int) * size);
        cudaMemcpy(ptr, v.data(), sizeof(int) * size, cudaMemcpyHostToDevice);
    }
    ~DeviceIntArray() {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = a[i] + b[i]; }
}
__global__ void add_scalar_kernel(const float* a, float scalar, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = a[i] + scalar; }
}
__global__ void mul_kernel(const float* a, const float* b, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = a[i] * b[i]; }
}
__global__ void mul_scalar_kernel(const float* a, float scalar, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = a[i] * scalar; }
}
__global__ void div_kernel(const float* a, const float* b, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = a[i] / b[i]; }
}
__global__ void div_scalar_kernel(const float* a, float scalar, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = a[i] / scalar; }
}
__global__ void pow_kernel(const float* a, const float* b, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = powf(a[i], b[i]); }
}
__global__ void pow_scalar_kernel(const float* a, float scalar, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = powf(a[i], scalar); }
}
__global__ void neg_kernel(const float* a, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = -a[i]; }
}
__global__ void exp_kernel(const float* a, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = expf(a[i]); }
}
__global__ void log_kernel(const float* a, float* out, int n) {
    CUDAKERNELLOOP(i, n) { out[i] = logf(a[i]); }
}

__global__ void transpose_kernel(
    const float* input,
    float* output,
    const int* in_strides,
    const int* perm,
    const int* out_strides,
    int ndim,
    int out_size
) {
    CUDAKERNELLOOP(i, out_size) {
        int idx = i;
        int in_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            int coord = idx / out_strides[d];
            idx -= coord * out_strides[d];
            int in_dim = perm[d];
            in_offset += coord * in_strides[in_dim];
        }
        output[i] = input[in_offset];
    }
}

__global__ void broadcast_kernel(
    const float* input,
    float* output,
    const int* in_strides,
    const int* out_strides,
    int ndim,
    int out_size
) {
    CUDAKERNELLOOP(i, out_size) {
        int idx = i;
        int in_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            int coord = idx / out_strides[d];
            idx -= coord * out_strides[d];
            in_offset += coord * in_strides[d];
        }
        output[i] = input[in_offset];
    }
}

__global__ void sum_kernel(
    const float* input,
    float* output,
    const int* in_strides,
    const int* out_strides,
    const int* reduce_axes,
    const int* reduce_strides,
    const int* reduce_sizes,
    int ndim,
    int out_ndim,
    int reduce_ndim,
    int out_size,
    int reduce_count
) {
    CUDAKERNELLOOP(i, out_size) {
        int idx = i;
        int base_offset = 0;
        int out_dim = 0;
        for (int d = 0; d < ndim; ++d) {
            bool is_reduce = false;
            for (int r = 0; r < reduce_ndim; ++r) {
                if (reduce_axes[r] == d) {
                    is_reduce = true;
                    break;
                }
            }
            if (is_reduce) {
                continue;
            }
            int coord = idx / out_strides[out_dim];
            idx -= coord * out_strides[out_dim];
            base_offset += coord * in_strides[d];
            out_dim += 1;
        }

        float sum = 0.0f;
        for (int r = 0; r < reduce_count; ++r) {
            int tmp = r;
            int offset = 0;
            for (int j = 0; j < reduce_ndim; ++j) {
                int coord = 0;
                if (reduce_strides[j] != 0) {
                    coord = tmp / reduce_strides[j];
                    tmp -= coord * reduce_strides[j];
                }
                offset += coord * in_strides[reduce_axes[j]];
            }
            sum += input[base_offset + offset];
        }
        output[i] = sum;
    }
}

}  // namespace

namespace Ops {

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("add: device mismatch");
    }
    if (a.shape != b.shape) {
        throw std::runtime_error("add: shape mismatch");
    }
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("add: CPU path removed");
    }
    add_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), b.get_ptr(), out.get_ptr(), n);
    return out;
}

Tensor add_scalar(const Tensor& a, float scalar) {
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("add_scalar: CPU path removed");
    }
    add_scalar_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), scalar, out.get_ptr(), n);
    return out;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("multiply: device mismatch");
    }
    if (a.shape != b.shape) {
        throw std::runtime_error("multiply: shape mismatch");
    }
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("multiply: CPU path removed");
    }
    mul_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), b.get_ptr(), out.get_ptr(), n);
    return out;
}

Tensor mul_scalar(const Tensor& a, float scalar) {
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("mul_scalar: CPU path removed");
    }
    mul_scalar_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), scalar, out.get_ptr(), n);
    return out;
}

Tensor divide(const Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("divide: device mismatch");
    }
    if (a.shape != b.shape) {
        throw std::runtime_error("divide: shape mismatch");
    }
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("divide: CPU path removed");
    }
    div_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), b.get_ptr(), out.get_ptr(), n);
    return out;
}

Tensor div_scalar(const Tensor& a, float scalar) {
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("div_scalar: CPU path removed");
    }
    div_scalar_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), scalar, out.get_ptr(), n);
    return out;
}

Tensor power(const Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("power: device mismatch");
    }
    if (a.shape != b.shape) {
        throw std::runtime_error("power: shape mismatch");
    }
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("power: CPU path removed");
    }
    pow_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), b.get_ptr(), out.get_ptr(), n);
    return out;
}

Tensor power_scalar(const Tensor& a, float scalar) {
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("power_scalar: CPU path removed");
    }
    pow_scalar_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), scalar, out.get_ptr(), n);
    return out;
}

Tensor negate(const Tensor& a) {
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("negate: CPU path removed");
    }
    neg_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), out.get_ptr(), n);
    return out;
}

Tensor exp(const Tensor& a) {
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("exp: CPU path removed");
    }
    exp_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), out.get_ptr(), n);
    return out;
}

Tensor log(const Tensor& a) {
    Tensor out(a.shape, a.device);
    int n = static_cast<int>(a.N);
    if (a.device == Device::cpu) {
        throw std::runtime_error("log: CPU path removed");
    }
    log_kernel<<<CudaGetBlocks(n), kCudaThreadsNum>>>(a.get_ptr(), out.get_ptr(), n);
    return out;
}

Tensor reshape(const Tensor& a, const std::vector<int>& new_shape) {
    if (a.device == Device::cpu) {
        throw std::runtime_error("reshape: CPU path removed");
    }
    if (new_shape.empty()) {
        throw std::runtime_error("reshape: new_shape must not be empty");
    }
    long long inferred = -1;
    long long known = 1;
    int infer_idx = -1;
    for (int i = 0; i < static_cast<int>(new_shape.size()); ++i) {
        int dim = new_shape[i];
        if (dim == -1) {
            if (infer_idx != -1) {
                throw std::runtime_error("reshape: only one dimension can be inferred");
            }
            infer_idx = i;
        } else if (dim <= 0) {
            throw std::runtime_error("reshape: dimensions must be positive or -1");
        } else {
            known *= dim;
        }
    }
    long long total = static_cast<long long>(a.N);
    std::vector<int> out_shape = new_shape;
    if (infer_idx != -1) {
        if (known == 0 || total % known != 0) {
            throw std::runtime_error("reshape: cannot infer shape with incompatible size");
        }
        inferred = total / known;
        out_shape[infer_idx] = static_cast<int>(inferred);
    }
    long long out_total = 1;
    for (int dim : out_shape) {
        out_total *= dim;
    }
    if (out_total != total) {
        throw std::runtime_error("reshape: total size mismatch");
    }

    Tensor out(out_shape, a.device);
    size_t bytes = sizeof(float) * static_cast<size_t>(a.N);
    cudaMemcpy(out.get_ptr(), a.get_ptr(), bytes, cudaMemcpyDeviceToDevice);
    return out;
}

Tensor transpose(const Tensor& a, const std::vector<int>& axes) {
    int ndim = static_cast<int>(a.shape.size());
    if (ndim < 2) {
        throw std::runtime_error("transpose: ndim must be >= 2");
    }

    std::vector<int> perm(ndim);
    if (axes.empty()) {
        for (int i = 0; i < ndim; ++i) {
            perm[i] = i;
        }
        std::swap(perm[ndim - 1], perm[ndim - 2]);
    } else if (static_cast<int>(axes.size()) == 2) {
        for (int i = 0; i < ndim; ++i) {
            perm[i] = i;
        }
        int i = axes[0] < 0 ? axes[0] + ndim : axes[0];
        int j = axes[1] < 0 ? axes[1] + ndim : axes[1];
        if (i < 0 || i >= ndim || j < 0 || j >= ndim) {
            throw std::runtime_error("transpose: axis out of range");
        }
        std::swap(perm[i], perm[j]);
    } else {
        if (static_cast<int>(axes.size()) != ndim) {
            throw std::runtime_error("transpose: axes size mismatch");
        }
        for (int i = 0; i < ndim; ++i) {
            int ax = axes[i] < 0 ? axes[i] + ndim : axes[i];
            if (ax < 0 || ax >= ndim) {
                throw std::runtime_error("transpose: axis out of range");
            }
            perm[i] = ax;
        }
    }

    std::vector<int> out_shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        out_shape[i] = a.shape[perm[i]];
    }

    Tensor out(out_shape, a.device);
    int out_size = static_cast<int>(out.N);
    std::vector<int> in_strides = compute_strides(a.shape);
    std::vector<int> out_strides = compute_strides(out_shape);

    if (a.device == Device::cpu) {
        throw std::runtime_error("transpose: CPU path removed");
    }
    DeviceIntArray d_in_strides(in_strides);
    DeviceIntArray d_perm(perm);
    DeviceIntArray d_out_strides(out_strides);
    transpose_kernel<<<CudaGetBlocks(out_size), kCudaThreadsNum>>>(
        a.get_ptr(),
        out.get_ptr(),
        d_in_strides.ptr,
        d_perm.ptr,
        d_out_strides.ptr,
        ndim,
        out_size
    );
    return out;
}

Tensor broadcast_to(const Tensor& a, const std::vector<int>& shape) {
    int in_ndim = static_cast<int>(a.shape.size());
    int out_ndim = static_cast<int>(shape.size());
    if (out_ndim < in_ndim) {
        throw std::runtime_error("broadcast_to: output rank smaller than input");
    }

    std::vector<int> padded_shape(out_ndim, 1);
    std::vector<int> padded_strides(out_ndim, 0);
    std::vector<int> in_strides = compute_strides(a.shape);

    int pad = out_ndim - in_ndim;
    for (int i = 0; i < out_ndim; ++i) {
        int in_dim = (i < pad) ? 1 : a.shape[i - pad];
        int out_dim = shape[i];
        if (in_dim != out_dim && in_dim != 1) {
            throw std::runtime_error("broadcast_to: incompatible shape");
        }
        padded_shape[i] = in_dim;
        int stride = (i < pad) ? 0 : in_strides[i - pad];
        if (in_dim == 1 && out_dim > 1) {
            stride = 0;
        }
        padded_strides[i] = stride;
    }

    Tensor out(shape, a.device);
    int out_size = static_cast<int>(out.N);
    std::vector<int> out_strides = compute_strides(shape);

    if (a.device == Device::cpu) {
        throw std::runtime_error("broadcast_to: CPU path removed");
    }
    DeviceIntArray d_in_strides(padded_strides);
    DeviceIntArray d_out_strides(out_strides);
    broadcast_kernel<<<CudaGetBlocks(out_size), kCudaThreadsNum>>>(
        a.get_ptr(),
        out.get_ptr(),
        d_in_strides.ptr,
        d_out_strides.ptr,
        out_ndim,
        out_size
    );
    return out;
}

Tensor summation(const Tensor& a, const std::vector<int>& axes) {
    int ndim = static_cast<int>(a.shape.size());
    std::vector<int> norm_axes = normalize_axes(axes, ndim);

    std::vector<int> reduce_flags(ndim, 0);
    for (int ax : norm_axes) {
        reduce_flags[ax] = 1;
    }

    std::vector<int> out_shape;
    for (int i = 0; i < ndim; ++i) {
        if (!reduce_flags[i]) {
            out_shape.push_back(a.shape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    Tensor out(out_shape, a.device);
    int out_size = static_cast<int>(out.N);

    std::vector<int> in_strides = compute_strides(a.shape);
    std::vector<int> out_strides = compute_strides(out_shape);

    std::vector<int> reduce_axes;
    std::vector<int> reduce_sizes;
    for (int i = 0; i < ndim; ++i) {
        if (reduce_flags[i]) {
            reduce_axes.push_back(i);
            reduce_sizes.push_back(a.shape[i]);
        }
    }

    int reduce_ndim = static_cast<int>(reduce_axes.size());
    int reduce_count = 1;
    for (int s : reduce_sizes) {
        reduce_count *= s;
    }

    std::vector<int> reduce_strides(reduce_ndim, 0);
    int acc = 1;
    for (int i = reduce_ndim - 1; i >= 0; --i) {
        reduce_strides[i] = acc;
        acc *= reduce_sizes[i];
    }

    if (a.device == Device::cpu) {
        throw std::runtime_error("summation: CPU path removed");
    }
    DeviceIntArray d_in_strides(in_strides);
    DeviceIntArray d_out_strides(out_strides);
    DeviceIntArray d_reduce_axes(reduce_axes);
    DeviceIntArray d_reduce_strides(reduce_strides);
    DeviceIntArray d_reduce_sizes(reduce_sizes);
    sum_kernel<<<CudaGetBlocks(out_size), kCudaThreadsNum>>>(
        a.get_ptr(),
        out.get_ptr(),
        d_in_strides.ptr,
        d_out_strides.ptr,
        d_reduce_axes.ptr,
        d_reduce_strides.ptr,
        d_reduce_sizes.ptr,
        ndim,
        static_cast<int>(out_shape.size()),
        reduce_ndim,
        out_size,
        reduce_count
    );
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.device != b.device) {
        throw std::runtime_error("matmul: device mismatch");
    }
    if (a.shape.size() < 2 || b.shape.size() < 2) {
        throw std::runtime_error("matmul: inputs must be at least 2D");
    }
    int a_ndim = static_cast<int>(a.shape.size());
    int b_ndim = static_cast<int>(b.shape.size());
    int m = a.shape[a_ndim - 2];
    int k = a.shape[a_ndim - 1];
    int kb = b.shape[b_ndim - 2];
    int n = b.shape[b_ndim - 1];
    if (k != kb) {
        throw std::runtime_error("matmul: inner dimensions mismatch");
    }

    std::vector<int> a_batch(a.shape.begin(), a.shape.end() - 2);
    std::vector<int> b_batch(b.shape.begin(), b.shape.end() - 2);
    int a_batch_ndim = static_cast<int>(a_batch.size());
    int b_batch_ndim = static_cast<int>(b_batch.size());
    int batch_ndim = (a_batch_ndim > b_batch_ndim) ? a_batch_ndim : b_batch_ndim;

    int a_pad = batch_ndim - a_batch_ndim;
    int b_pad = batch_ndim - b_batch_ndim;

    std::vector<int> batch_shape(batch_ndim, 1);
    for (int i = 0; i < batch_ndim; ++i) {
        int a_dim = (i < a_pad) ? 1 : a_batch[i - a_pad];
        int b_dim = (i < b_pad) ? 1 : b_batch[i - b_pad];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::runtime_error("matmul: batch shape mismatch");
        }
        batch_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    std::vector<int> out_shape = batch_shape;
    out_shape.push_back(m);
    out_shape.push_back(n);
    Tensor out(out_shape, a.device);

    std::vector<int> a_strides = compute_strides(a.shape);
    std::vector<int> b_strides = compute_strides(b.shape);
    std::vector<int> out_strides = compute_strides(out_shape);

    std::vector<int> a_batch_strides(batch_ndim, 0);
    std::vector<int> b_batch_strides(batch_ndim, 0);
    for (int i = 0; i < batch_ndim; ++i) {
        int a_dim = (i < a_pad) ? 1 : a_batch[i - a_pad];
        int b_dim = (i < b_pad) ? 1 : b_batch[i - b_pad];
        if (i >= a_pad) {
            int stride = a_strides[i - a_pad];
            if (a_dim == 1 && batch_shape[i] > 1) {
                stride = 0;
            }
            a_batch_strides[i] = stride;
        }
        if (i >= b_pad) {
            int stride = b_strides[i - b_pad];
            if (b_dim == 1 && batch_shape[i] > 1) {
                stride = 0;
            }
            b_batch_strides[i] = stride;
        }
    }

    long long batch_count = 1;
    for (int d : batch_shape) {
        batch_count *= d;
    }

    if (a.device == Device::cpu) {
        throw std::runtime_error("matmul: CPU path removed");
    }
    for (long long bi = 0; bi < batch_count; ++bi) {
        long long tmp = bi;
        long long a_off = 0;
        long long b_off = 0;
        long long out_off = 0;
        for (int i = batch_ndim - 1; i >= 0; --i) {
            int coord = static_cast<int>(tmp % batch_shape[i]);
            tmp /= batch_shape[i];
            a_off += static_cast<long long>(coord) * a_batch_strides[i];
            b_off += static_cast<long long>(coord) * b_batch_strides[i];
            out_off += static_cast<long long>(coord) * out_strides[i];
        }
        Tensor A_tmp({m, k}, Device::gpu);
        Tensor B_tmp({k, n}, Device::gpu);
        Tensor C_tmp({m, n}, Device::gpu);
        cudaMemcpy(A_tmp.get_ptr(), a.get_ptr() + a_off, sizeof(float) * m * k, cudaMemcpyDeviceToDevice);
        cudaMemcpy(B_tmp.get_ptr(), b.get_ptr() + b_off, sizeof(float) * k * n, cudaMemcpyDeviceToDevice);
        nn::gemm_gpu(A_tmp, B_tmp, C_tmp);
        cudaMemcpy(out.get_ptr() + out_off, C_tmp.get_ptr(), sizeof(float) * m * n, cudaMemcpyDeviceToDevice);
    }
    return out;
}

}  // namespace Ops
