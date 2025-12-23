#include<cuda_runtime.h>
#include"utils.cuh"

// ...existing code...

// GPU: (Out,In,kH,kW) -> (In*kH*kW, Out)
__global__ void flatten_OIHW_to_KO_kernel(const float* __restrict__ k,
                                          float* __restrict__ out,
                                          int Cout, int Cin, int kH, int kW){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * Cin * kH * kW;
    if (idx >= total) return;

    int kw = idx % kW;
    int t1 = idx / kW;
    int kh = t1 % kH;
    int t2 = t1 / kH;
    int ci = t2 % Cin;
    int co = t2 / Cin;

    int row = ((ci * kH) + kh) * kW + kw; // 0..Cin*kH*kW-1
    int col = co;                         // 0..Cout-1
    out[row * Cout + col] = k[idx];
}

// CPU 版本
static inline void flatten_kernel_cpu(const Tensor& kernel4d, Tensor& out2d){
    if (kernel4d.shape.size() != 4 || out2d.shape.size() != 2)
        throw std::runtime_error("flatten_kernel: shape rank mismatch");

    int Cout = kernel4d.shape[0];
    int Cin  = kernel4d.shape[1];
    int kH   = kernel4d.shape[2];
    int kW   = kernel4d.shape[3];

    int K    = Cin * kH * kW;
    if (out2d.shape[0] != K || out2d.shape[1] != Cout)
        throw std::runtime_error("flatten_kernel: out2d shape must be (Cin*kH*kW, Cout)");

    const float* src = kernel4d.get_ptr();
    float* dst       = out2d.get_ptr();

    // 遍历 (co,ci,kh,kw) 并写入 (row, col)
    for (int co = 0; co < Cout; ++co){
        for (int ci = 0; ci < Cin; ++ci){
            for (int kh = 0; kh < kH; ++kh){
                for (int kw = 0; kw < kW; ++kw){
                    int src_idx = (((co * Cin) + ci) * kH + kh) * kW + kw;
                    int row = ((ci * kH) + kh) * kW + kw;
                    int col = co;
                    dst[row * Cout + col] = src[src_idx];
                }
            }
        }
    }
}

// 对外封装
void flatten_kernel(const Tensor& kernel4d, Tensor& out2d){
    // 形状校验
    if (kernel4d.shape.size() != 4 || out2d.shape.size() != 2)
        throw std::runtime_error("flatten_kernel: shape rank mismatch");
    int Cout = kernel4d.shape[0];
    int Cin  = kernel4d.shape[1];
    int kH   = kernel4d.shape[2];
    int kW   = kernel4d.shape[3];
    int K    = Cin * kH * kW;
    if (out2d.shape[0] != K || out2d.shape[1] != Cout)
        throw std::runtime_error("flatten_kernel: out2d shape must be (Cin*kH*kW, Cout)");

    if (kernel4d.device != out2d.device)
        throw std::runtime_error("flatten_kernel: device mismatch");

    if (kernel4d.device == Device::gpu){
        int total = Cout * Cin * kH * kW;
        int blocks = CudaGetBlocks(total);
        flatten_OIHW_to_KO_kernel<<<blocks, kCudaThreadsNum>>>(
            kernel4d.get_ptr(), out2d.get_ptr(), Cout, Cin, kH, kW);
    } else {
        flatten_kernel_cpu(kernel4d, out2d);
    }
}

__global__ void col_to_image_kernel(const float* __restrict__ col_data,
                                    float* __restrict__ image_data,
                                    int N, int Cout, int H, int W) {
    // 每个线程负责 image_data 中的一个像素
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = N * Cout * H * W;

    if (index >= total_pixels) return;

    // 从线性索引反算出 N, Cout, H, W 的坐标
    int w = index % W;
    int h = (index / W) % H;
    int co = (index / (W * H)) % Cout;
    int n = index / (W * H * Cout);

    // 计算源矩阵 col_data 中的索引
    // 源矩阵的行对应 (n, h, w) 的组合
    int src_row = n * (H * W) + h * W + w;
    // 源矩阵的列对应输出通道 co
    int src_col = co;
    int src_index = src_row * Cout + src_col;

    image_data[index] = col_data[src_index];
}

// CPU 版本
static inline void reshape_col_to_image_cpu(const Tensor& col, Tensor& image) {
    const int N = image.shape[0];
    const int Cout = image.shape[1];
    const int H = image.shape[2];
    const int W = image.shape[3];

    const float* src = col.get_ptr();
    float* dst = image.get_ptr();

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < Cout; ++co) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int dst_index = (((n * Cout + co) * H) + h) * W + w;
                    int src_row = n * (H * W) + h * W + w;
                    int src_index = src_row * Cout + co;
                    dst[dst_index] = src[src_index];
                }
            }
        }
    }
}

// 对外封装
void reshape_col_to_image(const Tensor& col, Tensor& image) {
    // 形状校验
    if (col.shape.size() != 2 || image.shape.size() != 4) {
        throw std::runtime_error("reshape_col_to_image: shape rank mismatch");
    }
    const int N = image.shape[0];
    const int Cout = image.shape[1];
    const int H = image.shape[2];
    const int W = image.shape[3];
    if (col.shape[0] != N * H * W || col.shape[1] != Cout) {
        throw std::runtime_error("reshape_col_to_image: shape dimension mismatch");
    }
    if (col.device != image.device) {
        throw std::runtime_error("reshape_col_to_image: device mismatch");
    }

    if (col.device == Device::gpu) {
        const int total_pixels = N * Cout * H * W;
        const int blocks = CudaGetBlocks(total_pixels);
        col_to_image_kernel<<<blocks, kCudaThreadsNum>>>(col.get_ptr(), image.get_ptr(), N, Cout, H, W);
    } else {
        reshape_col_to_image_cpu(col, image);
    }
}

__global__ void nchw_to_rows_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H * W;
    if (idx >= total) return;

    int w  = idx % W;
    int h  = (idx / W) % H;
    int c  = (idx / (W * H)) % C;
    int n  = idx / (W * H * C);

    int row = n * (H * W) + h * W + w;
    int col = c;
    dst[(long long)row * C + col] = src[idx];
}

void pack_NCHW_rows(const Tensor& y4d, Tensor& y2d) {
    // y4d: (N,C,H,W), y2d: (N*H*W, C)
    int N = y4d.shape[0], C = y4d.shape[1], H = y4d.shape[2], W = y4d.shape[3];
    if (y2d.shape[0] != (long long)N*H*W || y2d.shape[1] != C)
        throw std::runtime_error("pack_NCHW_rows: shape mismatch");
    if (y4d.device != y2d.device) throw std::runtime_error("pack_NCHW_rows: device mismatch");

    long long total = (long long)N * C * H * W;
    nchw_to_rows_kernel<<<CudaGetBlocks((int)total), kCudaThreadsNum>>>(
        y4d.get_ptr(), y2d.get_ptr(), N, C, H, W);
}

// (Cin*kH*kW, Cout) -> (Cout,Cin,kH,kW)
__global__ void KO_to_OIHW_kernel(const float* __restrict__ K,
                                  float* __restrict__ OIHW,
                                  int Cin, int Cout, int kH, int kW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)Cout * Cin * kH * kW;
    if (idx >= total) return;

    int kw  = idx % kW;
    int kh  = (idx / kW) % kH;
    int ci  = (idx / (kW * kH)) % Cin;
    int co  = idx / (kW * kH * Cin);

    int r = ci * (kH * kW) + kh * kW + kw;  // 行：Cin*kH*kW
    int c = co;                              // 列：Cout
    OIHW[idx] = K[(long long)r * Cout + c];
}

void unflatten_KO_to_OIHW(const Tensor& K, Tensor& kernel4d) {
    int Cout = kernel4d.shape[0], Cin = kernel4d.shape[1];
    int kH = kernel4d.shape[2], kW = kernel4d.shape[3];
    if (K.shape[0] != (long long)Cin*kH*kW || K.shape[1] != Cout)
        throw std::runtime_error("unflatten_KO_to_OIHW: shape mismatch");
    if (K.device != kernel4d.device) throw std::runtime_error("unflatten_KO_to_OIHW: device mismatch");

    long long total = (long long)Cout * Cin * kH * kW;
    KO_to_OIHW_kernel<<<CudaGetBlocks((int)total), kCudaThreadsNum>>>(
        K.get_ptr(), kernel4d.get_ptr(), Cin, Cout, kH, kW);
}