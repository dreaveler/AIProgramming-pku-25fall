#pragma once
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDAKERNELLOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

#endif
