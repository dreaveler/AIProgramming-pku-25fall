#pragma once
#ifndef FUNCTION_CUH
#define FUNCTION_CUH

#include"tensor.cuh"
#include<cublas_v2.h>

class Function{
public:
    virtual Tensor forward(const Tensor&) = 0;
    virtual Tensor backward(const Tensor&) = 0;
    virtual ~Function() = default;
protected:
    Tensor saved;
};

class Relu:public Function{
public:
    Relu() = default;
    Tensor forward(const Tensor&);
    Tensor backward(const Tensor&);
    ~Relu() = default;
};

class Sigmoid :public Function{
public:
    Sigmoid() = default;
    Tensor forward(const Tensor&);
    Tensor backward(const Tensor&);
    ~Sigmoid() = default;
};


namespace HW3{
    void gemm_gpu(const Tensor& A,const Tensor& B,Tensor& C,const float alpha = 1.0,const float beta = 0.0,const bool A_trans = false,const bool B_trans = false);
    void FC_forward(const Tensor& input,const Tensor& weight,const Tensor& bias,Tensor& output);
    void FC_forward_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,Tensor& output, int batch_size, int in_c, int out_c);
    void FC_backward(const Tensor& input,const Tensor& weight,const Tensor& bias,Tensor& output,
    const Tensor& grad_output , Tensor& grad_input , Tensor& grad_weights,Tensor& grad_bias);
    void FC_backward_cpu(const Tensor& input,const Tensor& weight,const Tensor& grad_output,
                          Tensor& grad_input,Tensor& grad_weights,Tensor& grad_bias,
                          int batch_size,int in_c,int out_c);
    void img2col(const Tensor& img,Tensor& col,const int padding = 1,const int stride=1,const int k_h = 3,const int k_w = 3);
    void convolve(const Tensor& img,const Tensor& kernel,Tensor& output,const int padding=1,const int stride=1);
    void convolve_backward(const Tensor& input,const Tensor& kernel,const Tensor& grad_y,Tensor& grad_input,Tensor& grad_kernel);
    void col2img(const Tensor& gard_col,Tensor& grad_img,const int padding=1,const int stride=1,const int k_h = 3,const int k_w = 3);

    void maxpooling(const Tensor& input,Tensor& output,Tensor& mask);
    void maxpooling_backward(const Tensor& grad_y,const Tensor& mask,Tensor& grad_x);

    void softmax(const Tensor& input,Tensor& output);

    void crossentropyloss(const Tensor& input,const Tensor& label,Tensor& output);
    void softmaxsel_backward(const Tensor& soutput,const Tensor& label,Tensor& grad_sinput);
}

const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N){
    return (N + kCudaThreadsNum -1) / kCudaThreadsNum;
}
#define CUDAKERNELLOOP(i,n)                             \
for(int i = blockIdx.x * blockDim.x + threadIdx.x;      \
i < (n);                                                \
i += blockDim.x  * gridDim.x)

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#endif