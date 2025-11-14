#include"function.cuh"
#include<cuda_runtime.h>
#include"utils.cuh"

__global__ void gpu_relu(const float* in , float* out , float* mask , int N){
    CUDAKERNELLOOP(i,N){
        float v = in[i];
        out[i] = v > 0 ? v : 0.0f;
        mask[i] = v > 0 ? 1.0f : 0.0f;
    }
}
__global__ void gpu_relu_backward(const float* grad_y,const float* mask, float* grad_x,int N){
    CUDAKERNELLOOP(i , N){
        grad_x[i] = grad_y[i] * mask[i];
    }
}

__global__ void gpu_sigmoid(const float* in, float* out , float* saved, int N){
    CUDAKERNELLOOP(i , N){
        float v = in[i];
        float s = 1.0f / (1.0f + expf(-v));
        out[i] = s;
        saved[i] = s;
    }
}
__global__ void gpu_sigmoid_backward(const float* grad_y , const float* y_saved , float* grad_x,int N){
    CUDAKERNELLOOP(i , N){
        float s = y_saved[i];
        grad_x[i] = grad_y[i] * s * (1.0f - s);
    }
}

Tensor cpu_relu(const Tensor& x,Tensor& saved){
    Tensor out(x.shape,x.device);
    for(auto i = 0 ; i < x.N ; i++){
        out.get_ptr()[i] = x.get_ptr()[i] > 0 ? x.get_ptr()[i] : 0.f;
        saved.get_ptr()[i] = x.get_ptr()[i] > 0 ? 1.f : 0.f;
    }
    return out;
}

Tensor cpu_relu_backward(const Tensor& grad_y,const Tensor& saved){
    Tensor grad_x(grad_y.shape,grad_y.device);
    for(auto i = 0 ; i < grad_y.N ; i++){
        grad_x.get_ptr()[i] = grad_y.get_ptr()[i] * saved.get_ptr()[i];
    }
    return grad_x;
}

Tensor cpu_sigmoid(const Tensor& x,Tensor& saved){
    Tensor out(x.shape,x.device);
    for(auto i = 0 ; i < x.N ; i++){
        out.get_ptr()[i] = 1.0 / (1.0 + expf(-x.get_ptr()[i]));
        saved.get_ptr()[i] = out.get_ptr()[i];
    }
    return out;
}

Tensor cpu_sigmoid_backward(const Tensor& grad_y,const Tensor& saved){
    Tensor grad_x(grad_y.shape,grad_y.device);
    for(auto i = 0 ; i < grad_y.N ; i++){
        grad_x.get_ptr()[i] = grad_y.get_ptr()[i] * saved.get_ptr()[i] * (1.0 - saved.get_ptr()[i]);
    }
    return grad_x;
}


Tensor Relu::forward(const Tensor& x){
    if(x.device == Device::cpu){
        saved = Tensor(x.shape,x.device);
        return cpu_relu(x,saved);
    }
    else if(x.device == Device::gpu){
        Tensor out(x.shape,x.device);
        saved = Tensor(x.shape,x.device);
        gpu_relu<<<CudaGetBlocks(x.N),kCudaThreadsNum>>>(x.get_ptr(),out.get_ptr(),saved.get_ptr(),x.N);
        return out;
    }else{
        throw std::runtime_error("device is not defined");
    }
}

Tensor Relu::backward(const Tensor& grad_y){
    if(grad_y.device == Device::cpu) return cpu_relu_backward(grad_y,saved);
    else if(grad_y.device == Device::gpu){
        Tensor grad_x(grad_y.shape,grad_y.device);
        gpu_relu_backward<<<CudaGetBlocks(grad_y.N),kCudaThreadsNum>>>(grad_y.get_ptr(),saved.get_ptr(),grad_x.get_ptr(),grad_y.N);
        return grad_x;
    }else{
        throw std::runtime_error("device is not defined");
    }
}

Tensor Sigmoid::forward(const Tensor& x){
    if(x.device == Device::cpu) {
        saved = Tensor(x.shape,x.device);
        return cpu_sigmoid(x,saved);
    }else if(x.device == Device::gpu){
        Tensor out(x.shape, x.device);
        saved = Tensor(x.shape, x.device); 
        gpu_sigmoid<<<CudaGetBlocks(x.N), kCudaThreadsNum>>>(x.get_ptr(), out.get_ptr(),saved.get_ptr(),x.N);
        return out;
    }else{
        throw std::runtime_error("device is not defined");
    }
}

Tensor Sigmoid::backward(const Tensor& grad_y){
    if(grad_y.device == Device::cpu) return cpu_sigmoid_backward(grad_y,saved);
    else if(grad_y.device == Device::gpu){
        Tensor grad_x(grad_y.shape, grad_y.device);
        gpu_sigmoid_backward<<<CudaGetBlocks(grad_y.N), kCudaThreadsNum>>>(grad_y.get_ptr(), saved.get_ptr(), grad_x.get_ptr(), grad_y.N);
        return grad_x;
    }else{
        throw std::runtime_error("device is not defined");
    }
}

//A(m,k)*B(k,n)  由于cuda是列主序  需要进行转置之后的乘法
void HW3::gemm_gpu(const Tensor& A, const Tensor& B, Tensor& C,
                   const float alf, const float bet,
                   const bool A_trans, const bool B_trans) {
    
    // 验证输入维度
    const int A_rows = A.shape[0];
    const int A_cols = A.shape[1];
    const int B_rows = B.shape[0];
    const int B_cols = B.shape[1];
    const int C_rows = C.shape[0];
    const int C_cols = C.shape[1];

    // 计算数学维度
    const int m = C_rows;  // C的行数
    const int n = C_cols;  // C的列数
    const int k = A_trans ? A_rows : A_cols;  // 内积维度

    const int lda = A_cols;
    const int ldb = B_cols;
    const int ldc = C_cols;

    // 转置操作
    const cublasOperation_t transA = A_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transB = B_trans ? CUBLAS_OP_T : CUBLAS_OP_N;

    const float* alpha = &alf;
    const float* beta = &bet;
    
    cublasHandle_t handle;  cublasCreate(&handle);
    cublasSgemm(handle,transB,transA,
                n,m,k, 
                alpha,
                B.get_ptr(), ldb, 
                A.get_ptr(), lda, 
                beta, 
                C.get_ptr(), ldc); 
    cublasDestroy(handle);
}

void HW3::FC_forward(const Tensor& input,const Tensor& weight,const Tensor& bias,
Tensor& output){
    int batch_size = input.shape[0];
    int in_c = input.shape[1];
    int out_c = output.shape[1];
    gemm_gpu(input,weight,output);
    gemm_gpu(Tensor::ones({batch_size,1},Device::gpu),bias,output,1.0,1.0);
}

void HW3::FC_forward_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias,
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

void HW3::FC_backward(const Tensor& input,const Tensor& weight,const Tensor& bias,Tensor& output,
const Tensor& grad_output , Tensor& grad_input , Tensor& grad_weights,Tensor& grad_bias){
    int batch_size = input.shape[0];
    int in_c = input.shape[1];
    int out_c = output.shape[1];
    gemm_gpu(grad_output,weight,grad_input,1.0,0.0,false,true);
    gemm_gpu(input,grad_output,grad_weights,1.0,0.0,true,false);
    gemm_gpu(Tensor::ones({batch_size,1},Device::gpu),grad_output,grad_bias,1.0,0.0,true,false);
}

void HW3::FC_backward_cpu(const Tensor& input,
                          const Tensor& weight,
                          const Tensor& grad_output,
                          Tensor& grad_input,
                          Tensor& grad_weights,
                          Tensor& grad_bias,
                          int batch_size,
                          int in_c,
                          int out_c){
    // grad_input = grad_output * W^T
    std::fill(grad_input.get_ptr(), grad_input.get_ptr() + batch_size * in_c, 0.f);
    for(int b = 0; b < batch_size; ++b){
        for(int i = 0; i < in_c; ++i){
            float sum = 0.f;
            for(int o = 0; o < out_c; ++o){
                sum += grad_output.get_ptr()[b*out_c + o] * weight.get_ptr()[i*out_c + o];
            }
            grad_input.get_ptr()[b*in_c + i] = sum;
        }
    }

    // grad_weights = X^T * grad_output
    std::fill(grad_weights.get_ptr(), grad_weights.get_ptr() + in_c * out_c, 0.f);
    for(int i = 0; i < in_c; ++i){
        for(int o = 0; o < out_c; ++o){
            float sum = 0.f;
            for(int b = 0; b < batch_size; ++b){
                sum += input.get_ptr()[b*in_c + i] * grad_output.get_ptr()[b*out_c + o];
            }
            grad_weights.get_ptr()[i*out_c + o] = sum;
        }
    }

    // grad_bias = sum(grad_output, axis=0)
    std::fill(grad_bias.get_ptr(), grad_bias.get_ptr() + out_c, 0.f);
    for(int o = 0; o < out_c; ++o){
        float sum = 0.f;
        for(int b = 0; b < batch_size; ++b){
            sum += grad_output.get_ptr()[b*out_c + o];
        }
        grad_bias.get_ptr()[o] = sum;
    }
}

__global__ void kernel_img2col(const float* img, float* col,
                               int N, int C, int H, int W,
                               int k_h=3, int k_w=3, int stride=1, int padding=1) {
    int H_out = H;
    int W_out = W;
    CUDAKERNELLOOP(i, (long long)N * H_out * W_out * C * k_h * k_w) {
        // 从 col 的线性索引 i 反算出其多维坐标
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

void HW3::img2col(const Tensor& img,Tensor& col,const int padding,const int stride,const int k_h,const int k_w){
    int N = img.shape[0] , C = img.shape[1] , H = img.shape[2] , W = img.shape[3];
    int total_num = N * C * H * W * k_h * k_w;
    kernel_img2col<<<CudaGetBlocks(total_num),kCudaThreadsNum>>>(img.get_ptr(),col.get_ptr(),N,C,H,W);
}

void HW3::convolve(const Tensor& img,const Tensor& kernel,Tensor& output,const int padding,const int stride){
    int Cout = kernel.shape[0],Cin = kernel.shape[1],k_h = kernel.shape[2],k_w = kernel.shape[3];
    int N = img.shape[0] , C = img.shape[1] , H = img.shape[2] , W = img.shape[3];
    Tensor col({N*H*W,C*k_h*k_w},Device::gpu);
    Tensor flattened_kernel({Cin*k_h*k_w,Cout},Device::gpu);
    flatten_kernel(kernel,flattened_kernel);
    img2col(img,col);
    Tensor ans_N({N*H*W,Cout},Device::gpu);
    gemm_gpu(col,flattened_kernel,ans_N);
    reshape_col_to_image(ans_N, output);
}

void HW3::convolve_backward(const Tensor& img,const Tensor& kernel,const Tensor& grad_y,Tensor& grad_img,Tensor& grad_kernel){
    int Cout = kernel.shape[0],Cin = kernel.shape[1],k_h = kernel.shape[2],k_w = kernel.shape[3];
    int N = img.shape[0] , C = img.shape[1] , H = img.shape[2] , W = img.shape[3];
    Tensor col = Tensor::zeros({N*H*W,C*k_h*k_w},Device::gpu);
    img2col(img,col);


    Tensor grad_col = Tensor::zeros({N*H*W,C*k_h*k_w},Device::gpu);
    grad_img = Tensor::zeros(grad_img.shape,grad_img.device);

    Tensor grad_y_reshaped = Tensor({N*H*W,Cout},Device::gpu);
    pack_NCHW_rows(grad_y, grad_y_reshaped);

    Tensor grad_kernel_reshaped =  Tensor({Cin * k_h * k_w,Cout},Device::gpu);
    gemm_gpu(col,grad_y_reshaped,grad_kernel_reshaped,1.0,0.0,true,false);

    unflatten_KO_to_OIHW(grad_kernel_reshaped, grad_kernel);

    Tensor flattened_kernel = Tensor::zeros({Cin*k_h*k_w,Cout},Device::gpu);
    flatten_kernel(kernel,flattened_kernel);
    gemm_gpu(grad_y_reshaped,flattened_kernel,grad_col,1.0,0.0,false,true);

    col2img(grad_col,grad_img);
}

__global__ void col2img_kernel(const float* grad_col,float* grad_img,
    const int N,const int C,const int H,const int W,
    const int stride=1,const int padding=1,
    const int k_h=3,const int k_w=3){
    int OH = H,OW = W;
    CUDAKERNELLOOP(i, (long long)N * OH * OW * C * k_h * k_w) {
        // 从 col 的线性索引 i 反算出其多维坐标
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
            atomicAdd(&grad_img[img_idx],grad_col[i]);
        }
    }
}

void HW3::col2img(const Tensor& grad_col,Tensor& grad_img,const int stride,const int pdding,const int k_h,const int k_w){
    int N = grad_img.shape[0] , C = grad_img.shape[1] , H = grad_img.shape[2] , W = grad_img.shape[3];
    int total_num = N * C * H * W * k_h * k_w;
    col2img_kernel<<<CudaGetBlocks(total_num),kCudaThreadsNum>>>(grad_col.get_ptr(),grad_img.get_ptr(),N,C,H,W);
}

__global__ void kernel_maxpooling(const float* input,float* output,int* mask,
const int N, const int C , const int H , const int W,
const int k_h = 2,const int k_w =2,const int stride = 2,const int padding = 0){
    int OH = H/2,OW = W/2;
    CUDAKERNELLOOP(i,N*C*OH*OW){
        int n = i / (C*OH*OW);
        int c = (i % (C*OH*OW))/(OH*OW);
        int ph = (i % (OH*OW)) / OW;
        int pw = i % OW;
        const int oh = ph * stride;
        const int ow = pw * stride;
        float max_val = -FLT_MAX;
        int max_idx = -1;
        for(int j = oh ; j < oh +2 ; j++){
            for(int k = ow; k < ow + 2 ; k++){
                const int input_idx = ((n*C+c)*H + j) * W + k;
                float val = input[input_idx];
                if(val > max_val){
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }
        output[i] = max_val;
        mask[i] = max_idx;
    }
}

void HW3::maxpooling(const Tensor& input,Tensor& output,Tensor& mask){
    int N = input.shape[0],C = input.shape[1],H = input.shape[2],W = input.shape[3];
    int OH  = H/2, OW = W / 2;
    int total_num = N * C * OH*OW;
    kernel_maxpooling<<<CudaGetBlocks(total_num),kCudaThreadsNum>>>(input.get_ptr(),output.get_ptr(),reinterpret_cast<int*>(mask.get_ptr()),N,C,H,W);
}

__global__ void kernel_maxpooling_backward(const float* grad_y,const int* mask,float* grad_input,
const int N,const int C,const int H,const int W,const int OH,const int OW,
const int k_h = 2,const int k_w =2,const int stride = 2,const int padding = 0){
    const int output_size = N * C * OH * OW;
    CUDAKERNELLOOP(index, output_size) {
        const int max_index = mask[index];
        const float grad = grad_y[index];
        
        if (max_index >= 0) {
            atomicAdd(&grad_input[max_index], grad);
        }
    }
}

void HW3::maxpooling_backward(const Tensor& grad_y,const Tensor& mask,Tensor& grad_input){
    int N = grad_input.shape[0],C = grad_input.shape[1],H = grad_input.shape[2],W = grad_input.shape[3];
    int OH = grad_y.shape[2],OW = grad_y.shape[3];
    int total_num = N * C * OH * OW;
    grad_input = Tensor::zeros(grad_input.shape,Device::gpu);
    kernel_maxpooling_backward<<<CudaGetBlocks(total_num),kCudaThreadsNum>>>(grad_y.get_ptr(),reinterpret_cast<const int*>(mask.get_ptr()),grad_input.get_ptr(),N,C,H,W,OH,OW);
}


__global__ void kernel_softmax(const float* input,float* output,int N,int C){
    CUDAKERNELLOOP(i,N){
        const float* input_ptr = input + i * C;   //似乎可以使用shared_memory进行加速
        float* output_ptr = output + i * C;
        float sum = 0.0f;
        for(int c = 0; c < C; c++){
            sum += expf(input_ptr[c]);
        }
        for(int c = 0; c < C; c++){
            output_ptr[c] = expf(input_ptr[c]) / sum;
        }
    }
}

void HW3::softmax(const Tensor& input,Tensor& output){
    int N = input.shape[0];
    int C = input.shape[1];
    kernel_softmax<<<CudaGetBlocks(N),kCudaThreadsNum>>>(input.get_ptr(),output.get_ptr(),N,C);
}

__global__ void kernel_crosseloss(const float* input,const float* label,float* output,int N,int C){
    CUDAKERNELLOOP(i,N){
        const float* input_ptr = input + i * C;
        int label_i = static_cast<int>(label[i]);
        output[i] = - log(input_ptr[label_i]); 
    }
}


void HW3::crossentropyloss(const Tensor& input,const Tensor& label,Tensor& output){
    int N = input.shape[0];
    int C = input.shape[1];
    kernel_crosseloss<<<CudaGetBlocks(N),kCudaThreadsNum>>>(input.get_ptr(),label.get_ptr(),output.get_ptr(),N,C);
}


__global__ void kernel_smel_backward(const float* soutput,const float* label,float* grad_input,int N,int C){
    CUDAKERNELLOOP(i,N*C){
        int n = i / C;
        int c = i % C;

        int correct_class = static_cast<int>(label[n]);

        if(c == correct_class){
            grad_input[i] = soutput[i] - 1.0f;
        } else {
            grad_input[i] = soutput[i];
        }
    }
}

void HW3::softmaxsel_backward(const Tensor& soutput,const Tensor& label,Tensor& grad_sinput){
    int N = soutput.shape[0];
    int C = soutput.shape[1];
    kernel_smel_backward<<<CudaGetBlocks(N*C),kCudaThreadsNum>>>(soutput.get_ptr(),label.get_ptr(),grad_sinput.get_ptr(),N,C);
}
