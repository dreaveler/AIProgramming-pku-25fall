#include"tensor.cuh"
#include<random>

struct GpuDeleter {
    void operator()(float* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

Tensor::Tensor():h(),d(nullptr),shape(),N(0),size(0),device(Device::None),stride(0),offset(0){}

Tensor::Tensor(std::vector<int>_shape,Device _device):shape(_shape),device(_device),stride(1),offset(0){
    N = 1;
    for(auto dim : shape){
        if(dim<=0){
            throw std::runtime_error("shape输入不正确！需为正");
        }
        N = N * dim;
    }
    size = N * sizeof(float);
    if (device == Device::cpu){
        h = std::make_shared<float[]>(N);
        d = nullptr;
    }
    else if(device == Device::gpu){
        float* gpu_ptr;
        cudaMalloc(&gpu_ptr, N * sizeof(float));
        d = FloatPtr(gpu_ptr,GpuDeleter());
        h = nullptr;
    }else{
        throw std::runtime_error("device输入不正确！应为gpu or cpu");
    }
}

Tensor::Tensor(const std::vector<int>& _shape, const std::vector<float>& data, Device _device)
    : shape(_shape), device(_device), stride(1), offset(0)
{
    N = 1;
    for(auto dim : shape){
        if(dim <= 0){
            throw std::runtime_error("shape输入不正确！需为正");
        }
        N *= static_cast<size_t>(dim);
    }
    if (N != data.size()) {
        throw std::runtime_error("数据大小与shape不匹配！");
    }

    size = N * sizeof(float);

    if (device == Device::cpu){
        h = std::make_shared<float[]>(N);
        // 从 host vector 拷贝数据到 CPU 内存
        std::memcpy(h.get(), data.data(), size);
        d = nullptr;
    }
    else if(device == Device::gpu){
        float* gpu_ptr;
        cudaMalloc(&gpu_ptr, N * sizeof(float));
        cudaMemcpy(gpu_ptr, data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        // 使用自定义删除器创建 shared_ptr
        this->d = FloatPtr(gpu_ptr, GpuDeleter());
        this->h = nullptr; // GPU 模式下，CPU 指针为空
    }else{
        throw std::runtime_error("device输入不正确！应为gpu or cpu");
    }
}

Tensor::Tensor(const Tensor& other)
    : h(other.h), d(other.d), shape(other.shape),
        size(other.size), device(other.device),N(other.N){}

Tensor& Tensor::operator=(const Tensor& other){
    if (this != &other){
        h = other.h;
        d = other.d;
        shape = other.shape;
        size = other.size;
        device = other.device;
        N = other.N;
    }
    return *this;
}

Tensor::Tensor(Tensor&& other)
    : h(std::move(other.h)), d(std::move(other.d)), shape(std::move(other.shape)),
        size(other.size), device(other.device),N(other.N){
    other.N= 0;
    other.size = 0;
    other.device = Device::None;
}

Tensor& Tensor::operator=(Tensor&& other){
    if (this != &other){
        h = std::move(other.h);
        d = std::move(other.d);
        shape = std::move(other.shape);
        size = other.size;
        device = other.device;
        N = other.N;
        other.N = 0;
        other.size = 0;
        other.device = Device::None;
    }
    return *this;
}

//cpu和gpu的实现形式相仿，如果本身在cpu上要转移到cpu则移动语义移动 转移到另一device上则正常实现
Tensor Tensor::gpu(){
    if (device == Device::cpu){
        Tensor out(shape,Device::gpu);
        cudaMemcpy(out.d.get(), h.get(), size, cudaMemcpyHostToDevice);
        return out;
    } else if (device == Device::gpu){
        return *this;
    }
}

Tensor Tensor::cpu(){
    if (device == Device::cpu){
        return *this;
    } else if (device == Device::gpu){
        Tensor out(shape, Device::cpu);
        cudaMemcpy(out.h.get(), d.get(), size, cudaMemcpyDeviceToHost);
        return out;
    }
}

void Tensor::random(){
    if(device==Device::gpu){
        curandGenerator_t prng;
        curandCreateGenerator(&prng,CURAND_RNG_QUASI_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(prng,(unsigned long long)clock());
        curandGenerateUniform(prng,get_ptr(),N);
        curandDestroyGenerator(prng);
    }else if(device == Device::cpu){
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float* ptr = h.get();
        for (size_t i = 0; i < N; ++i) {
            ptr[i] = dis(gen);
        }
    }
}

void Tensor::print(int cur_dim,size_t flat_idx){
    FloatPtr t;
    if(device == Device::cpu) t = h;
    else if(device == Device::gpu){
        t = std::make_shared<float[]>(N);
        cudaMemcpy(t.get(),d.get(),size,cudaMemcpyDeviceToHost);
    }
    if(cur_dim == shape.size()-1){
        std::cout<<"[";
        for(auto i = 0;i < shape[cur_dim];i++){
            std::cout<<t[flat_idx + i];
            if(i < shape[cur_dim] - 1) std::cout<<",";
        }
        std::cout<<"]";
    }else{
        std::cout<<"[\n";
        size_t stride = 1;
        for (auto dim = cur_dim + 1;dim < shape.size();dim++){
            stride *= shape[dim];
        }
        for(auto i = 0;i<shape[cur_dim];i++){
            print(cur_dim+1,flat_idx + i*stride);
            if(i<shape[cur_dim] - 1) std::cout<<",\n";
            else std::cout<<"\n";
        }
        std::cout<<']';
    }
}

__global__ void fill_kernel(float* data, float value, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = value;
    }
}

Tensor Tensor::ones(const std::vector<int>& _shape, Device _device) {
    Tensor t(_shape, _device);
    if (_device == Device::cpu) {
        float* ptr = t.h.get();
        for (int i = 0; i < t.N; ++i) {
            ptr[i] = 1.0f;
        }
    } else if (_device == Device::gpu) {
    const int threads_per_block = 256;
    const int blocks = (t.N + threads_per_block - 1) / threads_per_block;
    fill_kernel<<<blocks, threads_per_block>>>(t.d.get(), 1.0f, t.N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::ones fill_kernel 启动失败: ") + cudaGetErrorString(err));
    }
    }

    return t;
}

Tensor Tensor::zeros(const std::vector<int>& _shape, Device _device) {
    Tensor t(_shape, _device);
    if (_device == Device::cpu) {
        float* ptr = t.h.get();
        for (int i = 0; i < t.N; ++i) {
            ptr[i] = 1.0f;
        }
    } else if (_device == Device::gpu) {
    const int threads_per_block = 256;
    const int blocks = (t.N + threads_per_block - 1) / threads_per_block;
    fill_kernel<<<blocks, threads_per_block>>>(t.d.get(), 0.0f, t.N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Tensor::zeros fill_kernel 启动失败: ") + cudaGetErrorString(err));
    }
    }

    return t;
}

bool Tensor::operator==(const Tensor& other) const {
    // 1. 检查 Shape 是否匹配
    if (this->shape != other.shape) {
        std::cout << "比较失败: Shape不匹配!" << std::endl;
        return false;
    }

    // 如果两个张量都为空，则它们相等
    if (this->N == 0 && other.N == 0) {
        return true;
    }

    // 2. 将张量数据准备到 CPU 内存中
    FloatPtr this_h;
    if (this->device == Device::cpu) {
        this_h = this->h;
    } else { // 如果在 GPU 上，拷贝到临时的 CPU 内存
        this_h = std::make_shared<float[]>(this->N);
        cudaMemcpy(this_h.get(), d.get(), size, cudaMemcpyDeviceToHost);
    }

    FloatPtr other_h;
    if (other.device == Device::cpu) {
        other_h = other.h;
    } else { // 如果在 GPU 上，拷贝到临时的 CPU 内存
        other_h = std::make_shared<float[]>(other.N);
        cudaMemcpy(other_h.get(), other.d.get(), other.size, cudaMemcpyDeviceToHost);
    }

    // 3. 逐元素比较
    const float tolerance = 1e-5f;
    const float* p1 = this_h.get();
    const float* p2 = other_h.get();

    for (size_t i = 0; i < this->N; ++i) {
        if (std::abs(p1[i] - p2[i]) > tolerance) {
            std::cout << "比较失败于索引 " << i << ": " << p1[i] << " vs " << p2[i] << std::endl;
            return false;
        }
    }
    return true;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    long long new_N = 1;
    for (long long dim : new_shape) new_N *= dim;
    if (new_N != this->N) {
        throw std::runtime_error("Tensor::reshape: total number of elements must remain the same.");
    }

    Tensor reshaped_tensor;
    reshaped_tensor.shape = new_shape;
    reshaped_tensor.N = new_N;
    reshaped_tensor.device = this->device;

    reshaped_tensor.d = this->d;
    reshaped_tensor.h = this->h;

    return reshaped_tensor;
}

//需要保证只读和临时引用以及不存储这个指针  否则会造成悬垂指针
float* Tensor::get_ptr() const{
    if(device == Device::gpu){
        return d.get();
    }else if(device == Device::cpu){
        return h.get();
    }else throw std::runtime_error("向未赋值的tensor获取ptr");
}