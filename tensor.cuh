#pragma once
#ifndef TENSOR_CUH
#define TENSOR_CUH

#include<vector>
#include<iostream>
#include<memory>
#include<string>
#include<stdexcept>
#include<cuda_runtime.h>
#include<cstring>
#include<curand.h>

using FloatPtr = std::shared_ptr<float[]>;

enum class Device{
    cpu,
    gpu,
    None
};

class Tensor{
public:
    Tensor();
    Tensor(std::vector<int>_shape,Device _device);
    Tensor(const std::vector<int>& _shape, const std::vector<float>& data, Device _device);
 
    Tensor(const Tensor&);
    Tensor& operator=(const Tensor&);

    Tensor(Tensor&& other);
    Tensor& operator=(Tensor&& other);

    Tensor gpu();
    Tensor cpu();

    void random();

    void print(int cur_dim = 0,size_t flat_idx = 0);

    int N;
    std::vector<int>shape;
    Device device;

    static Tensor ones(const std::vector<int>& _shape, Device _device);
    static Tensor zeros(const std::vector<int>& _shape, Device _device);

    bool operator==(const Tensor& other) const;

    Tensor reshape(const std::vector<int>& new_shape) const;

    float* get_ptr() const;


private:
    size_t size;

    FloatPtr h;
    FloatPtr d;

    int stride{1};
    int offset{0};
};


#endif