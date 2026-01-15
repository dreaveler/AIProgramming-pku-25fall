#pragma once
#ifndef ACTIVATION_CUH
#define ACTIVATION_CUH

#include "../tensor.cuh"

class Function {
public:
    virtual Tensor forward(const Tensor&) = 0;
    virtual Tensor backward(const Tensor&) = 0;
    virtual ~Function() = default;
protected:
    Tensor saved;
};

class Relu : public Function {
public:
    Relu() = default;
    Tensor forward(const Tensor&);
    Tensor backward(const Tensor&);
    ~Relu() = default;
};

class Sigmoid : public Function {
public:
    Sigmoid() = default;
    Tensor forward(const Tensor&);
    Tensor backward(const Tensor&);
    ~Sigmoid() = default;
};

#endif
