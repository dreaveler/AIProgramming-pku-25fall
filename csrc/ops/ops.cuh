#pragma once
#ifndef OPS_CUH
#define OPS_CUH

#include "../tensor.cuh"
#include <vector>

namespace Ops {

Tensor add(const Tensor& a, const Tensor& b);
Tensor add_scalar(const Tensor& a, float scalar);

Tensor multiply(const Tensor& a, const Tensor& b);
Tensor mul_scalar(const Tensor& a, float scalar);

Tensor divide(const Tensor& a, const Tensor& b);
Tensor div_scalar(const Tensor& a, float scalar);

Tensor power(const Tensor& a, const Tensor& b);
Tensor power_scalar(const Tensor& a, float scalar);

Tensor negate(const Tensor& a);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);

Tensor reshape(const Tensor& a, const std::vector<int>& new_shape);
Tensor transpose(const Tensor& a, const std::vector<int>& axes);
Tensor broadcast_to(const Tensor& a, const std::vector<int>& shape);
Tensor summation(const Tensor& a, const std::vector<int>& axes);

Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace Ops

#endif
