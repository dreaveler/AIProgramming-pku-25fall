#pragma once
#ifndef UTILS_CUH
#define UTILS_CUH

#include"tensor.cuh"
#include"cublas_v2.h"
#include"function.cuh"

void flatten_kernel(const Tensor& kernel4d, Tensor& out2d);

void reshape_col_to_image(const Tensor& col, Tensor& image);

void pack_NCHW_rows(const Tensor& y4d, Tensor& y2d);
void unflatten_KO_to_OIHW(const Tensor& K, Tensor& kernel4d);

#endif