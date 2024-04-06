#pragma once
#include <torch/extension.h>

void torch_launch_add( torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n);