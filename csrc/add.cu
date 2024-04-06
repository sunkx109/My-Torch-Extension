#include <torch/extension.h>

__global__ void add_kernel( float* c,
                            const float* a,
                            const float* b,
                            int n) 
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < n; i += gridDim.x * blockDim.x) 
    {
        c[i] = a[i] + b[i];
    }
}

void launch_add(float* c,
                 const float* a,
                 const float* b,
                 int n)
{
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    add_kernel<<<grid, block>>>(c, a, b, n);
}

void torch_launch_add(torch::Tensor &c,
                      const torch::Tensor &a,
                      const torch::Tensor &b,
                      int64_t n) 
{
    launch_add((float *)c.data_ptr(),
    (const float *)a.data_ptr(),
    (const float *)b.data_ptr(),
    n);
}