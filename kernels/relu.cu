// ReLU CUDA Kernel
// Demonstrates element-wise memory-bound operation: 1 read + 1 write per element

#include <torch/extension.h>
#include <cuda_runtime.h>

// Simple ReLU kernel
__global__ void relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

// Branchless ReLU using bit manipulation (for comparison)
__global__ void relu_branchless_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        // Use intrinsic: returns val if val > 0, else 0
        y[idx] = val * (val > 0.0f);
    }
}

// In-place ReLU kernel
__global__ void relu_inplace_kernel(
    float* __restrict__ x,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

// Vectorized ReLU using float4
__global__ void relu_vec4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 val = x[idx];
        y[idx] = make_float4(
            fmaxf(0.0f, val.x),
            fmaxf(0.0f, val.y),
            fmaxf(0.0f, val.z),
            fmaxf(0.0f, val.w)
        );
    }
}

// PyTorch wrappers
torch::Tensor relu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

torch::Tensor relu_branchless_cuda(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    relu_branchless_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

torch::Tensor relu_inplace_cuda(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    relu_inplace_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        n
    );

    return x;
}

torch::Tensor relu_vec4_cuda(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.numel() % 4 == 0, "size must be divisible by 4 for vec4");

    auto y = torch::empty_like(x);
    int n = x.numel() / 4;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    relu_vec4_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", &relu_cuda, "ReLU (CUDA)");
    m.def("relu_branchless", &relu_branchless_cuda, "ReLU branchless (CUDA)");
    m.def("relu_inplace", &relu_inplace_cuda, "ReLU in-place (CUDA)");
    m.def("relu_vec4", &relu_vec4_cuda, "ReLU with float4 vectorization (CUDA)");
}
