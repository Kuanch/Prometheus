// Vector Addition CUDA Kernel
// Demonstrates memory-bound operation: 2 reads + 1 write per element

#include <torch/extension.h>
#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vectorized version using float4 for better memory throughput
__global__ void vector_add_kernel_vec4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n  // n is number of float4 elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = make_float4(
            va.x + vb.x,
            va.y + vb.y,
            va.z + vb.z,
            va.w + vb.w
        );
    }
}

// PyTorch wrapper for simple kernel
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have same shape");

    auto c = torch::empty_like(a);
    int n = a.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}

// PyTorch wrapper for vectorized kernel
torch::Tensor vector_add_vec4_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have same shape");
    TORCH_CHECK(a.numel() % 4 == 0, "size must be divisible by 4 for vec4");

    auto c = torch::empty_like(a);
    int n = a.numel() / 4;  // Number of float4 elements

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    vector_add_kernel_vec4<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(a.data_ptr<float>()),
        reinterpret_cast<const float4*>(b.data_ptr<float>()),
        reinterpret_cast<float4*>(c.data_ptr<float>()),
        n
    );

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Vector addition (CUDA)");
    m.def("vector_add_vec4", &vector_add_vec4_cuda, "Vector addition with float4 vectorization (CUDA)");
}
