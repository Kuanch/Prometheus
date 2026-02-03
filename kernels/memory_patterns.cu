// Memory Access Pattern Kernels
// Demonstrates the critical importance of memory coalescing on GPUs

#include <torch/extension.h>
#include <cuda_runtime.h>

// Coalesced memory access: consecutive threads access consecutive memory
// This is the optimal pattern - GPU can issue one wide memory transaction
__global__ void copy_coalesced_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Strided memory access: threads access memory with a stride
// This is inefficient - each warp needs multiple memory transactions
__global__ void copy_strided_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n,
    int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = (idx * stride) % n;
    if (idx < n) {
        dst[idx] = src[strided_idx];
    }
}

// Column-major access pattern for 2D array (non-coalesced)
// When matrix is stored row-major but accessed column-wise
__global__ void copy_column_major_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        // Row-major storage, but column-major access pattern
        // Threads in same warp access elements cols apart
        int src_idx = col * rows + row;  // Column-major read
        int dst_idx = row * cols + col;  // Row-major write
        dst[dst_idx] = src[src_idx];
    }
}

// Row-major access pattern for 2D array (coalesced)
__global__ void copy_row_major_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        // Row-major storage with row-major access pattern
        // Threads in same warp access consecutive elements
        int idx = row * cols + col;
        dst[idx] = src[idx];
    }
}

// Misaligned access - start from non-aligned offset
__global__ void copy_misaligned_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n,
    int offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - offset) {
        dst[idx] = src[idx + offset];
    }
}

// Bank conflict demonstration using shared memory
// 32 banks, consecutive 4-byte words go to consecutive banks
__global__ void shared_mem_no_conflict_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    __shared__ float smem[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        // Load: threads access consecutive elements (no bank conflicts)
        smem[tid] = src[idx];
        __syncthreads();

        // Store: threads access consecutive elements (no bank conflicts)
        dst[idx] = smem[tid];
    }
}

__global__ void shared_mem_bank_conflict_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    __shared__ float smem[256 * 32];  // Much larger to allow strided access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        // Load with stride of 32 causes maximum bank conflicts
        // All threads in a warp access the same bank
        smem[tid * 32] = src[idx];
        __syncthreads();

        dst[idx] = smem[tid * 32];
    }
}

// PyTorch wrappers
torch::Tensor copy_coalesced_cuda(torch::Tensor src) {
    TORCH_CHECK(src.device().is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");

    auto dst = torch::empty_like(src);
    int n = src.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    copy_coalesced_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        n
    );

    return dst;
}

torch::Tensor copy_strided_cuda(torch::Tensor src, int stride) {
    TORCH_CHECK(src.device().is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");

    auto dst = torch::empty_like(src);
    int n = src.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    copy_strided_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        n,
        stride
    );

    return dst;
}

torch::Tensor copy_column_major_cuda(torch::Tensor src) {
    TORCH_CHECK(src.device().is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(src.dim() == 2, "src must be 2D");

    int rows = src.size(0);
    int cols = src.size(1);
    auto dst = torch::empty_like(src);

    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);

    copy_column_major_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        rows, cols
    );

    return dst;
}

torch::Tensor copy_row_major_cuda(torch::Tensor src) {
    TORCH_CHECK(src.device().is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(src.dim() == 2, "src must be 2D");

    int rows = src.size(0);
    int cols = src.size(1);
    auto dst = torch::empty_like(src);

    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);

    copy_row_major_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        rows, cols
    );

    return dst;
}

torch::Tensor copy_misaligned_cuda(torch::Tensor src, int offset) {
    TORCH_CHECK(src.device().is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(offset >= 0 && offset < src.numel(), "Invalid offset");

    int n = src.numel();
    auto dst = torch::empty({n - offset}, src.options());

    const int threads = 256;
    const int blocks = (n - offset + threads - 1) / threads;

    copy_misaligned_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        n,
        offset
    );

    return dst;
}

torch::Tensor shared_no_conflict_cuda(torch::Tensor src) {
    TORCH_CHECK(src.device().is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");

    auto dst = torch::empty_like(src);
    int n = src.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    shared_mem_no_conflict_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        n
    );

    return dst;
}

torch::Tensor shared_bank_conflict_cuda(torch::Tensor src) {
    TORCH_CHECK(src.device().is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");

    auto dst = torch::empty_like(src);
    int n = src.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    shared_mem_bank_conflict_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        n
    );

    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_coalesced", &copy_coalesced_cuda, "Coalesced memory copy (CUDA)");
    m.def("copy_strided", &copy_strided_cuda, "Strided memory copy (CUDA)");
    m.def("copy_column_major", &copy_column_major_cuda, "Column-major 2D copy (CUDA)");
    m.def("copy_row_major", &copy_row_major_cuda, "Row-major 2D copy (CUDA)");
    m.def("copy_misaligned", &copy_misaligned_cuda, "Misaligned memory copy (CUDA)");
    m.def("shared_no_conflict", &shared_no_conflict_cuda, "Shared memory without bank conflicts (CUDA)");
    m.def("shared_bank_conflict", &shared_bank_conflict_cuda, "Shared memory with bank conflicts (CUDA)");
}
