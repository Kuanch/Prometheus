#!/usr/bin/env python3
"""
Experiment 2: ReLU - Element-wise Memory-Bound Operation

This experiment demonstrates:
1. ReLU is memory-bound (even simpler than vector add)
2. Different ReLU implementations have similar performance
3. In-place operations save memory bandwidth

Memory access pattern:
- Read X[i], Write Y[i] = 2 memory operations per element
- Total bytes = 2 * N * sizeof(float) = 8 * N bytes
- In-place: Read X[i], Write X[i] = still 2 ops, but same location

Expected insight: All ReLU implementations hit the same bandwidth limit
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace')

from utils.profiler import GPUProfiler, print_comparison_table, get_gpu_info

try:
    from cuda_kernels import relu
    CUSTOM_CUDA_AVAILABLE = True
except ImportError:
    CUSTOM_CUDA_AVAILABLE = False
    print("Custom CUDA kernels not available. Run: cd kernels && python setup.py install")


def main():
    print("\n" + "="*70)
    print(" EXPERIMENT 2: ReLU (Element-wise Memory-Bound Operation)")
    print("="*70)

    get_gpu_info()

    sizes = [1 << 22, 1 << 24, 1 << 26]  # 4M, 16M, 64M elements

    profiler = GPUProfiler(warmup_iters=10, measure_iters=100)

    for n in sizes:
        print(f"\n{'─'*70}")
        print(f"Vector size: {n:,} elements ({n * 4 / 1e6:.1f} MB)")
        print(f"Total memory accessed: {2 * n * 4 / 1e6:.1f} MB (read X, write Y)")
        print(f"{'─'*70}")

        # Create test data
        x = torch.randn(n, device='cuda', dtype=torch.float32)

        # Bytes accessed: 1 read + 1 write = 2 * n * 4 bytes
        bytes_accessed = 2 * n * 4

        results = []

        # 1. PyTorch F.relu
        def pytorch_relu():
            return F.relu(x)

        r = profiler.benchmark_memory_bound(
            "F.relu()",
            pytorch_relu,
            bytes_accessed=bytes_accessed
        )
        results.append(r)

        # 2. PyTorch torch.relu
        def pytorch_torch_relu():
            return torch.relu(x)

        r = profiler.benchmark_memory_bound(
            "torch.relu()",
            pytorch_torch_relu,
            bytes_accessed=bytes_accessed
        )
        results.append(r)

        # 3. PyTorch clamp
        def pytorch_clamp():
            return torch.clamp(x, min=0)

        r = profiler.benchmark_memory_bound(
            "torch.clamp(min=0)",
            pytorch_clamp,
            bytes_accessed=bytes_accessed
        )
        results.append(r)

        # 4. PyTorch maximum
        zero = torch.zeros(1, device='cuda')
        def pytorch_maximum():
            return torch.maximum(x, zero)

        r = profiler.benchmark_memory_bound(
            "torch.maximum(x, 0)",
            pytorch_maximum,
            bytes_accessed=bytes_accessed
        )
        results.append(r)

        # 5. PyTorch in-place
        x_copy = x.clone()
        def pytorch_relu_inplace():
            nonlocal x_copy
            x_copy = x.clone()  # Need to reset
            return F.relu(x_copy, inplace=True)

        # For in-place, measure separately since we need to clone each time
        # In-place still accesses same bytes but no allocation
        r = profiler.benchmark_memory_bound(
            "F.relu(inplace=True)*",
            pytorch_relu_inplace,
            bytes_accessed=bytes_accessed
        )
        results.append(r)

        # Custom CUDA kernels
        if CUSTOM_CUDA_AVAILABLE:
            def cuda_relu():
                return relu.relu(x)

            r = profiler.benchmark_memory_bound(
                "Custom CUDA",
                cuda_relu,
                bytes_accessed=bytes_accessed
            )
            results.append(r)

            def cuda_relu_branchless():
                return relu.relu_branchless(x)

            r = profiler.benchmark_memory_bound(
                "CUDA (branchless)",
                cuda_relu_branchless,
                bytes_accessed=bytes_accessed
            )
            results.append(r)

            def cuda_relu_vec4():
                return relu.relu_vec4(x)

            r = profiler.benchmark_memory_bound(
                "CUDA (vec4)",
                cuda_relu_vec4,
                bytes_accessed=bytes_accessed
            )
            results.append(r)

        print_comparison_table(results, f"ReLU (N={n:,})", show_bandwidth=True)

        # Verify correctness
        if CUSTOM_CUDA_AVAILABLE:
            ref = F.relu(x)
            cuda_result = relu.relu(x)
            if torch.allclose(ref, cuda_result):
                print("✓ Results verified: PyTorch and Custom CUDA match")
            else:
                max_diff = (ref - cuda_result).abs().max()
                print(f"✗ Max difference: {max_diff}")

    # Analysis
    print("\n" + "="*70)
    print(" ANALYSIS")
    print("="*70)
    print("""
    Key observations:

    1. ALL IMPLEMENTATIONS ARE EQUIVALENT:
       - F.relu(), torch.relu(), torch.clamp() all hit same bandwidth
       - The compute (max operation) is trivial compared to memory access
       - Framework overhead is negligible for large tensors

    2. BRANCHING VS BRANCHLESS:
       - Modern GPUs handle simple branches efficiently
       - Branchless version shows no significant difference
       - Branch divergence matters more for complex conditionals

    3. IN-PLACE OPERATIONS:
       - In-place saves memory allocation but same bandwidth
       - Useful for reducing memory footprint, not speed
       - *Note: Our benchmark includes clone() overhead

    4. VECTORIZATION:
       - float4 can help but L2 cache effects dominate
       - GPU memory controller handles coalescing automatically

    5. COMPARISON TO VECTOR ADD:
       - ReLU: 8 bytes/element (1 read + 1 write)
       - Vector Add: 12 bytes/element (2 reads + 1 write)
       - ReLU should show ~50% HIGHER bandwidth numbers
         (because bandwidth = bytes / time, and ReLU moves fewer bytes)

    * The inplace benchmark includes clone() to reset data each iteration,
      so it appears slower. True in-place would match other implementations.
    """)


if __name__ == "__main__":
    main()
