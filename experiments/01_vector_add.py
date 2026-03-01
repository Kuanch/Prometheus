#!/usr/bin/env python3
"""
Experiment 1: Vector Addition - Memory-Bound Operation

This experiment demonstrates:
1. Vector addition is memory-bound (limited by memory bandwidth, not compute)
2. PyTorch overhead for simple operations
3. How to calculate and interpret achieved bandwidth

Memory access pattern:
- Read A[i], Read B[i], Write C[i] = 3 memory operations per element
- Total bytes = 3 * N * sizeof(float) = 12 * N bytes

Expected insight: PyTorch ≈ Custom CUDA because both hit the same memory bandwidth limit
"""

import torch
import sys
sys.path.insert(0, '/workspace')

from utils.profiler import GPUProfiler, TimingResult, print_comparison_table, get_gpu_info

# Try to import custom CUDA kernels
try:
    from cuda_kernels import vector_add
    CUSTOM_CUDA_AVAILABLE = True
except ImportError:
    CUSTOM_CUDA_AVAILABLE = False
    print("Custom CUDA kernels not available. Run: cd kernels && python setup.py install")


def main():
    print("\n" + "="*70)
    print(" EXPERIMENT 1: Vector Addition (Memory-Bound Operation)")
    print("="*70)

    get_gpu_info()

    # Test sizes (powers of 2)
    sizes = [1 << 20, 1 << 22, 1 << 24, 1 << 26]  # 1M, 4M, 16M, 64M elements

    profiler = GPUProfiler(warmup_iters=10, measure_iters=100)

    for n in sizes:
        print(f"\n{'─'*70}")
        print(f"Vector size: {n:,} elements ({n * 4 / 1e6:.1f} MB per vector)")
        print(f"Total memory accessed: {3 * n * 4 / 1e6:.1f} MB (read A, read B, write C)")
        print(f"{'─'*70}")

        # Create test data on GPU
        a = torch.randn(n, device='cuda', dtype=torch.float32)
        b = torch.randn(n, device='cuda', dtype=torch.float32)

        # Bytes accessed: 2 reads + 1 write = 3 * n * 4 bytes
        bytes_accessed = 3 * n * 4

        results = []

        # 1. PyTorch operator
        def pytorch_add():
            return a + b

        r = profiler.benchmark_memory_bound(
            "PyTorch (a + b)",
            pytorch_add,
            bytes_accessed=bytes_accessed
        )
        results.append(r)

        # 2. PyTorch torch.add
        def pytorch_add_fn():
            return torch.add(a, b)

        r = profiler.benchmark_memory_bound(
            "torch.add()",
            pytorch_add_fn,
            bytes_accessed=bytes_accessed
        )
        results.append(r)

        # 3. PyTorch with pre-allocated output
        c = torch.empty_like(a)
        def pytorch_add_out():
            return torch.add(a, b, out=c)

        r = profiler.benchmark_memory_bound(
            "torch.add(out=c)",
            pytorch_add_out,
            bytes_accessed=bytes_accessed
        )
        results.append(r)

        # 4. Custom CUDA kernel
        if CUSTOM_CUDA_AVAILABLE:
            def cuda_add():
                return vector_add.vector_add(a, b)

            r = profiler.benchmark_memory_bound(
                "Custom CUDA",
                cuda_add,
                bytes_accessed=bytes_accessed
            )
            results.append(r)

            # 5. Custom CUDA kernel with vectorization
            def cuda_add_vec4():
                return vector_add.vector_add_vec4(a, b)

            r = profiler.benchmark_memory_bound(
                "Custom CUDA (vec4)",
                cuda_add_vec4,
                bytes_accessed=bytes_accessed
            )
            results.append(r)

        print_comparison_table(results, f"Vector Add (N={n:,})", show_bandwidth=True)

        # Verify correctness
        if CUSTOM_CUDA_AVAILABLE:
            ref = a + b
            cuda_result = vector_add.vector_add(a, b)
            if torch.allclose(ref, cuda_result):
                print("✓ Results verified: PyTorch and Custom CUDA match")
            else:
                print("✗ ERROR: Results don't match!")

    # Analysis
    print("\n" + "="*70)
    print(" ANALYSIS")
    print("="*70)
    print("""
    Key observations:

    1. MEMORY-BOUND BEHAVIOR:
       - Vector addition does very little compute (1 add per 3 memory ops)
       - Performance is entirely limited by memory bandwidth (~256 GB/s)
       - All implementations should achieve similar bandwidth

    2. PYTORCH OVERHEAD:
       - For large vectors, PyTorch overhead is negligible
       - The "+ operator" and torch.add() are equivalent
       - Pre-allocated output (out=) can reduce allocation overhead

    3. VECTORIZATION:
       - float4 loads can improve bandwidth utilization
       - Modern GPUs handle this automatically for aligned access

    4. ROOFLINE MODEL:
       - Arithmetic Intensity (AI) = FLOPs / Bytes = N / (12*N) = 0.083 FLOPs/byte
       - This is far below the ridge point, confirming memory-bound behavior
    """)


if __name__ == "__main__":
    main()
