#!/usr/bin/env python3
"""
Experiment 3: Matrix Multiplication - Compute-Bound Operation

This experiment demonstrates:
1. Matrix multiplication is compute-bound (limited by FLOPS, not bandwidth)
2. The massive performance gap between naive and optimized implementations
3. Why cuBLAS (used by PyTorch) is so much faster than naive CUDA

Compute complexity:
- C[M,N] = A[M,K] @ B[K,N]
- FLOPs = 2 * M * N * K (multiply + add for each output element)
- Memory = (M*K + K*N + M*N) * sizeof(float)

Arithmetic Intensity:
- AI = FLOPs / Bytes = 2*M*N*K / (4*(M*K + K*N + M*N))
- For square matrices: AI ≈ N/6 FLOPs/byte
- For N=1024: AI ≈ 170 FLOPs/byte (highly compute-bound!)

Expected insight: cuBLAS >> Tiled CUDA >> Naive CUDA (10-100x differences)
"""

import torch
import sys
sys.path.insert(0, '/workspace')

from utils.profiler import GPUProfiler, print_comparison_table, get_gpu_info

try:
    from cuda_kernels import matmul
    CUSTOM_CUDA_AVAILABLE = True
except ImportError:
    CUSTOM_CUDA_AVAILABLE = False
    print("Custom CUDA kernels not available. Run: cd kernels && python setup.py install")


def main():
    print("\n" + "="*70)
    print(" EXPERIMENT 3: Matrix Multiplication (Compute-Bound Operation)")
    print("="*70)

    get_gpu_info()

    # Square matrix sizes
    sizes = [256, 512, 1024, 2048]

    profiler = GPUProfiler(warmup_iters=5, measure_iters=20)

    for n in sizes:
        M, K, N = n, n, n

        # Calculate metrics
        num_flops = 2 * M * N * K
        bytes_accessed = (M * K + K * N + M * N) * 4
        arithmetic_intensity = num_flops / bytes_accessed

        print(f"\n{'─'*70}")
        print(f"Matrix size: {M}x{K} @ {K}x{N}")
        print(f"FLOPs: {num_flops / 1e9:.2f} GFLOPs")
        print(f"Memory: {bytes_accessed / 1e6:.1f} MB")
        print(f"Arithmetic Intensity: {arithmetic_intensity:.1f} FLOPs/byte")
        print(f"{'─'*70}")

        # Create test matrices
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)

        results = []

        # 1. PyTorch matmul (uses cuBLAS)
        def pytorch_matmul():
            return torch.matmul(A, B)

        r = profiler.benchmark_compute_bound(
            "PyTorch (cuBLAS)",
            pytorch_matmul,
            num_flops=num_flops
        )
        results.append(r)

        # 2. PyTorch mm
        def pytorch_mm():
            return torch.mm(A, B)

        r = profiler.benchmark_compute_bound(
            "torch.mm()",
            pytorch_mm,
            num_flops=num_flops
        )
        results.append(r)

        # 3. PyTorch @ operator
        def pytorch_at():
            return A @ B

        r = profiler.benchmark_compute_bound(
            "A @ B",
            pytorch_at,
            num_flops=num_flops
        )
        results.append(r)

        # Custom CUDA kernels
        if CUSTOM_CUDA_AVAILABLE:
            # 4. Naive CUDA (only for small sizes - very slow!)
            if n <= 1024:
                def cuda_naive():
                    return matmul.matmul_naive(A, B)

                r = profiler.benchmark_compute_bound(
                    "CUDA Naive",
                    cuda_naive,
                    num_flops=num_flops
                )
                results.append(r)

            # 5. Tiled CUDA
            def cuda_tiled():
                return matmul.matmul_tiled(A, B)

            r = profiler.benchmark_compute_bound(
                "CUDA Tiled (shared mem)",
                cuda_tiled,
                num_flops=num_flops
            )
            results.append(r)

            # 6. Tiled with double buffering
            def cuda_tiled_db():
                return matmul.matmul_tiled_doublebuf(A, B)

            r = profiler.benchmark_compute_bound(
                "CUDA Tiled (double buf)",
                cuda_tiled_db,
                num_flops=num_flops
            )
            results.append(r)

        print_comparison_table(
            results,
            f"Matrix Multiply ({n}x{n})",
            show_bandwidth=False,
            show_flops=True
        )

        # Verify correctness
        if CUSTOM_CUDA_AVAILABLE:
            ref = torch.matmul(A, B)
            cuda_result = matmul.matmul_tiled(A, B)
            if torch.allclose(ref, cuda_result, rtol=1e-3, atol=1e-3):
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

    1. COMPUTE-BOUND BEHAVIOR:
       - Arithmetic Intensity grows with matrix size (N/6 for square)
       - Performance is limited by FP32 throughput (~15 TFLOPS)
       - Memory bandwidth is NOT the bottleneck

    2. NAIVE VS OPTIMIZED:
       - Naive: O(K) global memory reads per output element
       - Tiled: Shared memory reduces global reads by TILE_SIZE factor
       - cuBLAS: Highly optimized with register tiling, vectorization, etc.

    3. PERFORMANCE HIERARCHY:
       - cuBLAS achieves ~80-90% of peak TFLOPS
       - Our tiled kernel achieves ~10-30% of peak
       - Naive kernel is 10-100x slower

    4. WHY cuBLAS IS FAST:
       - Register-level tiling (thread computes multiple outputs)
       - Vectorized memory access (float4, etc.)
       - Warp-level primitives (tensor cores on newer GPUs)
       - Autotuning for specific GPU architectures
       - Years of optimization by NVIDIA engineers

    5. LESSON FOR DEEP LEARNING:
       - Always use cuBLAS/cuDNN for matrix operations
       - Writing efficient GEMM is extremely difficult
       - PyTorch's "overhead" is negligible for large matmuls
       - Focus optimization efforts elsewhere

    6. RTX 4060 NOTES:
       - Has Tensor Cores but we're using FP32 (not TF32/FP16)
       - With Tensor Cores + FP16: could reach 100+ TFLOPS
    """)


if __name__ == "__main__":
    main()
