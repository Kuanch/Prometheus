#!/usr/bin/env python3
"""
Experiment 4: Memory Access Patterns

This experiment demonstrates:
1. Memory coalescing is CRITICAL for GPU performance
2. Strided access can be 10-30x slower than coalesced
3. Bank conflicts in shared memory
4. Why data layout matters

Memory Coalescing:
- GPU memory is accessed in large transactions (32/64/128 bytes)
- When threads in a warp access consecutive addresses, ONE transaction suffices
- When threads access scattered addresses, MULTIPLE transactions needed
- Worst case: 32 transactions for 32 threads (vs 1 for coalesced)

Bank Conflicts:
- Shared memory has 32 banks (4-byte stride)
- Threads accessing same bank are serialized
- Stride of 32 elements = all threads hit same bank = 32x slowdown
"""

import torch
import sys
sys.path.insert(0, '/workspace')

from utils.profiler import GPUProfiler, print_comparison_table, get_gpu_info

try:
    from cuda_kernels import memory_patterns
    CUSTOM_CUDA_AVAILABLE = True
except ImportError:
    CUSTOM_CUDA_AVAILABLE = False
    print("Custom CUDA kernels not available. Run: cd kernels && python setup.py install")


def main():
    print("\n" + "="*70)
    print(" EXPERIMENT 4: Memory Access Patterns")
    print("="*70)

    get_gpu_info()

    if not CUSTOM_CUDA_AVAILABLE:
        print("\nThis experiment requires custom CUDA kernels.")
        print("Run: cd kernels && python setup.py install")
        return

    profiler = GPUProfiler(warmup_iters=10, measure_iters=100)

    # Part 1: Coalesced vs Strided Access
    print("\n" + "="*70)
    print(" PART 1: Coalesced vs Strided Memory Access")
    print("="*70)

    n = 1 << 24  # 16M elements
    bytes_accessed = 2 * n * 4  # read + write

    print(f"\nVector size: {n:,} elements ({n * 4 / 1e6:.1f} MB)")

    x = torch.randn(n, device='cuda', dtype=torch.float32)

    results = []

    # Coalesced (optimal)
    def coalesced():
        return memory_patterns.copy_coalesced(x)

    r = profiler.benchmark_memory_bound(
        "Coalesced (stride=1)",
        coalesced,
        bytes_accessed=bytes_accessed
    )
    results.append(r)

    # Various strides
    for stride in [2, 4, 8, 16, 32]:
        def strided(s=stride):
            return memory_patterns.copy_strided(x, s)

        r = profiler.benchmark_memory_bound(
            f"Strided (stride={stride})",
            strided,
            bytes_accessed=bytes_accessed
        )
        results.append(r)

    print_comparison_table(results, "Coalesced vs Strided Access", show_bandwidth=True)

    # Part 2: Row-major vs Column-major for 2D arrays
    print("\n" + "="*70)
    print(" PART 2: Row-major vs Column-major Access (2D)")
    print("="*70)

    rows, cols = 4096, 4096
    bytes_2d = 2 * rows * cols * 4

    print(f"\nMatrix size: {rows}x{cols} ({rows * cols * 4 / 1e6:.1f} MB)")

    x2d = torch.randn(rows, cols, device='cuda', dtype=torch.float32)

    results = []

    # Row-major (coalesced for row-major storage)
    def row_major():
        return memory_patterns.copy_row_major(x2d)

    r = profiler.benchmark_memory_bound(
        "Row-major access",
        row_major,
        bytes_accessed=bytes_2d
    )
    results.append(r)

    # Column-major (non-coalesced for row-major storage)
    def column_major():
        return memory_patterns.copy_column_major(x2d)

    r = profiler.benchmark_memory_bound(
        "Column-major access",
        column_major,
        bytes_accessed=bytes_2d
    )
    results.append(r)

    print_comparison_table(results, "Row vs Column Access", show_bandwidth=True)

    # Part 3: Misaligned access
    print("\n" + "="*70)
    print(" PART 3: Aligned vs Misaligned Access")
    print("="*70)

    n = 1 << 24
    x = torch.randn(n, device='cuda', dtype=torch.float32)

    results = []

    # Aligned (offset=0)
    def aligned():
        return memory_patterns.copy_coalesced(x)

    r = profiler.benchmark_memory_bound(
        "Aligned (offset=0)",
        aligned,
        bytes_accessed=2 * n * 4
    )
    results.append(r)

    # Various misalignments
    for offset in [1, 2, 3, 4, 7, 15]:
        def misaligned(o=offset):
            return memory_patterns.copy_misaligned(x, o)

        r = profiler.benchmark_memory_bound(
            f"Misaligned (offset={offset})",
            misaligned,
            bytes_accessed=2 * (n - offset) * 4
        )
        results.append(r)

    print_comparison_table(results, "Alignment Effects", show_bandwidth=True)

    # Part 4: Shared memory bank conflicts
    print("\n" + "="*70)
    print(" PART 4: Shared Memory Bank Conflicts")
    print("="*70)

    n = 1 << 20  # Smaller size for shared memory test
    bytes_accessed = 2 * n * 4

    x = torch.randn(n, device='cuda', dtype=torch.float32)

    results = []

    # No conflicts
    def no_conflict():
        return memory_patterns.shared_no_conflict(x)

    r = profiler.benchmark_memory_bound(
        "Shared mem (no conflicts)",
        no_conflict,
        bytes_accessed=bytes_accessed
    )
    results.append(r)

    # Maximum conflicts (stride=32)
    def bank_conflict():
        return memory_patterns.shared_bank_conflict(x)

    r = profiler.benchmark_memory_bound(
        "Shared mem (bank conflicts)",
        bank_conflict,
        bytes_accessed=bytes_accessed
    )
    results.append(r)

    print_comparison_table(results, "Bank Conflicts", show_bandwidth=True)

    # Analysis
    print("\n" + "="*70)
    print(" ANALYSIS")
    print("="*70)
    print("""
    Key observations:

    1. COALESCING IS CRITICAL:
       - Coalesced access: ~200+ GB/s (near peak)
       - Stride=32 access: ~20-50 GB/s (5-10x slower!)
       - This is the #1 optimization for memory-bound kernels

    2. HOW COALESCING WORKS:
       - 32 threads in a warp access memory together
       - If addresses are consecutive: 1 memory transaction
       - If addresses are scattered: up to 32 transactions
       - GPU memory controller handles merging automatically

    3. 2D ARRAY ACCESS:
       - Row-major storage + row-wise access = coalesced
       - Row-major storage + column-wise access = strided
       - Solution: transpose data or use tiled algorithms

    4. ALIGNMENT:
       - Modern GPUs handle misalignment well
       - Small offsets (1-3) have minimal impact
       - Larger offsets may cause extra transactions

    5. BANK CONFLICTS:
       - Shared memory: 32 banks, 4-byte words
       - Stride-32 access = all threads hit same bank
       - Up to 32x serialization!
       - Solution: pad shared memory arrays (+1 element)

    6. PRACTICAL IMPLICATIONS:
       - Always access memory in coalesced patterns
       - Transpose data if needed (pay cost once)
       - Pad shared memory to avoid bank conflicts
       - Profile with Nsight to find inefficiencies
    """)


if __name__ == "__main__":
    main()
