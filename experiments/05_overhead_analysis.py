#!/usr/bin/env python3
"""
Experiment 5: Framework Overhead Analysis

This experiment demonstrates:
1. Kernel launch overhead (~5-10 μs per launch)
2. Python loop overhead
3. Why operator fusion matters for models like Llama
4. The difference between "many small ops" vs "few large ops"

Key insight: For simple operations, the bottleneck shifts from
compute/memory to kernel launch and Python overhead.

This explains why:
- PyTorch is fast for large operations
- PyTorch can be slow for many small operations
- Compiled frameworks (TorchScript, torch.compile) help
- Custom fused kernels are important for transformers
"""

import torch
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/workspace')

from utils.profiler import GPUProfiler, print_comparison_table, get_gpu_info
from tabulate import tabulate


def measure_kernel_launch_overhead():
    """Measure the overhead of launching an empty-ish kernel"""
    print("\n" + "="*70)
    print(" PART 1: Kernel Launch Overhead")
    print("="*70)

    # Very small tensor - work is negligible, measures launch overhead
    x = torch.ones(1, device='cuda')

    # Warmup
    for _ in range(100):
        _ = x + x

    torch.cuda.synchronize()

    # Measure many kernel launches
    num_launches = 10000

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_launches):
        _ = x + x  # Minimal compute kernel
    end.record()

    torch.cuda.synchronize()
    total_time_ms = start.elapsed_time(end)

    overhead_us = (total_time_ms / num_launches) * 1000

    print(f"\nKernel launches: {num_launches}")
    print(f"Total time: {total_time_ms:.2f} ms")
    print(f"Per-launch overhead: {overhead_us:.2f} μs")

    return overhead_us


def measure_python_loop_overhead():
    """Compare Python loop vs single fused operation"""
    print("\n" + "="*70)
    print(" PART 2: Python Loop Overhead")
    print("="*70)

    n = 1 << 20  # 1M elements
    num_ops = 100  # Number of operations to chain

    x = torch.randn(n, device='cuda')

    # Warmup
    y = x.clone()
    for _ in range(10):
        for _ in range(num_ops):
            y = F.relu(y)
    torch.cuda.synchronize()

    # Method 1: Python loop with individual ops
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    y = x.clone()
    start.record()
    for _ in range(num_ops):
        y = F.relu(y)
    end.record()
    torch.cuda.synchronize()

    loop_time_ms = start.elapsed_time(end)

    # Method 2: Single ReLU (baseline for comparison)
    y = x.clone()
    start.record()
    y = F.relu(y)
    end.record()
    torch.cuda.synchronize()

    single_time_ms = start.elapsed_time(end)

    # Calculate overhead
    expected_time = single_time_ms * num_ops
    actual_overhead = loop_time_ms - expected_time

    print(f"\nTensor size: {n:,} elements")
    print(f"Number of chained ReLUs: {num_ops}")
    print(f"\nSingle ReLU: {single_time_ms:.4f} ms")
    print(f"Expected for {num_ops}x: {expected_time:.4f} ms")
    print(f"Actual time: {loop_time_ms:.4f} ms")
    print(f"Overhead: {actual_overhead:.4f} ms ({100*actual_overhead/loop_time_ms:.1f}%)")


def compare_granularity():
    """Compare many small ops vs few large ops"""
    print("\n" + "="*70)
    print(" PART 3: Operation Granularity")
    print("="*70)

    total_elements = 1 << 26  # 64M elements total

    profiler = GPUProfiler(warmup_iters=5, measure_iters=50)

    results = []
    headers = ["Scenario", "Ops", "Elements/Op", "Time (ms)", "Throughput"]

    # Different granularities (same total work)
    granularities = [
        (1, total_elements),           # 1 large op
        (64, total_elements // 64),    # 64 medium ops
        (1024, total_elements // 1024), # 1024 small ops
        (16384, total_elements // 16384), # 16K tiny ops
    ]

    for num_ops, elements_per_op in granularities:
        tensors = [torch.randn(elements_per_op, device='cuda') for _ in range(num_ops)]

        def run_ops():
            for t in tensors:
                F.relu(t)

        time_ms = profiler.time_kernel(run_ops)
        throughput = total_elements / (time_ms / 1000) / 1e9  # G elements/sec

        results.append([
            f"{num_ops} ops x {elements_per_op:,} elem",
            num_ops,
            elements_per_op,
            f"{time_ms:.4f}",
            f"{throughput:.1f} Gelem/s"
        ])

    print(f"\nTotal elements processed: {total_elements:,}")
    print(tabulate(results, headers=headers, tablefmt="grid"))


def simulate_transformer_layer():
    """Compare unfused vs fused operations in transformer-like patterns"""
    print("\n" + "="*70)
    print(" PART 4: Transformer-like Operation Patterns")
    print("="*70)

    batch_size = 32
    seq_len = 512
    hidden_dim = 768

    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    w1 = torch.randn(hidden_dim, hidden_dim * 4, device='cuda')
    w2 = torch.randn(hidden_dim * 4, hidden_dim, device='cuda')
    gamma = torch.ones(hidden_dim, device='cuda')
    beta = torch.zeros(hidden_dim, device='cuda')

    profiler = GPUProfiler(warmup_iters=10, measure_iters=50)

    results = []

    # Pattern 1: Unfused LayerNorm + Linear (multiple kernels)
    def unfused_pattern():
        # Manual layer norm (multiple ops)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        x_scaled = x_norm * gamma + beta
        # Linear
        return torch.matmul(x_scaled, w1)

    time_unfused = profiler.time_kernel(unfused_pattern)
    results.append(("Unfused (manual LN + matmul)", f"{time_unfused:.4f} ms"))

    # Pattern 2: PyTorch LayerNorm (more fused)
    ln = torch.nn.LayerNorm(hidden_dim).cuda()
    def pytorch_ln_pattern():
        x_norm = ln(x)
        return torch.matmul(x_norm, w1)

    time_pytorch_ln = profiler.time_kernel(pytorch_ln_pattern)
    results.append(("PyTorch nn.LayerNorm + matmul", f"{time_pytorch_ln:.4f} ms"))

    # Pattern 3: Just matmul (baseline)
    def just_matmul():
        return torch.matmul(x, w1)

    time_matmul = profiler.time_kernel(just_matmul)
    results.append(("Just matmul (baseline)", f"{time_matmul:.4f} ms"))

    # Pattern 4: MLP block (2 matmuls + activation)
    def mlp_unfused():
        h = torch.matmul(x, w1)
        h = F.gelu(h)
        return torch.matmul(h, w2)

    time_mlp = profiler.time_kernel(mlp_unfused)
    results.append(("MLP (2x matmul + GELU)", f"{time_mlp:.4f} ms"))

    print(f"\nInput shape: [{batch_size}, {seq_len}, {hidden_dim}]")
    print(f"Hidden dimension: {hidden_dim} -> {hidden_dim * 4} -> {hidden_dim}")
    print()
    print(tabulate(results, headers=["Pattern", "Time"], tablefmt="grid"))

    print("""
    Note: This is a simplified demonstration. Real transformer optimizations include:
    - Flash Attention (fused attention kernel)
    - Fused LayerNorm + Linear
    - Fused MLP blocks (GELU fusion)
    - KV-cache optimizations
    """)


def main():
    print("\n" + "="*70)
    print(" EXPERIMENT 5: Framework Overhead Analysis")
    print("="*70)

    get_gpu_info()

    measure_kernel_launch_overhead()
    measure_python_loop_overhead()
    compare_granularity()
    simulate_transformer_layer()

    # Analysis
    print("\n" + "="*70)
    print(" ANALYSIS")
    print("="*70)
    print("""
    Key observations:

    1. KERNEL LAUNCH OVERHEAD:
       - Each kernel launch has ~5-10 μs overhead
       - For 1μs of compute, this is 5-10x overhead!
       - For 1ms of compute, this is negligible (0.5-1%)

    2. PYTHON LOOP OVERHEAD:
       - Each Python iteration adds ~1-2 μs
       - Combined with launch overhead: significant for small ops
       - Solutions: torch.compile, custom fused kernels

    3. OPERATION GRANULARITY:
       - 1 large op >> many small ops (same total work)
       - "Death by a thousand kernel launches"
       - This is why batch size matters for efficiency

    4. IMPLICATIONS FOR LLMS:
       - Transformer layers have many small operations
       - LayerNorm, attention, MLP each have multiple kernels
       - Unfused: dozens of kernel launches per layer
       - Fused: fewer launches, better performance

    5. OPTIMIZATION STRATEGIES:
       a) Operator Fusion: Combine multiple ops into one kernel
          - Flash Attention fuses Q@K^T, softmax, @V
          - Fused LayerNorm eliminates intermediate tensors

       b) torch.compile: Automatic fusion and optimization
          - Traces operations and generates fused CUDA
          - Can achieve 2-3x speedup on transformer blocks

       c) Larger Batch Sizes: More work per kernel launch
          - Better amortization of overhead
          - Better GPU utilization

       d) Custom CUDA Kernels: Ultimate control
          - Used by FlashAttention, xFormers
          - Requires deep GPU expertise

    6. WHEN DOES PYTORCH OVERHEAD MATTER?
       - Small tensors: YES (overhead dominates)
       - Large tensors: NO (compute dominates)
       - Many ops: YES (launch overhead accumulates)
       - Few ops: NO (launches are rare)

       Rule of thumb: If your kernel takes <100μs, overhead matters.
    """)


if __name__ == "__main__":
    main()
