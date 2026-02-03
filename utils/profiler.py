"""
GPU Profiling Utilities

Provides accurate timing using CUDA events and bandwidth/FLOPS calculations
for comparing PyTorch vs custom CUDA kernel performance.
"""

import torch
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from tabulate import tabulate


# RTX 4060 Laptop theoretical peaks
RTX4060_MEMORY_BW_GBS = 256.0   # GB/s (GDDR6, 128-bit bus)
RTX4060_FP32_TFLOPS = 15.0      # TFLOPS


@dataclass
class TimingResult:
    """Results from a timing measurement"""
    name: str
    time_ms: float              # Kernel execution time in milliseconds
    bandwidth_gbs: float = 0.0  # Achieved bandwidth in GB/s
    flops_tflops: float = 0.0   # Achieved TFLOPS
    bandwidth_pct: float = 0.0  # % of theoretical peak bandwidth
    flops_pct: float = 0.0      # % of theoretical peak FLOPS


class GPUProfiler:
    """
    GPU Profiler using CUDA events for accurate timing.

    CUDA events measure time on the GPU, bypassing Python overhead.
    This gives the true kernel execution time.
    """

    def __init__(
        self,
        warmup_iters: int = 10,
        measure_iters: int = 100,
        memory_bw_gbs: float = RTX4060_MEMORY_BW_GBS,
        fp32_tflops: float = RTX4060_FP32_TFLOPS
    ):
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.memory_bw_gbs = memory_bw_gbs
        self.fp32_tflops = fp32_tflops

    def time_kernel(
        self,
        kernel_fn: Callable,
        *args,
        **kwargs
    ) -> float:
        """
        Time a kernel function using CUDA events.

        Returns:
            Average execution time in milliseconds
        """
        # Warmup
        for _ in range(self.warmup_iters):
            kernel_fn(*args, **kwargs)

        torch.cuda.synchronize()

        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Measure
        start_event.record()
        for _ in range(self.measure_iters):
            kernel_fn(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        return total_time_ms / self.measure_iters

    def time_with_transfer(
        self,
        kernel_fn: Callable,
        cpu_tensors: List[torch.Tensor],
        **kernel_kwargs
    ) -> Tuple[float, float, float]:
        """
        Time the full pipeline: H2D transfer + kernel + D2H transfer.

        Returns:
            (h2d_time_ms, kernel_time_ms, d2h_time_ms)
        """
        # Warmup
        gpu_tensors = [t.cuda() for t in cpu_tensors]
        result = kernel_fn(*gpu_tensors, **kernel_kwargs)
        _ = result.cpu()
        torch.cuda.synchronize()

        # Measure H2D transfer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        gpu_tensors = [t.cuda() for t in cpu_tensors]
        end.record()
        torch.cuda.synchronize()
        h2d_time_ms = start.elapsed_time(end)

        # Measure kernel
        start.record()
        for _ in range(self.measure_iters):
            result = kernel_fn(*gpu_tensors, **kernel_kwargs)
        end.record()
        torch.cuda.synchronize()
        kernel_time_ms = start.elapsed_time(end) / self.measure_iters

        # Measure D2H transfer
        start.record()
        _ = result.cpu()
        end.record()
        torch.cuda.synchronize()
        d2h_time_ms = start.elapsed_time(end)

        return h2d_time_ms, kernel_time_ms, d2h_time_ms

    def calc_bandwidth(self, bytes_accessed: int, time_ms: float) -> float:
        """Calculate achieved memory bandwidth in GB/s"""
        if time_ms <= 0:
            return 0.0
        time_s = time_ms / 1000.0
        return (bytes_accessed / 1e9) / time_s

    def calc_flops(self, num_ops: int, time_ms: float) -> float:
        """Calculate achieved TFLOPS"""
        if time_ms <= 0:
            return 0.0
        time_s = time_ms / 1000.0
        return (num_ops / 1e12) / time_s

    def bandwidth_efficiency(self, achieved_gbs: float) -> float:
        """Calculate percentage of theoretical peak bandwidth"""
        return (achieved_gbs / self.memory_bw_gbs) * 100.0

    def compute_efficiency(self, achieved_tflops: float) -> float:
        """Calculate percentage of theoretical peak FLOPS"""
        return (achieved_tflops / self.fp32_tflops) * 100.0

    def benchmark_memory_bound(
        self,
        name: str,
        kernel_fn: Callable,
        *args,
        bytes_accessed: int,
        **kwargs
    ) -> TimingResult:
        """
        Benchmark a memory-bound kernel.

        For memory-bound ops, performance is limited by memory bandwidth.
        Examples: vector add (3 * n * sizeof(float)), ReLU (2 * n * sizeof(float))
        """
        time_ms = self.time_kernel(kernel_fn, *args, **kwargs)
        bw = self.calc_bandwidth(bytes_accessed, time_ms)
        bw_pct = self.bandwidth_efficiency(bw)

        return TimingResult(
            name=name,
            time_ms=time_ms,
            bandwidth_gbs=bw,
            bandwidth_pct=bw_pct
        )

    def benchmark_compute_bound(
        self,
        name: str,
        kernel_fn: Callable,
        *args,
        num_flops: int,
        **kwargs
    ) -> TimingResult:
        """
        Benchmark a compute-bound kernel.

        For compute-bound ops, performance is limited by FLOPS.
        Example: matmul (2 * M * N * K FLOPs)
        """
        time_ms = self.time_kernel(kernel_fn, *args, **kwargs)
        tflops = self.calc_flops(num_flops, time_ms)
        tflops_pct = self.compute_efficiency(tflops)

        return TimingResult(
            name=name,
            time_ms=time_ms,
            flops_tflops=tflops,
            flops_pct=tflops_pct
        )

    def benchmark_full(
        self,
        name: str,
        kernel_fn: Callable,
        *args,
        bytes_accessed: int = 0,
        num_flops: int = 0,
        **kwargs
    ) -> TimingResult:
        """Benchmark with both bandwidth and FLOPS metrics"""
        time_ms = self.time_kernel(kernel_fn, *args, **kwargs)

        bw = self.calc_bandwidth(bytes_accessed, time_ms) if bytes_accessed > 0 else 0.0
        tflops = self.calc_flops(num_flops, time_ms) if num_flops > 0 else 0.0

        return TimingResult(
            name=name,
            time_ms=time_ms,
            bandwidth_gbs=bw,
            flops_tflops=tflops,
            bandwidth_pct=self.bandwidth_efficiency(bw) if bw > 0 else 0.0,
            flops_pct=self.compute_efficiency(tflops) if tflops > 0 else 0.0
        )


def benchmark(
    name: str,
    fn: Callable,
    *args,
    warmup: int = 10,
    iters: int = 100,
    **kwargs
) -> float:
    """
    Simple benchmark function that returns average time in ms.
    Convenience wrapper for quick timing.
    """
    profiler = GPUProfiler(warmup_iters=warmup, measure_iters=iters)
    return profiler.time_kernel(fn, *args, **kwargs)


def print_comparison_table(
    results: List[TimingResult],
    title: str = "Performance Comparison",
    show_bandwidth: bool = True,
    show_flops: bool = False
):
    """Print a formatted comparison table of timing results"""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    headers = ["Implementation", "Time (ms)"]
    if show_bandwidth:
        headers.extend(["BW (GB/s)", "BW %"])
    if show_flops:
        headers.extend(["TFLOPS", "TFLOPS %"])

    rows = []
    for r in results:
        row = [r.name, f"{r.time_ms:.4f}"]
        if show_bandwidth:
            row.extend([f"{r.bandwidth_gbs:.1f}", f"{r.bandwidth_pct:.1f}%"])
        if show_flops:
            row.extend([f"{r.flops_tflops:.2f}", f"{r.flops_pct:.1f}%"])
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Print speedup relative to first result
    if len(results) > 1:
        baseline = results[0].time_ms
        print(f"\nSpeedup vs {results[0].name}:")
        for r in results[1:]:
            speedup = baseline / r.time_ms if r.time_ms > 0 else float('inf')
            print(f"  {r.name}: {speedup:.2f}x")


def print_transfer_breakdown(
    h2d_ms: float,
    kernel_ms: float,
    d2h_ms: float,
    total_bytes: int
):
    """Print breakdown of transfer vs compute time"""
    total_ms = h2d_ms + kernel_ms + d2h_ms

    print(f"\n{'='*50}")
    print(f"{'Transfer vs Compute Breakdown':^50}")
    print(f"{'='*50}")
    print(f"  H2D Transfer:  {h2d_ms:8.4f} ms ({100*h2d_ms/total_ms:5.1f}%)")
    print(f"  Kernel:        {kernel_ms:8.4f} ms ({100*kernel_ms/total_ms:5.1f}%)")
    print(f"  D2H Transfer:  {d2h_ms:8.4f} ms ({100*d2h_ms/total_ms:5.1f}%)")
    print(f"  {'─'*40}")
    print(f"  Total:         {total_ms:8.4f} ms")
    print(f"\n  Data size: {total_bytes / 1e6:.1f} MB")
    print(f"  PCIe BW (H2D): {(total_bytes / 1e9) / (h2d_ms / 1000):.1f} GB/s")


def get_gpu_info():
    """Print GPU information"""
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"\n{'='*50}")
    print(f"{'GPU Information':^50}")
    print(f"{'='*50}")
    print(f"  Device: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Total Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"  SM Count: {props.multi_processor_count}")
    print(f"  Max Threads/Block: {props.max_threads_per_block}")
    print(f"  Max Threads/SM: {props.max_threads_per_multi_processor}")
    print(f"  Warp Size: {props.warp_size}")
    print(f"  L2 Cache: {props.l2_cache_size / 1e6:.1f} MB")
