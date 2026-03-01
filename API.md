# API Reference

## utils.profiler

### Constants

```python
RTX4060_MEMORY_BW_GBS = 256.0   # GB/s theoretical bandwidth
RTX4060_FP32_TFLOPS   = 15.0    # Theoretical FP32 throughput
```

### TimingResult (dataclass)

```python
@dataclass
class TimingResult:
    name: str               # Implementation name
    time_ms: float          # Kernel execution time (ms)
    bandwidth_gbs: float = 0.0    # Achieved bandwidth (GB/s)
    flops_tflops: float = 0.0    # Achieved TFLOPS
    bandwidth_pct: float = 0.0   # % of theoretical peak bandwidth
    flops_pct: float = 0.0       # % of theoretical peak FLOPS
```

### GPUProfiler

```python
GPUProfiler(warmup_iters=10, measure_iters=100, memory_bw_gbs=256.0, fp32_tflops=15.0)
```

#### Core Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `time_kernel` | `(kernel_fn, *args, **kwargs)` | `float` (ms) | Time a kernel using CUDA events |
| `time_with_transfer` | `(kernel_fn, cpu_tensors, **kwargs)` | `(h2d_ms, kernel_ms, d2h_ms)` | Measure H2D + kernel + D2H times |
| `calc_bandwidth` | `(bytes_accessed, time_ms)` | `float` (GB/s) | Achieved memory bandwidth |
| `calc_flops` | `(num_ops, time_ms)` | `float` (TFLOPS) | Achieved compute throughput |
| `bandwidth_efficiency` | `(achieved_gbs)` | `float` (%) | % of theoretical peak bandwidth |
| `compute_efficiency` | `(achieved_tflops)` | `float` (%) | % of theoretical peak FLOPS |

#### Benchmark Methods

| Method | Extra Params | Returns | Use For |
|--------|-------------|---------|---------|
| `benchmark_memory_bound` | `name, kernel_fn, *args, bytes_accessed=` | `TimingResult` | Memory-bound kernels |
| `benchmark_compute_bound` | `name, kernel_fn, *args, num_flops=` | `TimingResult` | Compute-bound kernels |
| `benchmark_full` | `name, kernel_fn, *args, bytes_accessed=0, num_flops=0` | `TimingResult` | Both metrics |

### Module Functions

```python
benchmark(name, fn, *args, warmup=10, iters=100, **kwargs) -> float
    # Simple benchmark, returns average time in ms

print_comparison_table(results: List[TimingResult], title="Performance Comparison",
                       show_bandwidth=True, show_flops=False)
    # Formatted table output

print_transfer_breakdown(h2d_ms, kernel_ms, d2h_ms, total_bytes)
    # H2D/kernel/D2H time breakdown

get_gpu_info()
    # Print GPU properties
```

### Package Exports (`from utils import ...`)

- `GPUProfiler`
- `benchmark`
- `print_comparison_table`

---

## CUDA Kernels (`cuda_kernels.*`)

Built via `pip install -e /workspace/kernels`. Functions operate on `torch.Tensor` (float32, CUDA). Some accept additional scalar parameters (noted below).

### cuda_kernels.vector_add

| Function | Description | Bytes Accessed |
|----------|-------------|---------------|
| `vector_add(a, b)` | Element-wise addition | `3 × N × 4` |
| `vector_add_vec4(a, b)` | float4-vectorized addition | `3 × N × 4` |

### cuda_kernels.relu

| Function | Description | Bytes Accessed |
|----------|-------------|---------------|
| `relu(x)` | `fmaxf(0, x)` | `2 × N × 4` |
| `relu_branchless(x)` | `x * (x > 0)` | `2 × N × 4` |
| `relu_inplace(x)` | In-place ReLU (modifies input) | `N × 4` (read) + `N × 4` (write) |
| `relu_vec4(x)` | float4-vectorized ReLU | `2 × N × 4` |

### cuda_kernels.matmul

| Function | Description | Thread Block | FLOPs |
|----------|-------------|-------------|-------|
| `matmul_naive(A, B)` | One thread per output element | 16×16 | `2 × M × N × K` |
| `matmul_tiled(A, B)` | 32×32 shared memory tiles | 32×32 | `2 × M × N × K` |
| `matmul_tiled_doublebuf(A, B)` | Double-buffered tiled | 32×32 | `2 × M × N × K` |

### cuda_kernels.memory_patterns

| Function | Description | Pattern |
|----------|-------------|---------|
| `copy_coalesced(src)` | Consecutive thread→consecutive addr | Optimal |
| `copy_strided(src, stride: int)` | Strided access pattern | Non-coalesced |
| `copy_row_major(src)` | 2D row-major traversal | Coalesced |
| `copy_column_major(src)` | 2D column-major traversal | Non-coalesced |
| `copy_misaligned(src, offset: int)` | Non-aligned offset copy (returns tensor of size `n - offset`) | Misaligned |
| `shared_no_conflict(src)` | Shared mem, no bank conflicts | Optimal |
| `shared_bank_conflict(src)` | Shared mem, stride-32 conflicts | Worst case |

---

## Experiment Scripts

All in `experiments/`, runnable as `python experiments/0X_*.py`.

### 01_vector_add.py

Sizes: 1M, 4M, 16M, 64M elements. Compares PyTorch ops vs custom CUDA.

### 02_relu.py

Sizes: 4M, 16M, 64M. Compares `F.relu`, `torch.relu`, `torch.clamp`, `torch.maximum`, and 3 custom CUDA variants (`relu`, `relu_branchless`, `relu_vec4`).

### 03_matmul.py

Sizes: 256², 512², 1024², 2048². Compares cuBLAS (via PyTorch) vs tiled/double-buffered CUDA. Naive kernel only benchmarked for sizes ≤ 1024.

### 04_memory_patterns.py

Four parts: coalesced vs strided, row vs column major, aligned vs misaligned, shared memory bank conflicts.

### 05_overhead_analysis.py

Key functions:
- `measure_kernel_launch_overhead() -> float` (microseconds)
- `measure_python_loop_overhead()`
- `compare_granularity()`
- `simulate_transformer_layer()`

### 06_tensor_pipeline_parallelism.py

```bash
# Single GPU demo
python experiments/06_tensor_pipeline_parallelism.py

# Multi-GPU tensor parallelism
torchrun --nproc_per_node=N experiments/06_tensor_pipeline_parallelism.py --mode tp

# Multi-GPU pipeline parallelism
torchrun --nproc_per_node=N experiments/06_tensor_pipeline_parallelism.py --mode pp
```

Key classes: `TransformerBlock`, `SimpleTransformerLM`, `TensorParallelTransformerBlock`, `PipelineStage`

Key functions: `apply_tensor_parallelism_manual()`, `create_pipeline_stages()`, `gpipe_forward()`, `demo_tensor_parallelism()`, `demo_pipeline_parallelism()`, `one_f1b_schedule_description()`, `demo_pytorch_native_apis()`, `print_parallelism_comparison()`, `run_distributed_tp()`, `run_distributed_pp()`

---

## Scripts

### run_profiler.sh

```bash
./run_profiler.sh run-all                  # Run all experiments
./run_profiler.sh nsys [experiment]        # Nsight Systems timeline
./run_profiler.sh nsys-quick [experiment]  # Quick timeline
./run_profiler.sh ncu [experiment]         # Nsight Compute kernel analysis
./run_profiler.sh ncu-kernel <name> [exp]  # Profile specific kernel
./run_profiler.sh ncu-memory [experiment]  # Memory-focused analysis
./run_profiler.sh nsys-stats [report]      # Print stats from .nsys-rep
./run_profiler.sh list                     # List available experiments
./run_profiler.sh help                     # Show usage
```

### env_check.sh / check_env.py

Environment validation: NVIDIA driver, CUDA toolkit, Python, PyTorch, GPU access, custom kernels, Nsight tools, workspace structure.
