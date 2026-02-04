# EXPERIMENTS.md - Detailed Experiment Guide

This document explains the methodology, interpretation, and future directions for each experiment in this GPU architecture learning project.

## Table of Contents

1. [Foundational Concepts](#foundational-concepts)
2. [Experiment 1: Vector Addition](#experiment-1-vector-addition)
3. [Experiment 2: ReLU](#experiment-2-relu)
4. [Experiment 3: Matrix Multiplication](#experiment-3-matrix-multiplication)
5. [Experiment 4: Memory Access Patterns](#experiment-4-memory-access-patterns)
6. [Experiment 5: Overhead Analysis](#experiment-5-overhead-analysis)
7. [Experiment 6: Llama 3.1 8B Inference](#experiment-6-llama-31-8b-inference)
8. [Experiment 7: Grouped Query Attention (GQA)](#experiment-7-grouped-query-attention-gqa)
9. [Future Roadmap](#future-roadmap)

---

## Foundational Concepts

### Memory-Bound vs Compute-Bound

Every GPU operation falls somewhere on the spectrum between memory-bound and compute-bound:

**Memory-Bound Operations**
- Performance limited by memory bandwidth (GB/s)
- Examples: vector add, ReLU, copy operations
- Arithmetic Intensity < ~10 FLOPs/byte
- Optimization: improve memory access patterns, not compute

**Compute-Bound Operations**
- Performance limited by compute throughput (TFLOPS)
- Examples: matrix multiplication, convolutions
- Arithmetic Intensity > ~100 FLOPs/byte
- Optimization: maximize FLOPs, use Tensor Cores

### The Roofline Model

The Roofline model predicts performance based on Arithmetic Intensity (AI):

```
AI = FLOPs / Bytes accessed

If AI < ridge_point: memory-bound (performance = AI × bandwidth)
If AI > ridge_point: compute-bound (performance = peak_flops)

RTX 4060 ridge point ≈ 15 TFLOPS / 256 GB/s ≈ 60 FLOPs/byte
```

### CUDA Event Timing

We use CUDA events instead of Python `time.time()` for accurate GPU timing:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# ... kernel execution ...
end.record()
torch.cuda.synchronize()

time_ms = start.elapsed_time(end)
```

This measures actual GPU time, not Python overhead.

---

## Experiment 1: Vector Addition

**File**: `experiments/01_vector_add.py`

### Purpose

Establish a baseline for memory-bound operations and verify that PyTorch overhead is negligible for large tensors.

### Hypothesis

Vector addition is purely memory-bound. The compute (one add per element) is trivial compared to the memory traffic (2 reads + 1 write = 12 bytes per element). All implementations should achieve similar bandwidth.

### What It Measures

| Metric | Calculation |
|--------|-------------|
| Time | CUDA event elapsed time |
| Bytes Accessed | 3 × N × sizeof(float) = 12N bytes |
| Bandwidth | Bytes / Time |
| Efficiency | Achieved BW / Theoretical BW (256 GB/s) |

### Implementations Compared

1. **PyTorch `a + b`**: Standard operator
2. **`torch.add()`**: Function form
3. **`torch.add(out=c)`**: Pre-allocated output
4. **Custom CUDA kernel**: Hand-written `__global__` function
5. **Custom CUDA (vec4)**: Vectorized float4 loads

### How to Run

```bash
docker-compose run --rm gpu-lab python experiments/01_vector_add.py
```

### Interpreting Results

- **All implementations within 5%**: Confirms memory-bound behavior
- **>80% bandwidth efficiency**: Good memory subsystem utilization
- **vec4 slightly faster**: Vectorized loads can help, but L2 cache effects often dominate

### Key Code

```cuda
// Simple kernel - one thread per element
__global__ void vector_add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
```

---

## Experiment 2: ReLU

**File**: `experiments/02_relu.py`

### Purpose

Compare different ReLU implementations to show they're all equivalent when memory-bound.

### Hypothesis

ReLU (1 read + 1 write = 8 bytes/element) is even more memory-bound than vector add. Framework overhead should be negligible. Different implementations (`F.relu`, `torch.clamp`, custom CUDA) should achieve the same bandwidth.

### What It Measures

| Metric | Calculation |
|--------|-------------|
| Bytes Accessed | 2 × N × sizeof(float) = 8N bytes |
| Bandwidth | Bytes / Time |
| Branch efficiency | Compare branching vs branchless |

### Implementations Compared

1. **`F.relu()`**: PyTorch functional
2. **`torch.relu()`**: Torch function
3. **`torch.clamp(min=0)`**: Alternative implementation
4. **`F.relu(inplace=True)`**: In-place variant
5. **Custom CUDA**: `fmaxf(0.0f, x[idx])`
6. **Custom CUDA (branchless)**: `val * (val > 0)`
7. **Custom CUDA (vec4)**: Vectorized version

### Interpreting Results

- **Higher bandwidth than vector add**: ReLU moves 8 bytes/element vs 12 bytes/element, so same time = higher reported bandwidth
- **Branching vs branchless**: Modern GPUs handle simple branches efficiently
- **In-place appears slower**: Because our benchmark resets data each iteration

---

## Experiment 3: Matrix Multiplication

**File**: `experiments/03_matmul.py`

### Purpose

Demonstrate compute-bound behavior and why optimized libraries (cuBLAS) matter.

### Hypothesis

Matrix multiplication has high arithmetic intensity (O(n³) compute, O(n²) memory). Naive implementations are memory-bound despite being "compute-intensive" because they access global memory inefficiently. Tiled implementations use shared memory to amortize global memory access.

### What It Measures

| Metric | Calculation |
|--------|-------------|
| FLOPs | 2 × M × N × K |
| TFLOPS | FLOPs / Time / 1e12 |
| Efficiency | Achieved / Peak (15 TFLOPS) |

### Implementations Compared

1. **PyTorch `torch.matmul()`**: Uses cuBLAS
2. **Custom Naive CUDA**: One thread per output element, each reads K values from A and B
3. **Custom Tiled CUDA**: 32×32 tiles in shared memory
4. **Custom Tiled + Double Buffering**: Overlap compute with memory loads

### The Performance Gap

```
For 2048×2048 matrices:
- Naive CUDA:    ~0.1 TFLOPS (memory-bound despite high AI!)
- Tiled CUDA:    ~2 TFLOPS   (20x improvement)
- cuBLAS:        ~12 TFLOPS  (6x more, 80% of peak)
```

### Why cuBLAS Wins

1. **Register tiling**: Each thread computes multiple outputs
2. **Vectorized loads**: float4 or wider memory transactions
3. **Double buffering**: Overlap memory and compute
4. **Tensor Core usage**: On supported hardware (we use FP32, so no TC)
5. **Autotuning**: Optimized for specific matrix sizes and GPUs

### Key Code: Naive vs Tiled

```cuda
// Naive: Each thread reads K elements from global memory
__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];  // K global reads per output!
    }
    C[row * N + col] = sum;
}

// Tiled: Load tiles into shared memory, reuse across threads
__global__ void matmul_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    // ... load tiles, compute partial products, accumulate ...
}
```

---

## Experiment 4: Memory Access Patterns

**File**: `experiments/04_memory_patterns.py`

### Purpose

Demonstrate the critical importance of memory coalescing for GPU performance.

### Hypothesis

Coalesced memory access (consecutive threads access consecutive addresses) allows the GPU to issue single wide memory transactions. Non-coalesced access requires multiple transactions, dramatically reducing effective bandwidth.

### What It Measures

| Pattern | Expected Bandwidth |
|---------|-------------------|
| Coalesced (stride=1) | ~220 GB/s |
| Stride=2 | ~150 GB/s |
| Stride=32 | ~20-50 GB/s |

### Patterns Tested

1. **Coalesced**: `dst[idx] = src[idx]`
2. **Strided**: `dst[idx] = src[idx * stride % n]`
3. **Row-major 2D**: Threads access consecutive columns
4. **Column-major 2D**: Threads access consecutive rows (bad for row-major storage)
5. **Misaligned**: Access starting from non-aligned offset
6. **Shared memory bank conflicts**: Stride-32 access pattern

### Why Coalescing Matters

```
32 threads in a warp execute together.

Coalesced: All 32 threads access addresses 0, 4, 8, ..., 124
  → GPU issues ONE 128-byte transaction

Strided (stride=32): Threads access addresses 0, 128, 256, ...
  → GPU issues 32 SEPARATE transactions
  → 32x more memory transactions = much slower
```

### Bank Conflicts in Shared Memory

Shared memory has 32 banks. Consecutive 4-byte words go to consecutive banks.

```
No conflict: smem[tid] - each thread hits different bank
Max conflict: smem[tid * 32] - all threads hit bank 0 → serialized
```

---

## Experiment 5: Overhead Analysis

**File**: `experiments/05_overhead_analysis.py`

### Purpose

Understand framework overhead and when it matters.

### Hypothesis

- Kernel launch has fixed overhead (~5-10 μs)
- Python loop has overhead (~1-2 μs per iteration)
- For small tensors, overhead dominates
- For large tensors, overhead is negligible

### What It Measures

1. **Kernel launch overhead**: Time to launch minimal kernel
2. **Python loop overhead**: Many ops vs single fused op
3. **Granularity impact**: Same total work, different op sizes
4. **Transformer patterns**: Multiple sequential operations

### Key Results

```
Kernel launch overhead: ~5-10 μs

1 op × 64M elements:     ~1.2 ms
1024 ops × 64K elements: ~12 ms (10x slower for same total work!)
```

### Implications for Deep Learning

- **Transformers have many small ops**: LayerNorm, attention, MLP
- **Unfused**: Dozens of kernel launches per layer
- **Fused (Flash Attention)**: One kernel for entire attention
- **torch.compile**: Automatic fusion can help

### When Does Overhead Matter?

| Tensor Size | Kernel Time | Overhead % |
|-------------|-------------|------------|
| 1K elements | ~1 μs | 90%+ |
| 1M elements | ~50 μs | 10-20% |
| 64M elements | ~1 ms | <1% |

---

## Experiment 6: Llama 3.1 8B Inference

**File**: `experiments/06_llama_inference.py`

### Purpose

Profile a real Llama 3.1 8B transformer model during inference, observing the difference between prefill and decode phases.

### Prerequisites

- Llama 3.1 8B model files (safetensors + tokenizer) at `/workspace/llama/` or set `LLAMA_MODEL_PATH`
- Uses 4-bit NF4 quantization to fit within 8GB VRAM

### What It Measures

| Phase | Behavior | Bottleneck |
|-------|----------|------------|
| Prefill | Process all prompt tokens in parallel | Compute-bound (large batched matmuls) |
| Decode | Generate one token at a time | Memory-bound (reads entire KV-cache per step) |

### Profiling Features

- **NVTX annotations** on every transformer layer and sublayer (Attention, MLP, Norm)
- **LayerProfiler hooks** automatically tag each layer for Nsight timeline
- **Step-by-step decode** with per-token timing for detailed analysis
- **Three phases**: prefill benchmark, full generation, step-by-step decode

### How to Run

```bash
# Direct run
python experiments/06_llama_inference.py

# With Nsight Systems
nsys profile --trace=cuda,nvtx -o /output/llama_nsys python experiments/06_llama_inference.py
```

### Interpreting Results

- **Prefill tokens/sec >> Decode tokens/sec**: Prefill processes all tokens in one batched matmul; decode does one at a time
- **In Nsight timeline**: Large cuBLAS kernels during prefill vs many small repeated kernels during decode
- **KV-cache growth**: Each decode step reads more data as the cache grows

---

## Experiment 7: Grouped Query Attention (GQA)

**File**: `experiments/07_gqa_attention.py`

### Purpose

Implement Llama 3's Grouped Query Attention from scratch with every fundamental operation decoupled for individual Nsight profiling. See [LLAMA.md](LLAMA.md) for full architecture details.

### Architecture

Uses Llama 3 8B attention dimensions:
- `d_model=4096`, `n_heads=32`, `n_kv_heads=8`, `head_dim=128`
- 4 query heads share each KV head
- RoPE with theta=500,000, causal masking, no bias

### Decoupled Operations

Each operation is a separate method with its own NVTX range:

| # | Operation | Type | Shape (8B, B=2, S=512) |
|---|-----------|------|------------------------|
| 01 | Q projection | GEMM | [1024, 4096] @ [4096, 4096] |
| 02 | K projection | GEMM | [1024, 4096] @ [4096, 1024] |
| 03 | V projection | GEMM | [1024, 4096] @ [4096, 1024] |
| 04 | RoPE on Q | Element-wise | [2, 32, 512, 128] |
| 05 | RoPE on K | Element-wise | [2, 8, 512, 128] |
| 06 | KV expansion | Repeat/view | 8 heads -> 32 heads |
| 07 | Q @ K^T | Batched GEMM | [64, 512, 128] @ [64, 128, 512] |
| 08 | Causal mask | Element-wise | [2, 32, 512, 512] |
| 09 | Softmax | Reduction | [2, 32, 512, 512] |
| 10 | Attn @ V | Batched GEMM | [64, 512, 512] @ [64, 512, 128] |
| 11 | Output projection | GEMM | [1024, 4096] @ [4096, 4096] |

### Profiling Phases

1. **Individual operation profiling**: Benchmark each of the 11 ops separately with FLOP/bandwidth metrics
2. **End-to-end forward**: Full GQA as a single unit
3. **GQA vs MHA vs MQA**: Compare 8 KV heads vs 32 vs 1
4. **Sequence length sweep**: Show quadratic scaling of attention ops

### How to Run

```bash
python experiments/07_gqa_attention.py

nsys profile --trace=cuda,nvtx -o /output/gqa_nsys python experiments/07_gqa_attention.py
```

### Key Insights

- **Projections (Q, O) dominate at short sequences**: Large GEMMs, compute-bound
- **Attention matmuls (Q@K^T, Attn@V) dominate at long sequences**: O(S^2) scaling
- **GQA vs MHA**: K/V projection cost reduced 4x, KV cache reduced 4x, attention compute unchanged (after expansion)
- **KV expansion is nearly free**: `expand()` is a view operation, no data copy
- **RoPE is negligible**: Cheap element-wise multiply, fully memory-bound

### Arithmetic Intensity

```
Q/K/V/O projections: AI = d_model / 6 ≈ 683 FLOPs/byte  (compute-bound)
Q @ K^T:             AI = S / 6 ≈ 85 FLOPs/byte at S=512 (compute-bound)
Softmax:             AI < 1 FLOPs/byte                    (memory-bound)
```

---

## Future Roadmap

### Phase 1: Precision Experiments (FP16/TF32/BF16)

**Rationale**: Modern GPUs have Tensor Cores that provide 2-8x speedup for reduced precision. Understanding when to use FP16 vs FP32 is crucial for practical optimization.

**Implementation Steps**:

1. **Add FP16 variants to existing kernels**
   - Modify `matmul.cu` to use `__half` type
   - Use `__hmul`, `__hadd` intrinsics or WMMA API
   - Compare cuBLAS FP16 vs FP32

2. **Measure Tensor Core utilization**
   - Use Nsight Compute to verify TC usage
   - Compare theoretical TC TFLOPS vs achieved

3. **Mixed precision patterns**
   - FP16 compute with FP32 accumulation
   - Automatic mixed precision (AMP) comparison

**Files to create**:
- `kernels/matmul_fp16.cu`
- `experiments/06_precision.py`

**Expected insight**: Tensor Cores can provide 4-8x speedup for matrix operations with minimal accuracy loss.

---

### Phase 2: Attention Mechanisms

**Rationale**: Attention is the core of transformers. Flash Attention shows how algorithmic improvements (tiling, recomputation) can beat naive GPU implementations.

**Implementation Steps**:

1. **Naive attention implementation**
   ```python
   # Q @ K^T -> softmax -> @ V
   scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
   attention = torch.softmax(scores, dim=-1)
   output = torch.matmul(attention, V)
   ```

2. **Memory analysis**
   - Naive: O(n²) memory for attention matrix
   - Profile memory bandwidth, identify bottleneck

3. **Compare with Flash Attention**
   - Install `flash-attn` package
   - Benchmark same input sizes
   - Use Nsight to compare kernel patterns

4. **Implement simplified fused attention**
   - Single kernel: load Q, K, V tiles, compute in shared memory
   - Avoid materializing full attention matrix

**Files to create**:
- `kernels/attention.cu`
- `experiments/07_attention.py`

**Expected insight**: Flash Attention is faster not because of better compute, but because it avoids materializing the O(n²) attention matrix.

---

### Phase 3: Profile Real Models

**Rationale**: Understanding how experiments translate to real workloads.

**Implementation Steps**:

1. **Profile GPT-2 inference**
   ```python
   from transformers import GPT2LMHeadModel
   model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
   # Profile with Nsight
   ```

2. **Identify bottlenecks**
   - Which kernels take most time?
   - Memory-bound or compute-bound?
   - What's the kernel launch overhead?

3. **Profile Llama (if GPU memory allows)**
   - Use 4-bit quantization to fit in 8GB
   - Compare quantized vs full precision

4. **Create optimization report**
   - Roofline analysis for each major kernel
   - Recommendations based on our experiments

**Files to create**:
- `experiments/08_gpt2_profile.py`
- `experiments/09_llama_profile.py`

**Expected insight**: Real models spend significant time in attention, but many small operations (LayerNorm, activations) can accumulate overhead.

---

### Phase 4: Multi-GPU Experiments

**Rationale**: Understanding communication overhead for distributed training.

**Implementation Steps**:

1. **NCCL basics**
   - All-reduce timing
   - Ring vs tree topology

2. **Overlap computation and communication**
   - Gradient accumulation
   - Pipeline parallelism basics

3. **Tensor parallelism**
   - Split matrix multiply across GPUs
   - Measure communication overhead

**Note**: Requires multi-GPU setup. Could simulate with single GPU + CPU to understand concepts.

**Files to create**:
- `experiments/10_nccl_basics.py`
- `experiments/11_tensor_parallel.py`

---

### Phase 5: Custom Operator Development

**Rationale**: Learn to write production-quality CUDA kernels.

**Implementation Steps**:

1. **Fused LayerNorm + Linear**
   - Common pattern in transformers
   - Eliminates intermediate tensor

2. **Fused Attention (simplified Flash Attention)**
   - Apply tiling concepts from matmul
   - Online softmax computation

3. **Quantized kernels**
   - INT8 matrix multiplication
   - Weight dequantization on-the-fly

**Files to create**:
- `kernels/fused_layernorm_linear.cu`
- `kernels/fused_attention.cu`
- `kernels/quantized_matmul.cu`

---

## Appendix: Useful Nsight Commands

```bash
# Full timeline with API trace
nsys profile --trace=cuda,nvtx,osrt -o output python script.py

# Kernel-only analysis
ncu --set full --kernel-name "matmul" python script.py

# Memory analysis
ncu --section MemoryWorkloadAnalysis python script.py

# Roofline analysis
ncu --section SpeedOfLight_RooflineChart python script.py

# Compare two runs
ncu --diff baseline.ncu-rep optimized.ncu-rep
```

## Appendix: Key Formulas

```
Memory Bandwidth (GB/s) = Bytes Transferred / Time (s) / 1e9

FLOPS = Floating Point Operations
TFLOPS = FLOPS / Time (s) / 1e12

Arithmetic Intensity = FLOPS / Bytes

Vector Add AI = N / (12N) = 0.083 FLOPs/byte (memory-bound)
ReLU AI = N / (8N) = 0.125 FLOPs/byte (memory-bound)
MatMul AI = 2MNK / (4(MK + KN + MN)) ≈ N/6 for square (compute-bound for large N)
```
