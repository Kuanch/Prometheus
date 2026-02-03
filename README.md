# GPU Architecture Learning Lab

**Understand how NVIDIA GPUs really work** by comparing PyTorch operations with custom CUDA kernels and profiling everything.

## Why This Project?

When you call `torch.matmul()`, what actually happens on the GPU? Is PyTorch slow compared to "raw CUDA"? When does framework overhead matter?

This project answers these questions through hands-on experiments:

- **Write custom CUDA kernels** for vector add, ReLU, and matrix multiplication
- **Compare with PyTorch** to see where overhead exists (spoiler: less than you think for large ops)
- **Profile with Nsight** to visualize memory transfers, kernel execution, and bottlenecks
- **Learn GPU fundamentals**: memory coalescing, shared memory, compute vs memory bound

## What You'll Learn

| Concept | Experiment | Key Insight |
|---------|------------|-------------|
| Memory Bandwidth | Vector Add, ReLU | Simple ops are memory-bound, not compute-bound |
| Compute Throughput | Matrix Multiply | cuBLAS is 10-100x faster than naive CUDA |
| Memory Coalescing | Memory Patterns | Strided access can be 10-30x slower |
| Framework Overhead | Overhead Analysis | Matters for small ops, negligible for large ops |

## Quick Start

### Prerequisites

- NVIDIA GPU (tested on RTX 4060)
- Docker with NVIDIA Container Toolkit
- WSL2 (for Windows) or native Linux

### Run Your First Experiment

```bash
# Clone and enter the project
git clone https://github.com/YOUR_USERNAME/GPU_arch_modeling.git
cd GPU_arch_modeling

# Build the Docker environment
docker-compose build

# Build custom CUDA kernels
docker-compose run --rm gpu-lab pip install -e /workspace/kernels

# Run the vector addition experiment
docker-compose run --rm gpu-lab python experiments/01_vector_add.py
```

### Expected Output

```
======================================================================
 EXPERIMENT 1: Vector Addition (Memory-Bound Operation)
======================================================================

GPU Information:
  Device: NVIDIA GeForce RTX 4060 Laptop GPU
  Compute Capability: 8.9
  Total Memory: 8.0 GB

Vector size: 16,777,216 elements (64.0 MB per vector)
Total memory accessed: 192.0 MB

+----------------------+------------+-----------+--------+
| Implementation       | Time (ms)  | BW (GB/s) | BW %   |
+----------------------+------------+-----------+--------+
| PyTorch (a + b)      | 0.0891     | 215.5     | 84.2%  |
| Custom CUDA          | 0.0887     | 216.5     | 84.6%  |
| Custom CUDA (vec4)   | 0.0883     | 217.4     | 84.9%  |
+----------------------+------------+-----------+--------+

✓ Results verified: PyTorch and Custom CUDA match
```

**Key observation**: PyTorch matches custom CUDA because both hit the memory bandwidth limit (~256 GB/s theoretical).

## Project Structure

```
GPU_arch_modeling/
├── experiments/           # Python scripts comparing PyTorch vs CUDA
│   ├── 01_vector_add.py   # Memory-bound baseline
│   ├── 02_relu.py         # Element-wise operations
│   ├── 03_matmul.py       # Compute-bound comparison
│   ├── 04_memory_patterns.py  # Coalescing experiments
│   └── 05_overhead_analysis.py # Framework overhead
├── kernels/               # Custom CUDA implementations
│   ├── vector_add.cu      # Simple kernel (learning baseline)
│   ├── relu.cu            # Multiple ReLU variants
│   ├── matmul.cu          # Naive → Tiled → Double-buffered
│   └── memory_patterns.cu # Coalesced vs strided access
├── utils/
│   └── profiler.py        # CUDA event timing utilities
├── scripts/
│   └── run_profiler.sh    # Nsight profiling helpers
├── Dockerfile             # NVIDIA PyTorch + Nsight tools
└── docker-compose.yml     # GPU passthrough config
```

## Experiments Overview

### 1. Vector Addition (Memory-Bound)
Compares `a + b` in PyTorch vs custom CUDA. Both achieve ~85% of peak memory bandwidth because the operation is limited by how fast you can read/write memory, not compute.

### 2. ReLU (Element-wise)
Tests different ReLU implementations: `F.relu()`, `torch.clamp()`, custom CUDA. All perform similarly because they're all memory-bound (1 read + 1 write per element).

### 3. Matrix Multiplication (Compute-Bound)
The most dramatic experiment. Shows why cuBLAS exists:
- **Naive CUDA**: ~0.1 TFLOPS (each thread reads entire rows/columns)
- **Tiled CUDA**: ~2 TFLOPS (shared memory reduces global reads)
- **cuBLAS**: ~12 TFLOPS (register tiling, vectorization, tensor cores)

### 4. Memory Access Patterns
Demonstrates why memory coalescing matters:
- **Coalesced** (stride=1): ~220 GB/s
- **Strided** (stride=32): ~20 GB/s

A 10x slowdown just from bad access patterns!

### 5. Overhead Analysis
Measures the cost of:
- Kernel launch: ~5-10 μs per launch
- Python loop overhead: ~1-2 μs per iteration
- Many small ops vs few large ops

This explains why `torch.compile()` and fused kernels matter for transformers.

## Profiling with Nsight

Generate visual profiles for deeper analysis:

```bash
# Timeline view (Nsight Systems)
docker-compose run --rm gpu-lab nsys profile -o /output/vector_add python experiments/01_vector_add.py

# Kernel analysis (Nsight Compute)
docker-compose run --rm gpu-lab ncu --set full --export /output/matmul python experiments/03_matmul.py
```

Open the `.nsys-rep` and `.ncu-rep` files in Nsight Systems/Compute GUI on Windows.

## Key Takeaways

1. **PyTorch is not slow** - for large operations, it's within a few percent of hand-written CUDA
2. **Memory bandwidth is often the limit** - simple ops like add/relu can't go faster no matter how you optimize compute
3. **cuBLAS is magic** - don't write your own GEMM unless you're learning
4. **Memory coalescing is critical** - bad access patterns kill performance
5. **Overhead matters for small ops** - this is why transformers need fused kernels

## Requirements

- Docker 20.10+
- NVIDIA Driver 525+ (for CUDA 12.x support)
- NVIDIA Container Toolkit
- 4GB+ GPU memory recommended
- Nsight Systems/Compute GUI (optional, for viewing profiles on Windows)

## Further Reading

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) - Excellent intro to GPU optimization

## License

MIT License - feel free to use this for learning and teaching.

## Contributing

Found a bug or want to add an experiment? PRs welcome!

Ideas for contribution:
- Add FP16/TF32 experiments for Tensor Cores
- Profile a real model (GPT-2, Llama)
- Add Flash Attention comparison
- Multi-GPU experiments
