# Project Layout

```
Prometheus/
├── Dockerfile                    # NVIDIA PyTorch base (25.04-py3) + Nsight tools
├── docker-compose.yml            # GPU passthrough, SYS_ADMIN cap, volume mounts
├── requirements.txt              # tabulate, numpy, ninja
├── .gitignore
├── .dockerignore
│
├── kernels/                      # Custom CUDA kernels (PyTorch C++ Extensions)
│   ├── __init__.py               # Package imports for cuda_kernels.*
│   ├── setup.py                  # Build config: 4 CUDAExtension modules, -O3 + fast_math
│   ├── vector_add.cu             # Memory-bound: simple + float4 vectorized (92 lines)
│   ├── relu.cu                   # 4 variants: fmaxf, branchless, inplace, vec4 (143 lines)
│   ├── matmul.cu                 # Naive, tiled (32×32 shared mem), double-buffered (237 lines)
│   ├── memory_patterns.cu        # Coalesced/strided/row/col/misaligned/bank conflicts (275 lines)
│   └── gemm/                     # Advanced GEMM implementations
│       ├── gemm_cuda.cu          # Custom GEMM kernel
│       ├── gemm_cublas.cu        # cuBLAS wrapper
│       ├── gemm_cuda.cuh         # Header
│       ├── benchmark.cu          # GEMM benchmarking
│       └── Profile.md            # GEMM profiling guide
│
├── experiments/                  # Profiling & comparison scripts (each runnable standalone)
│   ├── __init__.py               # Package marker
│   ├── 01_vector_add.py          # PyTorch vs CUDA bandwidth comparison (155 lines)
│   ├── 02_relu.py                # Element-wise operation variants (201 lines)
│   ├── 03_matmul.py              # cuBLAS vs naive vs tiled performance (198 lines)
│   ├── 04_memory_patterns.py     # Memory coalescing impact analysis (242 lines)
│   ├── 05_overhead_analysis.py   # Kernel launch + Python overhead (304 lines)
│   └── 06_tensor_pipeline_parallelism.py  # TP/PP distributed demos (628 lines)
│
├── utils/                        # Core profiling utilities
│   ├── __init__.py               # Exports: GPUProfiler, benchmark, print_comparison_table
│   └── profiler.py               # CUDA event timing, bandwidth/FLOPS calc (318 lines)
│
├── scripts/                      # Shell & Python environment tools
│   ├── run_profiler.sh           # Nsight Systems/Compute command wrappers (156 lines)
│   ├── env_check.sh              # Bash environment validation (252 lines)
│   └── check_env.py              # Python/CUDA/GPU verification (278 lines)
│
├── output/                       # Profiler reports (.nsys-rep, .ncu-rep) — Docker volume mount
│   └── .gitkeep
│
├── README.md                     # Project overview
├── CLAUDE.md                     # Development guide & project conventions
├── EXPERIMENTS.md                # Experiment methodology
├── GPU_ARCH.md                   # GPU architecture concepts (~40KB)
└── FLOATING_POINT_PRECISION.md   # FP precision documentation
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Docker Container (nvcr.io/nvidia/pytorch:25.04) │
│                                                  │
│  ┌──────────────┐    ┌────────────────────────┐  │
│  │ experiments/  │───>│ utils/profiler.py      │  │
│  │ 01..06.py     │    │ GPUProfiler            │  │
│  └──────┬───────┘    │ CUDA event timing      │  │
│         │            │ bandwidth/FLOPS calc    │  │
│         ▼            └────────────────────────┘  │
│  ┌──────────────┐                                │
│  │ kernels/     │  Built via: pip install -e     │
│  │ cuda_kernels │  PyTorch C++ Extension (pybind)│
│  │ .vector_add  │                                │
│  │ .relu        │                                │
│  │ .matmul      │                                │
│  │ .memory_pat  │                                │
│  └──────────────┘                                │
│                                                  │
│  ┌──────────────┐    ┌────────────────────────┐  │
│  │ scripts/     │    │ output/                │  │
│  │ profiling &  │───>│ .nsys-rep, .ncu-rep    │  │
│  │ env checks   │    │ (view in Windows GUI)  │  │
│  └──────────────┘    └────────────────────────┘  │
└─────────────────────────────────────────────────┘
         │ nvidia-runtime
         ▼
    RTX 4060 Laptop (SM89, 8GB GDDR6)
```

## Experiment Progression

| # | Experiment | Category | Key Concept |
|---|-----------|----------|-------------|
| 01 | Vector Add | Memory-bound | Bandwidth efficiency baseline |
| 02 | ReLU | Memory-bound | Branching vs branchless, vectorization |
| 03 | MatMul | Compute-bound | Shared memory tiling, double buffering |
| 04 | Memory Patterns | Memory-bound | Coalescing, striding, bank conflicts |
| 05 | Overhead | Framework | Kernel launch cost, Python overhead |
| 06 | Parallelism | Distributed | Tensor parallel, pipeline parallel |

## Target Hardware

- **GPU**: RTX 4060 Laptop — Ada Lovelace (SM89)
- **Memory**: 8GB GDDR6, 256 GB/s theoretical bandwidth
- **Compute**: ~15 TFLOPS FP32
- **CUDA Arch**: `TORCH_CUDA_ARCH_LIST=8.9`
