# Development Guide

GPU architecture learning project with PyTorch and custom CUDA kernels. Compares framework overhead vs raw CUDA performance through profiling experiments.

## Quick Start

```bash
# Build container
docker-compose build

# Build CUDA kernels (required first time)
docker-compose run --rm gpu-lab pip install -e /workspace/kernels

# Run all experiments
docker-compose run --rm gpu-lab ./scripts/run_profiler.sh run-all

# Run specific experiment
docker-compose run --rm gpu-lab python experiments/01_vector_add.py
```

## Environment Check

```bash
docker-compose run --rm gpu-lab ./scripts/env_check.sh
```

## Profiling Commands

```bash
# Nsight Systems (timeline view)
docker-compose run --rm gpu-lab nsys profile -o /output/NAME python experiments/01_vector_add.py

# Nsight Compute (kernel analysis)
docker-compose run --rm gpu-lab ncu --set full --export /output/NAME python experiments/01_vector_add.py

# View reports: open ./output/*.nsys-rep or *.ncu-rep in Windows Nsight GUI
```

## Project Structure

```
GPU_arch_modeling/
├── Dockerfile              # NVIDIA PyTorch base + Nsight tools
├── docker-compose.yml      # GPU passthrough config
├── kernels/
│   ├── vector_add.cu       # Memory-bound: 2 read + 1 write
│   ├── relu.cu             # Memory-bound: 1 read + 1 write
│   ├── matmul.cu           # Compute-bound: naive + tiled + double-buffered
│   ├── memory_patterns.cu  # Coalesced vs strided access patterns
│   └── setup.py            # PyTorch C++ extension build
├── experiments/
│   ├── 01_vector_add.py    # PyTorch vs CUDA bandwidth comparison
│   ├── 02_relu.py          # Element-wise operation variants
│   ├── 03_matmul.py        # cuBLAS vs naive vs tiled performance
│   ├── 04_memory_patterns.py # Memory coalescing impact (10-30x)
│   ├── 05_overhead_analysis.py # Kernel launch + Python overhead
│   ├── 06_llama_inference.py   # Llama 3.1 8B prefill/decode profiling
│   └── 07_gqa_attention.py     # Decoupled GQA ops for Nsight profiling
├── utils/
│   └── profiler.py         # GPUProfiler class, CUDA event timing
├── scripts/
│   ├── run_profiler.sh     # Profiling helper commands
│   ├── env_check.sh        # Environment verification
│   └── check_env.py        # Python/CUDA checks
└── output/                 # Profiler reports (mounted volume)
```

## Key Files

| File | Purpose |
|------|---------|
| `utils/profiler.py` | `GPUProfiler` class with CUDA event timing, bandwidth/FLOPS calculation |
| `kernels/setup.py` | Build command: `pip install -e /workspace/kernels` |
| `scripts/run_profiler.sh` | All profiling commands with `./run_profiler.sh help` |

## Target GPU Specs (RTX 4060 Laptop)

- Architecture: Ada Lovelace (SM89)
- Memory Bandwidth: ~256 GB/s (theoretical)
- FP32 Performance: ~15 TFLOPS (theoretical)
- Memory: 8GB GDDR6

## Common Operations

### Rebuild CUDA kernels after changes
```bash
docker-compose run --rm gpu-lab pip install -e /workspace/kernels --force-reinstall
```

### Interactive shell
```bash
docker-compose run --rm gpu-lab bash
```

### Run with specific GPU
```bash
NVIDIA_VISIBLE_DEVICES=0 docker-compose run --rm gpu-lab python experiments/01_vector_add.py
```

## Troubleshooting

### "CUDA out of memory"
- Reduce tensor sizes in experiments
- Check `nvidia-smi` for other processes using GPU

### "Custom CUDA kernels not available"
```bash
docker-compose run --rm gpu-lab pip install -e /workspace/kernels
```

### Nsight Compute permission denied
- Container needs `--cap-add SYS_ADMIN` (already in docker-compose.yml)
- Or run with `--security-opt seccomp=unconfined`

### Profiler reports not appearing
- Check `./output/` directory
- Ensure volume mount is working: `docker-compose run --rm gpu-lab ls /output`

## Architecture Decisions

1. **PyTorch C++ Extension** (vs raw CUDA): Easy Python interop, automatic memory management
2. **CUDA Events** (vs Python time): Accurate GPU timing without Python overhead
3. **Docker** (vs local install): Reproducible environment, easy Nsight tools setup
4. **Separate kernel files**: Each `.cu` file = one concept, easier to understand

## Metrics Reference

| Metric | Formula | Good Value (RTX 4060) |
|--------|---------|----------------------|
| Memory Bandwidth | bytes / time | >200 GB/s |
| FP32 TFLOPS | flops / time | >10 TFLOPS |
| Bandwidth Efficiency | achieved / theoretical | >80% |
| Kernel Launch Overhead | - | ~5-10 μs |
