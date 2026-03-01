# cuBLAS SGEMM Profiling — RTX 4060 Laptop (Ada Lovelace, SM89)

## Setup

- **Kernel**: `gemm.cu` — cuBLAS SGEMM, M=N=K=4096, FP32, 5 warmup + 20 timed iterations
- **GPU**: RTX 4060 Laptop (8GB GDDR6, ~256 GB/s BW, ~15 TFLOPS FP32 peak)
- **Environment**: Docker (nvidia/pytorch base), WSL2

## Commands

```bash
# Build and run benchmark
docker compose run --rm gpu-lab bash -c \
  "nvcc -o /output/gemm /workspace/kernels/gemm.cu -lcublas -O2 && /output/gemm"

# Nsight Systems (timeline + CUDA API trace)
docker compose run --rm gpu-lab \
  nsys profile -o /output/gemm /output/gemm

# Nsight Systems with GPU metrics (ad10x set — see caveat in Debugging)
docker compose run --rm gpu-lab nsys profile \
  --gpu-metrics-set=ad10x --gpu-metrics-devices=0 \
  --gpu-metrics-frequency=10000 --cuda-memory-usage=true \
  -o /output/gemm_full --force-overwrite=true /output/gemm

# Nsight Compute (per-kernel analysis)
docker compose run --rm gpu-lab \
  ncu --set basic -o /output/gemm_ncu /output/gemm
```

## Output Files

Open in Windows Nsight GUI:
```
\\wsl.localhost\Ubuntu\home\sixigma\Prometheus\output\gemm.nsys-rep
\\wsl.localhost\Ubuntu\home\sixigma\Prometheus\output\gemm_full.nsys-rep
\\wsl.localhost\Ubuntu\home\sixigma\Prometheus\output\gemm_ncu.ncu-rep
```

---

## Benchmark Results

```
avg latency : ~17.4 ms
TFLOPS      : ~7.89  (52% of ~15 TFLOPS peak)
memory BW   : ~11.5 GB/s  (4.5% of ~256 GB/s peak)
```

### Kernel Launched

cuBLAS selected `ampere_sgemm_64x64_nn` — a 64x64 tile SGEMM kernel. Despite "ampere" in the name, cuBLAS reuses Ampere-era kernels on Ada Lovelace when they're already optimal for the problem shape.

## Analysis

### Why 52% compute utilization?

SGEMM at M=N=K=4096 has arithmetic intensity = 2*4096 / (2*4) = 1024 FLOPs/byte — firmly compute-bound. 52% of peak FP32 is reasonable for cuBLAS on a laptop GPU because:

1. **Clock throttling** — laptop GPUs run below max boost under sustained load (power/thermal limits). The 15 TFLOPS peak assumes max boost clock, which isn't sustained.
2. **Tile overhead** — 64x64 tiles across a 4096x4096 matrix = 64x64 = 4096 tiles. Some SMs finish before others (tail effect).
3. **Memory latency** — even compute-bound kernels stall waiting for shared memory bank conflicts and L2 misses on tile loads.

For a laptop 4060, 52% sustained is a solid result. Desktop GPUs with higher power budgets typically hit 60-70%.

### Why memory BW is only 4.5% of peak?

This is **expected and correct** — it does NOT mean the kernel is memory-inefficient. SGEMM is compute-bound (arithmetic intensity ~1024), so the GPU reuses data heavily from shared memory/registers. Low DRAM bandwidth means the kernel is doing its job: computing more, fetching less. The 11.5 GB/s reflects only the initial tile loads from global memory, which is exactly what a well-optimized GEMM should look like.

### What ncu report reveals

The `gemm_ncu.ncu-rep` contains per-kernel metrics across 8 counter passes:
- **SM throughput** — how busy the SMs are with compute vs stalled
- **Memory workload** — L1/L2 hit rates, shared memory utilization
- **Occupancy** — warps active vs theoretical max
- **Warp stall reasons** — where cycles are lost (barrier, memory, etc.)

Key things to look for in the report:
- Achieved occupancy vs theoretical — cuBLAS typically hits 50-75% occupancy for SGEMM
- Dominant warp stall reason — expect `stall_barrier` (tiles synchronizing) and `stall_mio` (shared memory)
- L2 hit rate — should be high since tiles are reused

---

## Debugging Log

### Docker Setup (resolved)
- Docker Desktop fixed — image pulls and builds work
- `gemm.cu` compiled and benchmark ran successfully

### ncu `ERR_NVGPUCTRPERM` (resolved)
`ncu --set full` initially failed with `ERR_NVGPUCTRPERM` in WSL2.

**Attempted fixes:**
- `EnablePerfOsDeviceCounters=1` registry key
- Full Windows reboot
- `NVIDIA_DRIVER_CAPABILITIES=all` in docker-compose.yml
- `ncu --clock-control none`
- `ncu --set basic`

Initially none worked. Root cause was WSL2's WDDM driver not exposing hardware SM counters. Eventually resolved (likely driver update or registry change taking effect after restart). `ncu --set basic` now works with full 8-pass counter collection.

### nsys GPU metrics all zeros (WSL2 limitation)
`nsys profile --gpu-metrics-set=ad10x` runs without error but all GPU metric rows (GPU Active, SMs Active, DRAM BW) show **zeros**.

**Root cause**: nsys GPU metrics use **continuous PMU sampling** via the Linux kernel-mode driver (`nvidia.ko`). WSL2 uses the WDDM driver path which doesn't expose the PMU sampling interface. nsys degrades gracefully — no error, just zero values.

**ncu works because** it uses a different mechanism: **replay-based** counter collection through CUPTI, which re-executes each kernel multiple times to read different counter groups. This works through WDDM.

**Summary of WSL2 profiling support:**

| Tool | Feature | Works? |
|------|---------|--------|
| nsys | CUDA API trace + timeline | Yes |
| nsys | GPU metrics (SM util, DRAM BW) | No — reads zeros |
| ncu  | Per-kernel counters (replay) | Yes |
| nsys | GPU context switch tracing | No — requires kernel module >= 435.17 |
