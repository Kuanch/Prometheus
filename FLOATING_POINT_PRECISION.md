# NVIDIA GPU Floating-Point Precision Hardware Architecture

Comprehensive research on how NVIDIA GPUs handle different floating-point precisions (FP8, FP16, BF16, TF32, FP32, FP64) at the hardware level.

---

## Table of Contents

1. [CUDA Core Behavior Per Precision](#1-cuda-core-behavior-per-precision)
2. [Tensor Core Throughput by Precision](#2-tensor-core-throughput-by-precision)
3. [Memory Representation](#3-memory-representation)
4. [Memory Bandwidth Impact](#4-memory-bandwidth-impact)
5. [Hardware Data Paths](#5-hardware-data-paths)
6. [Tensor Core Matrix Shapes (WMMA)](#6-tensor-core-matrix-shapes-wmma)
7. [Mixed Precision Implementation](#7-mixed-precision-implementation)
8. [RTX 4060 Specifications](#8-rtx-4060-specifications)

---

## 1. CUDA Core Behavior Per Precision

### FP32 Operations

**Standard CUDA Core Configuration:**
- **Volta (GV100)**: 64 FP32 cores per SM
- **Turing**: 64 FP32 cores per SM
- **Ampere (A100)**: 64 FP32 cores per SM (with dual FP32/INT32 capability)
- **Ada Lovelace**: 128 FP32 cores per SM

**Key Architecture Evolution:**
- **Ampere innovation**: The two data paths of Turing are still present, and one of them is still dedicated to FP32, but the other can now be used for either FP32 or INT32, depending on what is in demand. This flexible approach allows better resource utilization.
- **Ada Lovelace**: Devices of compute capability 8.9 have **2x more FP32 operations per cycle per SM** than devices of compute capability 8.0 (Ampere).

### FP64 Operations

**CUDA Core FP64 Capability:**
- **Volta (GV100)**: 32 FP64 cores per SM (1:2 FP64:FP32 ratio)
- **Turing**: 2 FP64 cores per SM (1:32 ratio)
- **Ampere (A100 data center)**: 32 FP64 cores per SM
- **Ampere (GA10x consumer)**: 1:64 FP64:FP32 throughput ratio
- **Ada Lovelace (consumer)**: 1:64 FP64:FP32 throughput ratio
- **Blackwell**: 1:64 FP64:FP32 throughput ratio

**Performance Characteristics:**
- The FP64 TFLOP rate is **1/64th the TFLOP rate of FP32** operations on consumer GPUs (GA10x, Ada, Blackwell).
- Small number of FP64 hardware units are included to ensure any programs with FP64 code operate correctly.
- **Theoretical limit**: For compute-bound algorithms (GEMM, FFT), the theoretical best case for FP64 performance is **1:2 FP32**, simply because it involves computing with double the number of bits. However, most consumer GPUs don't achieve this ratio due to limited FP64 hardware.

### FP16 Operations

**Packed FP16 (FP16x2) Support:**
- NVIDIA CUDA cores can process **1 FP32 or 2 FP16 operations per clock cycle** through FP16x2 cores.
- Peak FP16 throughput is attained by using a **paired operation** to perform two FP16 instructions per core simultaneously.
- To be eligible for the paired operation, operands must be stored in a **half2 vector type**.

**Architecture-Specific FP16 Throughput:**
- **Ampere (A100)**: CUDA core can compute **4 FP16 data per clock** (4x FP16:FP32 ratio)
- **Hopper (H100)**: CUDA core can compute **2 FP16 data per clock** (2x FP16:FP32 ratio)

**Native Support:**
- FP16 can be processed through both **CUDA cores** (via packed FP16x2) and **Tensor Cores**.
- Tensor Cores provide significantly higher throughput for matrix operations.

---

## 2. Tensor Core Throughput by Precision

### Tensor Core Evolution by Generation

#### 1st Generation (Volta)
- **8 Tensor Cores per SM**
- **64 FP16/FP32 FMA operations per clock per Tensor Core**
- **512 FMA operations per clock per SM total**
- Performs 4x4 FP16 matrix multiply-add with FP32 accumulate

#### 2nd Generation (Turing)
- **8 Tensor Cores per SM**
- **64 FP16/FP32 FMA operations per clock per Tensor Core**
- **1024 FMA operations per clock per SM total**

#### 3rd Generation (Ampere)
- **4 Tensor Cores per SM (A100)**
- **256 FP16/FP32 FMA operations per clock per Tensor Core**
- **1024 dense FP16/FP32 FMA operations per clock per SM**
- Added support for: **BF16, TF32, FP64**
- **2x computation horsepower per Tensor Core** compared to Volta/Turing
- Introduced **sparsity acceleration** (2x throughput on sparse matrices)

#### 4th Generation (Ada Lovelace)
- **4 Tensor Cores per SM**
- Added support for: **FP8 (E4M3, E5M2)**
- FP8 provides **1.45x speedup** compared to FP16
- **Sparsity acceleration** enabled across all precisions

#### 4th Generation (Hopper)
- **4 Tensor Cores per SM**
- **Up to 6x faster chip-to-chip** compared to A100
- **2x the MMA computational rates** on equivalent data types vs A100
- **4x the rate using FP8** vs FP16/BF16
- FP8 Tensor Cores support **FP32 and FP16 accumulators**

### Precision-Specific TFLOPS Performance

#### NVIDIA H100 (Hopper - 4th Gen Tensor Cores)
| Precision | TFLOPS |
|-----------|--------|
| FP64 | 60 |
| FP32 | 60 |
| TF32 | 1,000 |
| BF16 | 2,000 |
| FP16 | 2,000 |
| FP8 | 4,000 |

**Key insights:**
- FP8 provides **2x throughput** over FP16/BF16
- TF32 provides **16.7x throughput** over FP32
- FP16/BF16 provides **33.3x throughput** over FP32

#### NVIDIA A100 (Ampere - 3rd Gen Tensor Cores)
| Precision | TFLOPS |
|-----------|--------|
| FP64 | 19.5 |
| FP32 | 19.5 |
| TF32 | 156 |
| BF16 | 312 |
| FP16 | 312 |

#### NVIDIA L4 (Ada Lovelace - Data Center)
| Precision | TFLOPS |
|-----------|--------|
| TF32 | 120 |
| FP16 | 242 |
| BF16 | 242 |
| FP8 | 485 |

---

## 3. Memory Representation

### Bit Layout Summary

| Format | Total Bits | Sign | Exponent | Mantissa | IEEE 754 |
|--------|-----------|------|----------|----------|----------|
| FP64 | 64 | 1 | 11 | 52 | Yes |
| FP32 | 32 | 1 | 8 | 23 | Yes |
| TF32 | 19* | 1 | 8 | 10 | No |
| BF16 | 16 | 1 | 8 | 7 | No |
| FP16 | 16 | 1 | 5 | 10 | Yes |
| FP8 E4M3 | 8 | 1 | 4 | 3 | No |
| FP8 E5M2 | 8 | 1 | 5 | 2 | No |

*TF32 uses 19 bits stored in 32-bit containers

### Detailed Format Specifications

#### FP64 (Double Precision)
```
[1 sign][11 exponent][52 mantissa]
Range: ~10^-308 to ~10^308
Precision: ~15-17 decimal digits
```

#### FP32 (Single Precision)
```
[1 sign][8 exponent][23 mantissa]
Range: ~10^-38 to ~10^38
Precision: ~7 decimal digits
```

#### TF32 (TensorFloat-32)
```
[1 sign][8 exponent][10 mantissa] (19 bits total, stored in 32-bit)
Range: Same as FP32 (~10^-38 to ~10^38)
Precision: ~3 decimal digits
```

**Key Characteristics:**
- Combines FP32's 8-bit exponent with FP16's 10-bit mantissa
- **Same dynamic range as FP32** (can represent same numeric range)
- Reduced precision but sufficient for AI workloads
- First implemented in Ampere architecture
- All studied DL workloads match FP32 accuracy with no hyperparameter changes

#### BF16 (Brain Float 16)
```
[1 sign][8 exponent][7 mantissa]
Range: Same as FP32 (~10^-38 to ~10^38)
Precision: ~2-3 decimal digits
```

**Key Characteristics:**
- **Truncated IEEE 754 FP32** (preserves exponent bits, reduces mantissa)
- Matches FP32's **dynamic range** but with reduced precision
- Fast conversion between FP32 and BF16
- Less prone to overflow/underflow than FP16

#### FP16 (Half Precision)
```
[1 sign][5 exponent][10 mantissa]
Range: ~10^-5 to 65,504
Precision: ~3 decimal digits
```

**Key Characteristics:**
- IEEE 754 compliant
- Lower precision AND range than BF16
- More prone to overflow/underflow
- Higher mantissa precision than BF16 within its range

#### FP8 E4M3 (4 Exponent, 3 Mantissa)
```
[1 sign][4 exponent][3 mantissa]
Range: +/-448 and NaN
Exponent bias: 7
```

**Key Characteristics:**
- **Does NOT represent infinities**
- Uses only **two bit patterns for NaN** to increase dynamic range
- Best for **forward pass**: weights and activation tensors requiring more precision
- Less dynamic range, more precision than E5M2

#### FP8 E5M2 (5 Exponent, 2 Mantissa)
```
[1 sign][5 exponent][2 mantissa]
Range: +/-57,344, +/-inf, and NaN
Exponent bias: 15
```

**Key Characteristics:**
- **Represents infinities and NaNs** (IEEE-like)
- Wider dynamic range, less precision than E4M3
- Best for **backward pass**: gradient tensors requiring higher dynamic range
- 128x wider range than E4M3

### Range and Precision Tradeoffs

| Format | Dynamic Range | Relative Precision | Best Use Case |
|--------|---------------|-------------------|---------------|
| FP64 | Widest | Highest | Scientific computing |
| FP32 | Wide | High | General purpose |
| TF32 | Wide (= FP32) | Medium | AI training/inference |
| BF16 | Wide (= FP32) | Medium-Low | AI training |
| FP16 | Narrow | Medium | AI inference |
| FP8 E5M2 | Medium | Low | Gradients (backward) |
| FP8 E4M3 | Narrow | Medium-Low | Activations (forward) |

---

## 4. Memory Bandwidth Impact

### Theoretical Bandwidth Multipliers

Switching to lower precision formats reduces data size, which can translate to effective bandwidth improvements:

| Precision | Bytes per Value | Bandwidth Multiplier vs FP32 |
|-----------|----------------|------------------------------|
| FP64 | 8 | 0.5x (half speed) |
| FP32 | 4 | 1x (baseline) |
| TF32 | 4* | 1x (same bandwidth as FP32) |
| BF16 | 2 | 2x |
| FP16 | 2 | 2x |
| FP8 | 1 | 4x |

*TF32 is stored in 32-bit containers

### Real-World Bandwidth Considerations

**Does 2x smaller data = 2x effective bandwidth?**

**YES, for memory-bound operations:**
- When throughput is limited by memory bandwidth (not compute)
- FP16 (2 bytes) vs FP32 (4 bytes) → **2x effective bandwidth**
- FP8 (1 byte) vs FP32 (4 bytes) → **4x effective bandwidth**

**Factors affecting real-world performance:**
1. **Memory access patterns**: Sequential vs random access
2. **Cache utilization**: L1/L2 cache hit rates
3. **Tensor Core utilization**: Compute may become the bottleneck
4. **Thermal throttling**: Sustained workloads may reduce clocks
5. **Framework overhead**: Software overhead in data handling

**Effective Bandwidth Formula:**
```
Effective Bandwidth = Theoretical × Utilization Factor
Utilization Factor = 0.7 to 0.9 (depending on access patterns)
```

### Throughput Gains from Lower Precision

**NVIDIA Official Metrics:**
- **FP8 vs FP16**: 2x throughput improvement
  - H100: FP8 halves data storage requirements and **doubles throughput** vs FP16/BF16
  - Hopper delivers **2x MMA rates** on equivalent data types, **4x rate using FP8**

- **FP16 vs FP32**: Up to 2x throughput
  - Depends on whether compute or memory is the bottleneck

- **FP4 vs FP8**: ~1.8x throughput
  - NVFP4 reduces memory footprint by **3.5x relative to FP16**, **1.8x vs FP8**

**Blackwell vs Hopper:**
- B300/B200 throughput is **over 2x in TF32, FP16, and FP8** compared to H200

**H100 vs A100:**
- H100 offers **3x FP8 performance over FP16**

---

## 5. Hardware Data Paths

### SM (Streaming Multiprocessor) Architecture Evolution

#### Ampere Architecture (A100)

**SM Composition:**
- **64 FP32 cores** (with dual FP32/INT32 capability)
- **32 FP64 cores** (data center variant)
- **64 INT32 cores** (can also do FP32)
- **4 Tensor Cores** (3rd generation)
- **192 KB combined shared memory and L1 cache**
- **64K 32-bit registers**
- **Maximum 64 concurrent warps**

**Dual Data Path Innovation:**
- One path: Dedicated **FP32**
- Other path: **FP32 OR INT32** (flexible, demand-driven)
- Allows better resource utilization for mixed workloads

#### Ada Lovelace Architecture (AD10x)

**SM Composition (Full SM):**
- **128 FP32 CUDA cores** (2x Ampere per SM)
- **4 Tensor Cores** (4th generation)
- **1 RT Core** (3rd generation)
- **4 Texture Units**
- **256 KB Register File**
- **128 KB L1/Shared Memory**

**Performance:**
- **2x more FP32 operations per cycle per SM** than Ampere (compute capability 8.0)

#### Volta Architecture (GV100)

**SM Composition:**
- **64 FP32 cores**
- **32 FP64 cores**
- **64 INT32 cores**
- **8 Tensor Cores** (1st generation)

### FP16 vs FP32 Execution Units

**Separate or Shared Units?**

**Short Answer:** It depends on the architecture and operation type.

#### CUDA Cores (Standard Math)
- **FP32 units handle both FP32 and FP16** (via packed FP16x2)
- **No separate FP16 execution units** in CUDA cores
- FP16x2 achieves **2x throughput** by processing 2 FP16 values in parallel
- Must use **half2 vector type** to enable paired operations

**Architecture-Specific:**
- **Pascal onwards**: FP16x2 support (2 FP16 ops per FP32 core)
- **Ampere A100**: 4x FP16:FP32 ratio via enhanced CUDA cores
- **Hopper H100**: 2x FP16:FP32 ratio

#### Tensor Cores (Matrix Math)
- **Completely separate execution units** from CUDA cores
- Dedicated hardware for matrix multiply-accumulate
- Optimized for specific precision formats:
  - Volta/Turing: FP16 input, FP32 accumulate
  - Ampere: Added BF16, TF32, FP64
  - Ada/Hopper: Added FP8 (E4M3, E5M2)

### Data Type Support by Architecture

| Architecture | CUDA Core Types | Tensor Core Types |
|--------------|----------------|-------------------|
| Volta | FP32, FP64, INT32, FP16x2 | FP16 (input), FP32 (accumulate) |
| Turing | FP32, FP64, INT32, FP16x2 | FP16 (input), FP32 (accumulate) |
| Ampere | FP32, FP64, INT32, FP16x2 | FP16, BF16, TF32, FP64, INT8, INT4 |
| Ada Lovelace | FP32, FP64, INT32, FP16x2 | FP8, FP16, BF16, TF32, INT8, INT4 |
| Hopper | FP32, FP64, INT32, FP16x2 | FP8, FP16, BF16, TF32, FP64, INT8 |

---

## 6. Tensor Core Matrix Shapes (WMMA)

### WMMA Overview

**WMMA = Warp Matrix Multiply-Accumulate**
- Warp-wide operation: **All 32 threads in a warp must cooperate**
- Executes in lockstep (synchronous execution required)
- Leverages Tensor Cores for accelerated matrix operations
- Computes: **D = A × B + C**

### Common Matrix Shapes

#### m16n16k16 (Volta/Turing/Ampere)
```
A: 16×16 matrix
B: 16×16 matrix
C: 16×16 matrix
D: 16×16 matrix (output)

Each warp computes a 16×16×16 GEMM
```

**Supported Precisions:**
- **FP16**: A, B in FP16; C, D in FP16 or FP32
- **BF16**: A, B in BF16; C, D in FP32
- Introduced in CUDA 9 for Volta

#### m16n16k8 (Ampere onwards)
```
A: 16×8 matrix
B: 8×16 matrix
C: 16×16 matrix
D: 16×16 matrix (output)
```

**Supported Precisions:**
- **TF32**: A, B in TF32; C, D in FP32
- **BF16**: A, B in BF16; C, D in FP32
- **FP16**: A, B in FP16; C, D in FP16 or FP32

#### m8n8k4 (Ampere onwards)
```
A: 8×4 matrix
B: 4×8 matrix
C: 8×8 matrix
D: 8×8 matrix (output)
```

**Supported Precisions:**
- **FP64**: A, B, C, D all in FP64
- Introduced in CUDA 11 with Ampere

#### m8n8k32 (Volta/Turing/Ampere)
```
A: 8×32 matrix
B: 32×8 matrix
C: 8×8 matrix
D: 8×8 matrix (output)
```

**Supported Precisions:**
- **INT8**: A, B in INT8; C, D in INT32

### WMMA Shape Summary by Precision

| Precision | Common Shapes | Introduced |
|-----------|--------------|------------|
| FP16 | m16n16k16, m16n16k8 | CUDA 9 (Volta) |
| BF16 | m16n16k16, m16n16k8 | CUDA 11 (Ampere) |
| TF32 | m16n16k8 | CUDA 11 (Ampere) |
| FP64 | m8n8k4 | CUDA 11 (Ampere) |
| INT8 | m8n8k32, m8n8k16 | CUDA 9 (Volta) |
| FP8 | m16n16k16 (likely) | CUDA 12 (Hopper) |

### Architecture-Specific Notes

**Volta/Turing (1st/2nd Gen):**
- Each Tensor Core performs **4×4 matrix operation**
- 8 Tensor Cores per SM work together for 16×16 tile

**Ampere (3rd Gen):**
- Each Tensor Core performs **larger matrix operations**
- 4 Tensor Cores per SM (but each 4x more powerful)

**Hopper (4th Gen):**
- Introduced **WGMMA** (Warpgroup Matrix Multiply-Accumulate)
- Operates on warpgroups (4 warps = 128 threads)
- Further optimized for transformer models

---

## 7. Mixed Precision Implementation

### Hardware Implementation

**FP16 Compute with FP32 Accumulation:**

```
Operation: D = A × B + C
Inputs:  A (FP16), B (FP16)
Compute: A × B performed in FP16
Accumulator: C, D in FP32
```

**How it works:**
1. **Multiplication**: Two 4×4 FP16 matrices multiplied → produces FP16 products
2. **Conversion**: FP16 products converted to FP32 (lossless, expands mantissa)
3. **Accumulation**: FP32 products added to FP32 accumulator
4. **Output**: Result can be FP16 or FP32

### Tensor Core Mixed Precision

#### Volta/Turing (1st/2nd Gen)
```
Input:  FP16 × FP16
Multiply: FP16 precision
Accumulate: FP32 precision
Output: FP16 or FP32
```
- **64 FMA operations per Tensor Core per clock**
- Accumulation to FP32 is **key differentiator** from other architectures

#### Ampere (3rd Gen)
```
Supported combinations:
- FP16 × FP16 → FP32 accumulate
- BF16 × BF16 → FP32 accumulate
- TF32 × TF32 → FP32 accumulate
- FP64 × FP64 → FP64 accumulate
```
- **256 FMA operations per Tensor Core per clock**

#### Hopper (4th Gen)
```
New FP8 combinations:
- FP8 × FP8 → FP32 accumulate
- FP8 × FP8 → FP16 accumulate
- All previous Ampere combinations
```
- **FP8 supports both FP32 and FP16 accumulators**
- **E4M3** for forward pass (activations, weights)
- **E5M2** for backward pass (gradients)

### Why Mixed Precision?

**Advantages:**
1. **Speed**: Lower precision multiply is faster (less bits to process)
2. **Accuracy**: Higher precision accumulate prevents rounding errors
3. **Numerical Stability**: FP32 accumulator reduces error accumulation
4. **Memory Efficiency**: Store weights/activations in lower precision
5. **Bandwidth**: Transfer less data (FP16 vs FP32)

**Performance Gains:**
- **Up to 3x overall speedup** for arithmetically intense models
- FP16-TC (Tensor Core) is **more accurate** than homogeneous FP16
- Maintains **high accuracy** while accelerating training/inference

### Transformer Engine (Hopper)

**Dynamic Precision Management:**
- Intelligently manages FP8 ↔ 16-bit conversions
- **Automatically handles re-casting and scaling** between FP8 and 16-bit in each layer
- **Up to 9x faster AI training** vs A100
- **Up to 30x faster AI inference** on large language models vs A100

---

## 8. RTX 4060 Specifications

### GPU Architecture
- **Architecture**: Ada Lovelace (4th Gen Tensor Cores)
- **GPU Die**: AD107
- **Process**: 5nm (4N NVIDIA Custom)
- **Compute Capability**: 8.9

### Core Counts
- **CUDA Cores**: 3,072
- **Tensor Cores**: 96 (4th generation)
- **RT Cores**: 24 (3rd generation)
- **Texture Units**: 96
- **ROPs**: 48

### SM Configuration
- **Total SMs**: 24
- **CUDA Cores per SM**: 128 (full Ada Lovelace SM design)
- **Tensor Cores per SM**: 4
- **RT Cores per SM**: 1

### Memory Specifications
- **Memory**: 8 GB GDDR6
- **Memory Bus**: 128-bit
- **Memory Bandwidth**: 272 GB/s

### CUDA Core Performance (TFLOPS)
| Precision | TFLOPS |
|-----------|--------|
| FP16 (half) | 15.11 |
| FP32 (float) | 15.11 |

### Tensor Core Performance (TFLOPS/TOPS)

**Dense Operations:**
| Precision | TFLOPS/TOPS |
|-----------|-------------|
| FP8 | 60.46 |
| FP16 | 30.23 |
| BF16 | 30.23 |
| INT8 | 120.91 TOPS |

**Sparse Operations (with Sparsity Acceleration):**
| Precision | TFLOPS/TOPS |
|-----------|-------------|
| FP8 | 120.91 |
| FP16 | 60.46 |
| BF16 | 60.46 |
| INT8 | 241.83 TOPS |

### Power and Thermal
- **TDP**: 115W
- **Power Efficiency**: Exceptional for 1080p gaming and light AI/ML inference

### Use Cases
- **1080p gaming**: Primary target
- **Light AI/ML inference**: FP8/FP16 workloads
- **Entry-level content creation**: Video editing, rendering

### Supported Precision Formats
- FP32, FP16, BF16, TF32
- **FP8** (E4M3, E5M2) - 4th Gen Tensor Core feature
- INT8, INT4

### Key Features
- **4th Generation Tensor Cores**: FP8 support with structured sparsity
- **3rd Generation RT Cores**: Enhanced ray tracing performance
- **DLSS 3.5**: AI-powered upscaling and frame generation
- **AV1 Encode/Decode**: Dual encoders

---

## Sources and References

### NVIDIA Official Documentation
1. [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) - NVIDIA Technical Blog
2. [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) - NVIDIA Technical Blog
3. [Floating-Point 8: An Introduction to Efficient, Lower-Precision AI Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) - NVIDIA Technical Blog
4. [Using Tensor Cores for Mixed-Precision Scientific Computing](https://developer.nvidia.com/blog/tensor-cores-mixed-precision-scientific-computing/) - NVIDIA Technical Blog
5. [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/) - NVIDIA Technical Blog
6. [What is the TensorFloat-32 Precision Format?](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) - NVIDIA Blog
7. [Accelerating AI Training with NVIDIA TF32 Tensor Cores](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/) - NVIDIA Technical Blog
8. [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
9. [NVIDIA Ada GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
10. [NVIDIA Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)
11. [NVIDIA Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)

### Architecture Whitepapers
12. [NVIDIA Ada GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
13. [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)
14. [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

### Product Specifications
15. [NVIDIA H100 Datasheet](https://www.megware.com/fileadmin/user_upload/LandingPage%20NVIDIA/nvidia-h100-datasheet.pdf)
16. [NVIDIA L4 Tensor Core GPU Datasheet](https://www.cisco.com/c/dam/en/us/products/collateral/servers-unified-computing/ucs-c-series-rack-servers/nvidia-l4-gpu.pdf)
17. [GeForce RTX 4060 Ti & 4060 Graphics Cards](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/) - NVIDIA Official
18. [H100 GPU](https://www.nvidia.com/en-us/data-center/h100/) - NVIDIA Official

### Community and Technical Resources
19. [Ampere (microarchitecture) - Wikipedia](https://en.wikipedia.org/wiki/Ampere_(microarchitecture))
20. [Ada Lovelace (microarchitecture) - Wikipedia](https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture))
21. [TensorFloat-32 - Wikipedia](https://en.wikipedia.org/wiki/TensorFloat-32)
22. [Understanding FP64, FP32, FP16, BFLOAT16, TF32, FP8 Formats](https://jeffreytse.net/computer/2024/12/09/understanding-the-fp64-fp32-fp16-bfloat16-tf32-fp8-formats.html)
23. [FP64, FP32, FP16, BFLOAT16, TF32, and other members of the ZOO](https://moocaholic.medium.com/fp64-fp32-fp16-bfloat16-tf32-and-other-members-of-the-zoo-a1ca7897d407) - Medium
24. [NVIDIA GeForce RTX 4060 AI Performance and Hardware Specs](https://www.waredb.com/processor/nvidia-geforce-rtx-4060) - WareDB
25. [RTX 3060 Ti vs RTX 4060 Comparison](https://www.bestgpusforai.com/gpu-comparison/3060-ti-vs-4060)
26. [Using FP8 and FP4 with Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
27. [NVIDIA Tensor Core Evolution: From Volta To Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell) - SemiAnalysis
28. [Comparing Blackwell vs Hopper](https://www.exxactcorp.com/blog/hpc/comparing-nvidia-tensor-core-gpus) - Exxact Blog
29. [What is FP64, FP32, FP16? Defining Floating Point](https://www.exxactcorp.com/blog/hpc/what-is-fp64-fp32-fp16) - Exxact Blog
30. [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) - NVIDIA Technical Blog

### Research Papers and Academic Sources
31. [FP8 Formats for Deep Learning](https://arxiv.org/pdf/2209.05433) - ArXiv
32. [Dissecting Tensor Cores via Microbenchmarks](https://arxiv.org/pdf/2206.02874) - ArXiv
33. [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/html/2402.13499v1) - ArXiv
34. [Harnessing GPU Tensor Cores for Fast FP16 Arithmetic](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf)

### Forums and Developer Discussions
35. [Separate CUDA Core pipeline for FP16 and FP32?](https://forums.developer.nvidia.com/t/separate-cuda-core-pipeline-for-fp16-and-fp32/302018) - NVIDIA Developer Forums
36. [Theoretical TFLOPS for FP16, BF16 and TF32](https://forums.developer.nvidia.com/t/theoretical-tflops-for-fp16-bf16-and-tf32-for-tensor-and-non-tensor/218102) - NVIDIA Developer Forums
37. [About tensor core's flops/clk and wmma shape?](https://forums.developer.nvidia.com/t/about-tensor-cores-flops-clk-and-wmma-shape/270263) - NVIDIA Developer Forums
38. [NVIDIA Tensor Core Programming](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/) - Lei Mao's Log Book
39. [GPU Architecture Deep Dive: Nvidia Ada Lovelace, AMD RDNA 3 and Intel Arc Alchemist](https://www.techspot.com/article/2570-gpu-architectures-nvidia-intel-amd/) - TechSpot

---

## Key Takeaways

### CUDA Core Behavior
1. **FP32 is the baseline** - 1 operation per core per cycle
2. **FP64 is heavily limited** on consumer GPUs (1:64 ratio) but 1:2 on data center GPUs
3. **FP16 via packed FP16x2** - 2 operations per core per cycle (must use half2 type)
4. **No separate FP16 execution units** - FP32 units handle both via packing

### Tensor Core Throughput
1. **Each generation improves significantly**: 64 → 256 FMA/clock per Tensor Core
2. **FP8 doubles throughput** over FP16/BF16
3. **Mixed precision is standard**: Lower precision multiply, higher precision accumulate
4. **Sparsity can double performance** on sparse matrices (2:4 structured sparsity)

### Memory Impact
1. **FP16 = 2x effective bandwidth** vs FP32
2. **FP8 = 4x effective bandwidth** vs FP32
3. **Effective bandwidth depends on utilization** (0.7-0.9 factor)
4. **TF32 same bandwidth as FP32** (stored in 32-bit containers)

### Precision Selection
1. **FP32**: General-purpose, baseline
2. **TF32**: AI training, matches FP32 range, 3 decimal digits precision
3. **BF16**: AI training, matches FP32 range, less overflow than FP16
4. **FP16**: AI inference, narrower range, more mantissa bits than BF16
5. **FP8 E4M3**: Forward pass (weights, activations), more precision
6. **FP8 E5M2**: Backward pass (gradients), more range
7. **FP64**: Scientific computing, not for AI

---

**Document Version**: 1.0
**Last Updated**: 2026-02-07
**Research Focus**: NVIDIA GPU hardware-level floating-point precision handling
