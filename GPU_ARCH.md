# NVIDIA GPU Architecture Deep Dive

A comprehensive breakdown of modern NVIDIA GPU architecture, focusing on Ada Lovelace (RTX 40-series) with references to Ampere and Hopper.

---

## 1. The Big Picture — GPU Chip Hierarchy

> **Ref:** [NVIDIA Ada Lovelace Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), Figure 3 "AD102 Full GPU" & Figure 4 "AD102 GPC"; [NVIDIA Ampere GA102 Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), Figure 1 "GA102 Full Chip"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NVIDIA GPU (e.g. AD102)                            │
│                                                                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         ┌─────────┐        │
│  │  GPC 0  │ │  GPC 1  │ │  GPC 2  │ │  GPC 3  │  . . .  │  GPC N  │        │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘         └────┬────┘        │
│       │            │            │            │                   │          │
│  ┌────┴────────────┴────────────┴────────────┴───────────────────┴────┐     │
│  │                        GigaThread Engine                           │     │
│  │              (Global work distribution & scheduling)               │     │
│  └────────────────────────────┬───────────────────────────────────────┘     │
│                               │                                             │
│  ┌────────────────────────────┴───────────────────────────────────────┐     │
│  │                     L2 Cache (Unified)                             │     │
│  │           Ada: 96 MB  │  Hopper: 50 MB  │  Ampere: 6 MB          │       │
│  └────────────────────────────┬───────────────────────────────────────┘     │
│                               │                                             │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌┴─────┐ ┌──────┐          ┌──────┐             │
│  │MemC 0│ │MemC 1│ │MemC 2│ │MemC 3│ │MemC 4│  . . .   │MemC N│             │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘          └──┬───┘             │
│     └────────┴────────┴────────┴────────┴──────────────────┘                │
│                               │                                             │
│  ═════════════════════════════╪════════════════════════════════════════     │
│                     Memory Bus (128-384 bit)                                │
│  ═════════════════════════════╪════════════════════════════════════════     │
│                               │                                             │
│                    ┌──────────┴──────────┐                                  │
│                    │   Global Memory     │                                  │
│                    │  GDDR6/GDDR6X/HBM3  │                                  │
│                    │  8-80 GB             │                                 │
│                    └─────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Hierarchy:  GPU  →  GPC  →  TPC  →  SM  →  CUDA Cores
```

### What Each Level Does

> **Ref:** Hierarchy terminology defined in [CUDA C++ Programming Guide §7 "Hardware Implementation"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-implementation); component counts from each architecture's whitepaper.

| Level | Full Name | Contains | Role |
|-------|-----------|----------|------|
| **GPU** | Graphics Processing Unit | GPCs, L2 Cache, Memory Controllers | Top-level chip |
| **GPC** | Graphics Processing Cluster | TPCs, Raster Engine, ROPs | Independent processing cluster |
| **TPC** | Texture Processing Cluster | 2 SMs, PolyMorph Engine | Groups SMs with texture units |
| **SM** | Streaming Multiprocessor | CUDA/Tensor/RT cores, caches | Fundamental compute unit |

---

## 2. Inside a GPC (Graphics Processing Cluster)

> **Ref:** [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), Figure 4 "Ada Lovelace GPC": 6 TPCs × 2 SMs = 12 SMs per GPC, 2× ROP partitions, Raster Engine.

```
┌──────────────────────────────────────────────────────────────────┐
│                     GPC (Graphics Processing Cluster)            │
│                                                                  │
│  ┌──────────────────┐                                            │
│  │  Raster Engine   │  ← Triangle setup, rasterization, Z-cull   │
│  └──────────────────┘                                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  ROP Partition 0 (16 ROPs)  │  ROP Partition 1 (16 ROPs) │    │
│  └──────────────────────────────────────────────────────────┘    │
│                            32 ROPs per GPC (Ada)                 │
│                                                                  │
│  ┌───────────────┐ ┌───────────────┐       ┌───────────────┐     │
│  │    TPC 0      │ │    TPC 1      │ . . . │    TPC 5      │     │
│  │ ┌───┐ ┌───┐   │ │ ┌───┐ ┌───┐   │       │ ┌───┐ ┌───┐   │     │
│  │ │SM0│ │SM1│   │ │ │SM2│ │SM3│   │       │ │SM │ │SM │   │     │
│  │ └───┘ └───┘   │ │ └───┘ └───┘   │       │ └───┘ └───┘   │     │
│  │  PolyMorph    │ │  PolyMorph    │       │  PolyMorph    │     │
│  └───────────────┘ └───────────────┘       └───────────────┘     │
│     6 TPCs × 2 SMs = 12 SMs per GPC                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Inside an SM (Streaming Multiprocessor) — Ada Lovelace

> **Ref:** [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), Figure 5 "Ada Lovelace Streaming Multiprocessor (SM)"; [Ada Tuning Guide §1.4.1 "Streaming Multiprocessor"](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#streaming-multiprocessor). SM internals (4 partitions, 128 FP32 cores, 4 Tensor Cores, etc.) from these sources.

This is where all computation happens. The SM is the most important unit to understand.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Streaming Multiprocessor (SM)                        │
│                         Ada Lovelace (CC 8.9)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │Warp Sched. 0 │ │Warp Sched. 1 │ │Warp Sched. 2 │ │Warp Sched. 3 │    │
│  │ Dispatch Unit│ │ Dispatch Unit│ │ Dispatch Unit│ │ Dispatch Unit│    │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘    │
│         │                │                │                │            │
│  ┌──────┴───────┐ ┌──────┴───────┐ ┌──────┴───────┐ ┌──────┴───────┐    │
│  │ Partition 0  │ │ Partition 1  │ │ Partition 2  │ │ Partition 3  │    │
│  │              │ │              │ │              │ │              │    │
│  │ 32× FP32     │ │ 32× FP32     │ │ 32× FP32     │ │ 32× FP32     │    │
│  │ (CUDA Cores) │ │ (CUDA Cores) │ │ (CUDA Cores) │ │ (CUDA Cores) │    │
│  │              │ │              │ │              │ │              │    │
│  │ 16× FP64     │ │ 16× FP64     │ │ 16× FP64     │ │ 16× FP64     │    │
│  │              │ │              │ │              │ │              │    │
│  │ 1× Tensor    │ │ 1× Tensor    │ │ 1× Tensor    │ │ 1× Tensor    │    │
│  │   Core       │ │   Core       │ │   Core       │ │   Core       │    │
│  │              │ │              │ │              │ │              │    │
│  │ 4× LD/ST     │ │ 4× LD/ST     │ │ 4× LD/ST     │ │ 4× LD/ST     │    │
│  │ Units        │ │ Units        │ │ Units        │ │ Units        │    │
│  │              │ │              │ │              │ │              │    │
│  │ 4× SFU       │ │ 4× SFU       │ │ 4× SFU       │ │ 4× SFU       │    │
│  │ (sin,cos,..) │ │ (sin,cos,..) │ │ (sin,cos,..) │ │ (sin,cos,..) │    │
│  │              │ │              │ │              │ │              │    │
│  │ 1× Tex Unit  │ │ 1× Tex Unit  │ │ 1× Tex Unit  │ │ 1× Tex Unit  │    │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   1× RT Core (3rd Gen)                            │  │
│  │           BVH traversal + ray-triangle intersection               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                 Register File: 256 KB                             │  │
│  │           65,536 × 32-bit registers (max 255 per thread)          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │            L1 Data Cache / Shared Memory: 128 KB                  │  │
│  │         ┌──────────────────┬──────────────────────┐               │  │
│  │         │  Shared Memory   │    L1 Data Cache     │               │  │
│  │         │  (programmable)  │  (hardware-managed)  │               │  │
│  │         │  0-100 KB        │   remainder of 128KB │               │  │
│  │         └──────────────────┴──────────────────────┘               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │               Instruction Cache (read-only)                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              Constant Cache (read-only, 64 KB)                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  TOTALS PER SM:                                                         │
│  128 CUDA Cores │ 4 Tensor Cores │ 1 RT Core │ 4 Tex │ 16 LD/ST │16 SFU │
│  Max 2,048 threads │ 64 warps │ 32 thread blocks                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### SM Partition Detail

> **Ref:** [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), p.14–17 SM partition details; [CUDA C++ Programming Guide §7.1 "SIMT Architecture"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) for dispatch unit behavior.

Each SM has 4 processing partitions, each managed by one warp scheduler:

```
          Warp Scheduler
                                                    │
               ▼
    ┌───────────────────────────────────────────────┐
    │   Dispatch Unit                               │ ← Issues 1 instruction per cycle
    └─────────┬─────────────────────────────────────┘
                                                    │
    ┌─────────┴─────────────────────────────────────┐
    │                                               │
    ▼                                      ▼
 ┌────────────────┐              ┌──────────────────┐
 │  32× FP32 Path │              │   16× INT32 Path │
 │  (CUDA Cores)  │              │                  │
 └────────────────┘              └──────────────────┘
    │                                               │
    │  ┌──────────────────────────────┐             │
    ├──│  1× Tensor Core (4th Gen)   │              │
    │  │  FP8/FP16/BF16/TF32/FP64   │               │
    │  │  Sparse matrix support      │              │
    │  └──────────────────────────────┘             │
    │                                               │
    │  ┌──────────────────────────────┐             │
    ├──│  4× Load/Store Units        │──────────────┘
    │  └────────────────────────────────────────────┘
                                                    │
    │  ┌────────────────────────────────────────────┐
    └──│  4× SFU (Special Functions)                │
       │  sin, cos, rcp, sqrt, etc.                 │
       └────────────────────────────────────────────┘
```

---

## 4. Memory Hierarchy

> **Ref:** [CUDA C++ Programming Guide §5.3 "Memory Hierarchy"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy), Figure 4 "Memory Hierarchy"; latency/bandwidth numbers from [Ada Tuning Guide §1.4.2 "Memory System"](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#memory-system) and architecture whitepapers.

```
    FAST ◄──────────────────────────────────────────────── SLOW
    SMALL ─────────────────────────────────────────────── LARGE

    ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │Registers │   │ Shared   │   │ L2 Cache │   │   Global     │
    │          │   │ Mem / L1 │   │          │   │   Memory     │
    │ 256 KB   │   │ 128 KB   │   │ 96 MB    │   │  8-80 GB     │
    │ per SM   │   │ per SM   │   │ unified  │   │ GDDR6/HBM    │
    │          │   │          │   │          │   │              │
    │ ~1 cycle │   │~30 cycle │   │~150 cycle│   │ ~500 cycle   │
    │ ~19 TB/s │   │ ~12 TB/s │   │ ~6 TB/s  │   │ 0.3-3 TB/s   │
    │(per SM)  │   │(per SM)  │   │ (total)  │   │  (total)     │
    │          │   │          │   │          │   │              │
    │ Thread-  │   │ Block-   │   │ All SMs  │   │ All SMs +    │
    │ private  │   │ shared   │   │ share    │   │ CPU share    │
    └──────────┘   └──────────┘   └──────────┘   └──────────────┘
         │               │              │                       │
         ▼               ▼              ▼                ▼
    Compiler-       Programmer-    Hardware-        Hardware-
    managed         managed        managed          managed
```

### Memory Scope and Visibility

> **Ref:** [CUDA C++ Programming Guide §5.3 "Memory Hierarchy"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy), Figure 5 "Memory Hierarchy"; scope rules from §5.3 and §10.2 "Variable Memory Space Specifiers".

```
┌─────────────── GPU ───────────────────────────────────────────────┐
│                                                                   │
│  ┌──── Thread Block 0 ──────────┐  ┌── Thread Block 1 ────────┐   │
│  │                               │  │                           │ │
│  │  ┌─Thread 0─┐ ┌─Thread 1─┐  │  │  ┌─Thread 0─┐            │    │
│  │  │ Registers│ │ Registers│  │  │  │ Registers│  . . .      │   │
│  │  │ Local Mem│ │ Local Mem│  │  │  │ Local Mem│            │    │
│  │  └──────────┘ └──────────┘  │  │  └──────────┘            │    │
│  │         │            │       │  │                           │  │
│  │         └─────┬──────┘       │  │                           │  │
│  │               ▼              │  │                           │  │
│  │     ┌──────────────────┐    │  │   ┌──────────────────┐   │    │
│  │     │  Shared Memory   │    │  │   │  Shared Memory   │   │    │
│  │     │  (per block)     │    │  │   │  (per block)     │   │    │
│  │     └──────────────────┘    │  │   └──────────────────┘   │    │
│  └──────────────┬───────────────┘  └────────────┬─────────────┘   │
│                 │                                │                │
│                 └───────────────┬────────────────┘                │
│                                 ▼                                 │
│                 ┌──────────────────────────┐                      │
│                 │    Global Memory          │                     │
│                 │    (visible to all)       │                     │
│                 │    Constant Memory (r/o)  │                     │
│                 │    Texture Memory  (r/o)  │                     │
│                 └──────────────────────────┘                      │
└───────────────────────────────────────────────────────────────────┘
```

### L1/Shared Memory Configuration (Ada Lovelace)

> **Ref:** [Ada Tuning Guide §1.4.2 "Memory System"](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#memory-system); [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), p.16 "128 KB L1/Shared Memory" configurable split.

The 128 KB per SM is split between shared memory and L1 cache:

```
Config 1:  [ Shared: 0 KB  ][     L1 Cache: 128 KB      ]  ← Compute-heavy
Config 2:  [ Shared: 16 KB ][    L1 Cache: 112 KB       ]
Config 3:  [ Shared: 32 KB ][   L1 Cache: 96 KB         ]
Config 4:  [ Shared: 64 KB ][  L1 Cache: 64 KB          ]  ← Balanced
Config 5:  [ Shared: 100 KB][ L1: 28 KB                 ]  ← Matmul tiling
```

---

## 5. Warp Execution Model (SIMT)

> **Ref:** [CUDA C++ Programming Guide §7.1 "SIMT Architecture"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) and §7.2 "Hardware Multithreading"; warp size = 32 defined in §5.2 "Thread Hierarchy".

### What is a Warp?

A **warp** is a group of 32 threads that execute the **same instruction simultaneously**. This is NVIDIA's SIMT (Single Instruction, Multiple Threads) model.

```
                    Thread Block (256 threads)
                                                   │
              ┌───────────────┼────────────────────┐
              │               │                    │
        ┌─────┴─────┐  ┌─────┴─────┐   ┌─────┴─────┐
        │  Warp 0   │  │  Warp 1   │   │  Warp 7   │
        │ T0  - T31 │  │ T32 - T63 │   │T224 - T255│
        └─────┬─────┘  └─────┬─────┘   └─────┬─────┘
              │               │                    │
              └───────────────┼────────────────────┘
                                                   │
                     Warp Scheduler
                                                   │
                              ▼
              ┌────────────────────────────────────┐
              │   Issue ONE instruction            │
              │   to 32 threads at once            │
              └────────────────────────────────────┘

    Cycle N:   ADD R1, R2, R3     ← All 32 threads execute ADD
    Cycle N+1: MUL R4, R1, R5     ← All 32 threads execute MUL
    Cycle N+2: LD  R6, [R7]       ← All 32 threads issue LOAD
```

### Warp Scheduling — Hiding Latency

> **Ref:** [CUDA C++ Programming Guide §7.2 "Hardware Multithreading"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading); [CUDA Best Practices Guide §10.1 "Occupancy"](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy).

```
    4 Warp Schedulers per SM, up to 64 warps resident

    Cycle │ Scheduler 0  │ Scheduler 1  │ Scheduler 2  │ Scheduler 3
    ──────┼──────────────┼──────────────┼──────────────┼──────────────
      0   │ Warp 0: ADD  │ Warp 1: MUL  │ Warp 2: LD   │ Warp 3: ADD
      1   │ Warp 4: SUB  │ Warp 5: ADD  │ Warp 2: wait │ Warp 7: MUL
      2   │ Warp 0: MUL  │ Warp 8: LD   │ Warp 6: ADD  │ Warp 3: ST
      3   │ Warp 9: ADD  │ Warp 1: ST   │ Warp 2: done │ Warp 10: LD
    ──────┼──────────────┼──────────────┼──────────────┼──────────────
                         ▲
                While Warp 2 waits for memory (500 cycles),
                other warps keep the SM busy!
```

### Warp Divergence

> **Ref:** [CUDA C++ Programming Guide §7.1 "SIMT Architecture"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture), Figure 7 — divergence with active mask; [NVIDIA Volta Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), "Independent Thread Scheduling" section.

```
    if (threadIdx.x < 16) {      // Threads 0-15 take TRUE path
        A();                      // Threads 16-31 are MASKED (idle)
    } else {
        B();                      // Threads 0-15 are MASKED (idle)
    }                             // All threads reconverge

    Execution timeline:
    ┌─────────────────────────────────────────────────────────────┐
    │ Step 1: Threads 0-15 execute A()                            │
    │         [ACTIVE ACTIVE ... ACTIVE  idle idle ... idle]      │
    │         Lanes:  0   1        15    16   17       31         │
    │                                                             │
    │ Step 2: Threads 16-31 execute B()                           │
    │         [idle   idle  ... idle   ACTIVE ACTIVE ... ACTIVE]  │
    │         Lanes:  0   1       15    16   17         31        │
    │                                                             │
    │ Step 3: Reconverge — all threads active again               │
    └─────────────────────────────────────────────────────────────┘

    ⚠  Both paths run SERIALLY → 50% throughput loss!
    ✓  Best practice: minimize divergence within a warp
```

---

## 6. Execution Pipeline

> **Ref:** Pipeline stages derived from [CUDA C++ Programming Guide §7.1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture) and architecture whitepaper SM block diagrams; [Ada Tuning Guide §1.4.1](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#streaming-multiprocessor).

```
┌─────────────────────────────────────────────────────────────────┐
│                   SM Execution Pipeline                         │
│                                                                 │
│   ┌─────────┐    Instruction Cache                              │
│   │ FETCH   │◄── Fetch at Warp's Program Counter                │
│   └────┬────┘                                                   │
│        ▼                                                        │
│   ┌─────────┐    Instruction Buffer                             │
│   │ DECODE  │◄── Decode opcode, identify operands               │
│   └────┬────┘                                                   │
│        ▼                                                        │
│   ┌──────────┐   Warp Scheduler (×4 per SM)                     │
│   │ SCHEDULE │◄── Pick ready warp (no dependency stalls)        │
│   │ / ISSUE  │    Issue up to 4 warps per cycle                 │
│   └────┬─────┘                                                  │
│        ▼                                                        │
│   ┌──────────┐   Register File (256 KB)                         │
│   │ OPERAND  │◄── Read source registers for all 32 threads      │
│   │ COLLECT  │                                                  │
│   └────┬─────┘                                                  │
│        ▼                                                        │
│   ┌──────────┐   Execution Units                                │
│   │ EXECUTE  │◄── FP32, INT32, Tensor Core, SFU, LD/ST          │
│   │          │    SIMT: 32 threads, same instruction            │
│   └────┬─────┘                                                  │
│        ▼                                                        │
│   ┌──────────┐   Memory Subsystem                               │
│   │ MEMORY / │◄── L1 → L2 → Global Memory                       │
│   │WRITEBACK │    Write result to destination register          │
│   └──────────┘                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Memory Coalescing — Why Access Patterns Matter

> **Ref:** [CUDA C++ Programming Guide §8.3.2 "Device Memory Accesses"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) — coalescing rules; [CUDA Best Practices Guide §9.2.1 "Coalesced Access to Global Memory"](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

```
  COALESCED ACCESS (stride = 1)              STRIDED ACCESS (stride = 32)
  ─────────────────────────────              ──────────────────────────────

  Warp of 32 threads:                        Warp of 32 threads:
  T0  T1  T2  T3 ... T31                    T0  T1  T2  T3 ... T31
   │   │   │   │       │                      │   │   │   │          │
   ▼   ▼   ▼   ▼       ▼                      ▼   ▼   ▼   ▼       ▼
  ┌───┬───┬───┬───┬───┬───┐                  ┌───┐           ┌───────┐
  │ 0 │ 1 │ 2 │ 3 │...│31 │                  │ 0 │           │128    │
  └───┴───┴───┴───┴───┴───┘                  └───┘           └───────┘
   ONE 128-byte transaction                   │                      │
                                              │  ┌───┐        │  ┌───┐
   Result: ~220 GB/s ✓                        │  │32 │        │  │160│
                                              │  └───┘        │  └───┘
                                              │  ...                 │  ...
                                              32 SEPARATE transactions!

                                              Result: ~20 GB/s ✗  (10x slower)
```

---

## 8. Concrete GPU Configurations

> **Ref:** Specifications sourced from [NVIDIA Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) Tables 1–3; [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/) for SKU-level details.

### RTX 4060 Laptop (AD107) — Your Target GPU

```
┌──────────────────────────────────────────────────────────┐
│                   RTX 4060 Laptop (AD107)                │
│                   Compute Capability 8.9                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │  GPC 0   │  │  GPC 1   │  │  GPC 2   │   3 GPCs       │
│  │  4 TPCs  │  │  4 TPCs  │  │  4 TPCs  │                │
│  │  8 SMs   │  │  8 SMs   │  │  8 SMs   │                │
│  └──────────┘  └──────────┘  └──────────┘                │
│                                                          │
│  24 SMs × 128 CUDA Cores = 3,072 CUDA Cores              │
│  24 SMs × 4 Tensor Cores =    96 Tensor Cores (Gen 4)    │
│  24 SMs × 1 RT Core      =    24 RT Cores    (Gen 3)     │
│                                                          │
│  Register File:   24 × 256 KB = 6 MB total               │
│  L1/Shared Mem:   24 × 128 KB = 3 MB total               │
│  L2 Cache:        24 MB                                  │
│                                                          │
│  Memory: 8 GB GDDR6, 128-bit bus                         │
│  Memory BW: ~256 GB/s (theoretical peak)                 │
│  FP32 Perf: ~15.11 TFLOPS                                │
│  Tensor Perf: ~242 TFLOPS (FP8)                          │
└──────────────────────────────────────────────────────────┘
```

### RTX 4090 (AD102) — Flagship

> **Ref:** [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), Table 1 "GeForce RTX 4090 Specifications".

```
┌───────────────────────────────────────────────────────────┐
│                    RTX 4090 (AD102)                       │
├───────────────────────────────────────────────────────────┤
│  11 GPCs → 128 SMs                                        │
│  16,384 CUDA Cores │ 512 Tensor Cores │ 128 RT Cores      │
│  96 MB L2 Cache                                           │
│  24 GB GDDR6X, 384-bit bus, ~1 TB/s                       │
│  FP32: ~82.6 TFLOPS                                       │
└───────────────────────────────────────────────────────────┘
```

### H100 (GH100) — Data Center

> **Ref:** [NVIDIA Hopper H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core), Figure 3 "GH100 Full GPU with 144 SMs", Table 1; [NVIDIA H100 Datasheet](https://www.nvidia.com/en-us/data-center/h100/).

```
┌──────────────────────────────────────────────────────────┐
│                    H100 (Hopper GH100)                   │
├──────────────────────────────────────────────────────────┤
│  8 GPCs → 144 SMs (132 enabled in SXM5)                  │
│  16,896 CUDA Cores │ 528 Tensor Cores │ No RT Cores      │
│  50 MB L2 Cache                                          │
│  80 GB HBM3, 5120-bit bus, ~3.35 TB/s                    │
│  FP32: ~67 TFLOPS │ FP8 Tensor: ~1,979 TFLOPS            │
│  New: TMA (Tensor Memory Accelerator), DPX instructions  │
└──────────────────────────────────────────────────────────┘
```

---

## 9. Architecture Comparison

> **Ref:** Comparison data compiled from [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) Table 2, [Ampere GA102 Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) Table 1, and [Hopper Whitepaper](https://resources.nvidia.com/en-us-tensor-core) Table 1.

| Feature | Ampere (GA102) | Ada Lovelace (AD102) | Hopper (GH100) |
|---------|---------------|---------------------|----------------|
| Process | Samsung 8nm | TSMC 4N | TSMC 4N |
| Transistors | 28.3B | 76.3B | 80B |
| Max SMs | 84 | 144 | 144 |
| CUDA Cores/SM | 128 | 128 | 128 |
| Tensor Core Gen | 3rd | 4th | 4th |
| RT Core Gen | 2nd | 3rd | — |
| L2 Cache | 6 MB | 96 MB (16×!) | 50 MB |
| Shared Mem/SM | 128 KB | 128 KB | 228 KB |
| Register File/SM | 256 KB | 256 KB | 256 KB |
| New Features | — | SER, Shader Exec Reorder | TMA, DPX, FP8 |

### Key Ada Lovelace Improvements
- **16× L2 Cache** (6 MB → 96 MB) — reduces global memory pressure dramatically
- **3rd Gen RT Cores** — 2× ray-triangle intersection throughput
- **4th Gen Tensor Cores** — FP8 support, 2× throughput over Ampere
- **Shader Execution Reordering (SER)** — reorders threads for better coherence
- **2× ROPs per GPC** compared to Ampere

---

## 10. Floating-Point Precision — How the GPU Handles FP8/FP16/FP32/FP64

This is one of the most impactful GPU concepts for ML/AI. Different precisions use different hardware paths, different amounts of memory, and achieve wildly different throughput.

### 10.1 Bit Layouts — What's Actually Stored in Memory

> **Ref:** IEEE 754 standard for FP16/FP32/FP64; BF16 from [Google Brain "BFloat16" paper](https://cloud.google.com/tpu/docs/bfloat16); TF32 from [NVIDIA Ampere Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) p.39; FP8 E4M3/E5M2 from ["FP8 Formats For Deep Learning" (arXiv:2209.05433)](https://arxiv.org/abs/2209.05433) by NVIDIA, Arm & Intel; [Hopper Whitepaper](https://resources.nvidia.com/en-us-tensor-core) Figure 6 "FP8 Data Types".

```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  FORMAT     BITS   LAYOUT                    RANGE        PRECISION     │
    │  ─────────  ────   ─────────────────────     ──────────   ──────────    │
    │                                                                         │
    │  FP64       64     [1 sign][11 exp][52 mant]  ±10^308     ~15 digits    │
    │  ████████ ████████ ████████ ████████ ████████ ████████ ████████ ████████│
    │  S EEEEEEEEEEE MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM   │
    │                                                                         │
    │  FP32       32     [1 sign][8 exp][23 mant]   ±3.4×10^38  ~7 digits     │
    │  ████████ ████████ ████████ ████████                                    │
    │  S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM                                     │
    │                                                                         │
    │  TF32       19     [1 sign][8 exp][10 mant]   ±3.4×10^38  ~3 digits     │
    │  ████████ ████████ ███                                                  │
    │  S EEEEEEEE MMMMMMMMMM                                                  │
    │  (same range as FP32, but truncated mantissa — Tensor Core only)        │
    │                                                                         │
    │  BF16       16     [1 sign][8 exp][7 mant]    ±3.4×10^38  ~2 digits     │
    │  ████████ ████████                                                      │
    │  S EEEEEEEE MMMMMMM                                                     │
    │  (same range as FP32! just less precision — great for gradients)        │
    │                                                                         │
    │  FP16       16     [1 sign][5 exp][10 mant]   ±65,504     ~3 digits     │
    │  ████████ ████████                                                      │
    │  S EEEEE MMMMMMMMMM                                                     │
    │  (more precision than BF16, but much smaller range)                     │
    │                                                                         │
    │  FP8 E4M3   8      [1 sign][4 exp][3 mant]    ±448        ~1 digit      │
    │  ████████                                                               │
    │  S EEEE MMM                                                             │
    │  (best for weights & activations in forward pass — no infinities)       │
    │                                                                         │
    │  FP8 E5M2   8      [1 sign][5 exp][2 mant]    ±57,344     ~0.6 digit    │
    │  ████████                                                               │
    │  S EEEEE MM                                                             │
    │  (best for gradients in backward pass — has infinities, wider range)    │
    └─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Memory Impact — Smaller Types = More Bandwidth

```
    Same 32×32 matrix stored in different precisions:

    ┌──────────┬──────────┬────────────────────────────────────────┐
    │ Format   │ Bytes/el │ Matrix size (32×32)  │ vs FP32         │
    ├──────────┼──────────┼──────────────────────┼─────────────────┤
    │ FP64     │ 8 bytes  │ 8,192 bytes  (8 KB)  │ 0.5× BW         │
    │ FP32     │ 4 bytes  │ 4,096 bytes  (4 KB)  │ 1× (baseline)   │
    │ TF32     │ 4 bytes* │ 4,096 bytes  (4 KB)  │ 1× storage**    │
    │ BF16     │ 2 bytes  │ 2,048 bytes  (2 KB)  │ 2× BW           │
    │ FP16     │ 2 bytes  │ 2,048 bytes  (2 KB)  │ 2× BW           │
    │ FP8      │ 1 byte   │ 1,024 bytes  (1 KB)  │ 4× BW           │
    └──────────┴──────────┴──────────────────────┴─────────────────┘

    * TF32 is stored as FP32 in memory, truncated to 19 bits inside Tensor Core
    ** TF32 saves NO memory bandwidth — it's a compute-only optimization

    For a 2048×2048 matrix:
    ┌──────────┬────────────┬──────────────────────────────────────┐
    │ FP32     │  16 MB     │ Takes ~62 μs to read at 256 GB/s     │
    │ FP16     │   8 MB     │ Takes ~31 μs to read at 256 GB/s     │
    │ FP8      │   4 MB     │ Takes ~16 μs to read at 256 GB/s     │
    └──────────┴────────────┴──────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │  KEY INSIGHT: For memory-bound ops (vector add, ReLU),       │
    │  switching FP32→FP16 gives ~2× speedup automatically,        │
    │  because you move half the bytes through the same pipe.      │
    │                                                              │
    │  For compute-bound ops (matmul), the speedup comes from      │
    │  BOTH bandwidth AND Tensor Core throughput increases.        │
    └──────────────────────────────────────────────────────────────┘
```

### 10.3 CUDA Core vs Tensor Core — Two Different Hardware Paths

> **Ref:** CUDA Core throughput from [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), p.14–17; Tensor Core speedup ratios from [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) Table 3 and [Hopper Whitepaper](https://resources.nvidia.com/en-us-tensor-core) Table 2.

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     SM Processing Paths                             │
    │                                                                     │
    │  PATH 1: CUDA Cores (scalar operations)                             │
    │  ═══════════════════════════════════════                            │
    │                                                                     │
    │  ┌─────────────┐     Each core: 1 FMA (a×b+c) per cycle             │
    │  │  FP32 Unit  │     = 2 FLOPs per core per cycle                   │
    │  │  (×128/SM)  │                                                    │
    │  │             │     FP16: Packed FP16×2 mode — process 2 FP16      │
    │  │             │     values in one FP32 unit = 2× throughput        │
    │  │             │                                                    │
    │  │  a ×  b + c │     FP64: 1:64 ratio on consumer GPUs              │
    │  │  ↑    ↑   ↑ │     (only 2 of 128 cores handle FP64)              │
    │  │ scalar scalar│                                                   │
    │  └─────────────┘                                                    │
    │                                                                     │
    │  CUDA Core throughput per SM (Ada Lovelace):                        │
    │  ┌──────────┬──────────────┬────────────────────────────┐           │
    │  │ FP64     │    2 ops/cyc │ 128 cores, but 1:64 ratio  │           │
    │  │ FP32     │  128 ops/cyc │ 128 cores × 1 FMA          │           │
    │  │ FP16×2   │  256 ops/cyc │ 128 cores × 2 packed FP16  │           │
    │  │ INT32    │  128 ops/cyc │ same units, integer mode    │          │
    │  └──────────┴──────────────┴────────────────────────────┘           │
    │                                                                     │
    │                                                                     │
    │  PATH 2: Tensor Cores (matrix operations)                           │
    │  ═════════════════════════════════════════                          │
    │                                                                     │
    │  ┌──────────────────────────────────────────────────────────┐       │
    │  │                   Tensor Core (×4 per SM)                │       │
    │  │                                                          │       │
    │  │   D[m×n]  =  A[m×k]  ×  B[k×n]  +  C[m×n]             │          │
    │  │                                                          │       │
    │  │   Operates on MATRIX FRAGMENTS, not scalars              │       │
    │  │   Entire WARP (32 threads) cooperates on one operation  │        │
    │  │                                                          │       │
    │  │   Throughput depends HEAVILY on precision:               │       │
    │  │                                                          │       │
    │  │   ┌────────┬──────────────────┬────────────────────┐    │        │
    │  │   │ Input  │ Accumulator      │ Speedup vs FP32    │    │        │
    │  │   │ Prec.  │ Precision        │ CUDA cores         │    │        │
    │  │   ├────────┼──────────────────┼────────────────────┤    │        │
    │  │   │ FP64   │ FP64             │  ~2×               │    │        │
    │  │   │ TF32   │ FP32             │  ~8×               │    │        │
    │  │   │ BF16   │ FP32             │  ~16×              │    │        │
    │  │   │ FP16   │ FP32 or FP16     │  ~16×              │    │        │
    │  │   │ FP8    │ FP32 or FP16     │  ~32×              │    │        │
    │  │   │ INT8   │ INT32            │  ~32×              │    │        │
    │  │   └────────┴──────────────────┴────────────────────┘    │        │
    │  │                                                          │       │
    │  │   Why? Smaller input → more elements packed per cycle   │        │
    │  │   FP8 = 1 byte, FP16 = 2 bytes, FP32 = 4 bytes        │          │
    │  │   Tensor Core data path is fixed width — smaller types  │        │
    │  │   means more elements processed simultaneously          │        │
    │  └──────────────────────────────────────────────────────────┘       │
    └─────────────────────────────────────────────────────────────────────┘
```

### 10.4 Tensor Core WMMA Operations — Matrix Fragment Shapes

> **Ref:** [CUDA C++ Programming Guide §10.24 "Warp Matrix Functions"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions), Table "Element Types and Matrix Sizes" (§10.24.6); fragment shape MxNxK definitions from WMMA API.

```
    Tensor Cores don't process individual scalars.
    They process MATRIX FRAGMENTS using Warp-level Matrix Multiply-Accumulate (WMMA).

    All 32 threads in a warp cooperate:

    ┌───────────────────────────────────────────────────────────────────┐
    │  WMMA: D = A × B + C    (warp-wide operation)                     │
    │                                                                   │
    │  Precision    │  A shape  │  B shape  │  C/D shape  │ FLOPs       │
    │  ─────────────┼───────────┼───────────┼─────────────┼──────────   │
    │  FP16         │  16 × 16  │  16 × 16  │  16 × 16   │  8,192       │
    │  BF16         │  16 × 16  │  16 × 16  │  16 × 16   │  8,192       │
    │  TF32         │  16 × 8   │  8 × 16   │  16 × 16   │  4,096       │
    │  FP64         │  8 × 4    │  4 × 8    │  8 × 8     │  512         │
    │  INT8         │  16 × 32  │  32 × 16  │  16 × 16   │  16,384      │
    │  INT4         │  8 × 32   │  32 × 8   │  8 × 8     │  4,096       │
    │  FP8 (Hopper) │  16 × 32  │  32 × 16  │  16 × 16   │  16,384      │
    └───────────────────────────────────────────────────────────────────┘

    How threads share the work (FP16 m16n16k16 example):
    ┌───────────────────────────────────────────────────────────────────┐
    │                                                                   │
    │  A[16×16]          B[16×16]          C[16×16]                     │
    │  ┌──────────┐      ┌──────────┐      ┌──────────┐                 │
    │  │ T0  T0   │      │ T0  T1   │      │ T0  T0   │                 │
    │  │ T0  T0   │  ×   │ T0  T1   │  +   │ T0  T0   │  = D[16×16]     │
    │  │ T1  T1   │      │ T2  T3   │      │ T1  T1   │                 │
    │  │ T1  T1   │      │ T2  T3   │      │ T1  T1   │                 │
    │  │ ...      │      │ ...      │      │ ...      │                 │
    │  │ T31 T31  │      │ T30 T31  │      │ T31 T31  │                 │
    │  └──────────┘      └──────────┘      └──────────┘                 │
    │                                                                   │
    │  Each thread holds a FRAGMENT of the matrices in its registers.   │
    │  The Tensor Core reads all fragments simultaneously and           │
    │  produces the matrix multiply result in one operation.            │
    └───────────────────────────────────────────────────────────────────┘
```

### 10.5 RTX 4060 — Throughput at Each Precision

> **Ref:** TFLOPS numbers calculated from [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) per-SM throughput × 24 SMs × boost clock; sparse throughput = 2× dense per [Ada Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) p.20 "2:4 Structured Sparsity".

```
    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │         RTX 4060 Laptop (AD107) — Performance by Precision                                                                                        │
    │         24 SMs, 3072 CUDA Cores, 96 Tensor Cores (4th Gen)                                                                                        │
    │                                                                                                                                                   │
    │  CUDA CORE PERFORMANCE:                                                                                                                           │
    │  ┌──────────┬──────────────┬────────────────────────────────────┐                                                                                 │
    │  │ FP64     │  ~0.24 TFLOPS│ 1:64 ratio, only for compatibility │                                                                                 │
    │  │ FP32     │  15.11 TFLOPS│ 128 cores/SM × 24 SMs × 2 FLOPs  │                                                                                   │
    │  │ FP16(×2) │  15.11 TFLOPS│ packed half2 on CUDA cores        │                                                                                  │
    │  └──────────┴──────────────┴────────────────────────────────────┘                                                                                 │
    │                                                                                                                                                   │
    │  TENSOR CORE PERFORMANCE (Dense):                                                                                                                 │
    │  ┌──────────┬──────────────┬────────────────────────────────────┐                                                                                 │
    │  │ FP8      │  60.5 TFLOPS │  4× FP32 CUDA   — best inference  │                                                                                  │
    │  │ FP16     │  30.2 TFLOPS │  2× FP32 CUDA   — training sweet │                                                                                   │
    │  │ BF16     │  30.2 TFLOPS │  2× FP32 CUDA     spot for DL     │                                                                                  │
    │  │ TF32     │  15.1 TFLOPS │  ~1× FP32 CUDA  — "free" upgrade │                                                                                   │
    │  │ INT8     │ 120.9 TOPS   │  8× FP32 CUDA   — quantized inf. │                                                                                   │
    │  └──────────┴──────────────┴────────────────────────────────────┘                                                                                 │
    │                                                                                                                                                   │
    │  TENSOR CORE PERFORMANCE (2:4 Sparse):                                                                                                            │
    │  ┌──────────┬──────────────┬────────────────────────────────────┐                                                                                 │
    │  │ FP8      │ 120.9 TFLOPS │  2× dense = 8× FP32 CUDA         │                                                                                   │
    │  │ FP16     │  60.5 TFLOPS │  2× dense = 4× FP32 CUDA         │                                                                                   │
    │  │ BF16     │  60.5 TFLOPS │  2× dense = 4× FP32 CUDA         │                                                                                   │
    │  │ INT8     │ 241.8 TOPS   │  2× dense = 16× FP32 CUDA        │                                                                                   │
    │  └──────────┴──────────────┴────────────────────────────────────┘                                                                                 │
    │                                                                                                                                                   │
    │  Comparison visualization:                                                                                                                        │
    │                                                                                                                                                   │
    │  FP64 CUDA:  █ 0.24 TFLOPS                                                                                                                        │
    │  FP32 CUDA:  ████████████████ 15.1 TFLOPS                                                                                                         │
    │  TF32 TC:    ████████████████ 15.1 TFLOPS                                                                                                         │
    │  FP16 TC:    ████████████████████████████████ 30.2 TFLOPS                                                                                         │
    │  BF16 TC:    ████████████████████████████████ 30.2 TFLOPS                                                                                         │
    │  FP8  TC:    ████████████████████████████████████████████████████████████████ 60.5 TFLOPS                                                         │
    │  INT8 TC:    ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 120.9 TOPS  │
    │  FP8 Sparse: ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 120.9 TFLOPS│
    └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 10.6 Packed FP16 on CUDA Cores — How FP32 Units Do FP16

> **Ref:** [NVIDIA Ampere GA102 Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) p.15 "FP16/BF16 packed operations"; `__half2` intrinsics in [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF2__ARITHMETIC.html).

```
    CUDA cores DON'T have separate FP16 hardware.
    Instead, FP32 units process TWO FP16 values simultaneously using "half2" packing.

    FP32 unit doing FP32:                FP32 unit doing packed FP16:
    ┌────────────────────────┐           ┌─────────────────────────┐
    │   32-bit data path     │           │   32-bit data path      │
    │                        │           │                         │
    │  ┌──────────────────┐  │           │  ┌─────────┬─────────┐  │
    │  │  1 × FP32 value  │  │           │  │ FP16 hi │ FP16 lo │  │
    │  │  a × b + c       │  │           │  │ a1×b1+c1│ a2×b2+c2│  │
    │  │  = 2 FLOPs       │  │           │  │ 2 FLOPs │ 2 FLOPs │  │
    │  └──────────────────┘  │           │  └─────────┴─────────┘  │
    │                        │           │  = 4 FLOPs total        │
    └────────────────────────┘           └─────────────────────────┘

    Code to use this:
    ┌──────────────────────────────────────────────────────────────┐
    │  // Slow: individual FP16 operations (no packing)            │
    │  __half a, b, c;                                             │
    │  c = __hadd(a, b);         // uses 1 FP32 unit, wastes half  │
    │                                                              │
    │  // Fast: packed FP16×2 operations                           │
    │  __half2 a2, b2, c2;                                         │
    │  c2 = __hadd2(a2, b2);    // uses 1 FP32 unit, full width    │
    │  // Processes 2 additions in the same cycle!                 │
    └──────────────────────────────────────────────────────────────┘

    ⚠ Important: This packed mode only works for CUDA core element-wise ops.
    For matrix multiply, Tensor Cores are MUCH faster (16-32×).
```

### 10.7 TF32 — The "Free Lunch" for FP32 Code

> **Ref:** [NVIDIA Ampere A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) p.39–41 "TensorFloat-32 Precision" (first introduced in Ampere); [PyTorch TF32 documentation](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices).

```
    TF32 (TensorFloat-32) is a special 19-bit format used ONLY inside Tensor Cores.
    Your data stays FP32 in memory — the Tensor Core truncates internally.

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  FP32 in memory:  [1 sign][8 exponent][23 mantissa] = 32 bits   │
    │                    S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM           │
    │                                                                 │
    │  When Tensor Core reads it, it TRUNCATES to TF32:               │
    │                    S EEEEEEEE MMMMMMMMMM                        │
    │                    [1 sign][8 exponent][10 mantissa] = 19 bits  │
    │                                         ▲                       │
    │                              13 mantissa bits DROPPED           │
    │                              (~3 decimal digits of precision)   │
    │                                                                 │
    │  WHY?                                                           │
    │  • Same RANGE as FP32 (8-bit exponent preserved)                │
    │  • Fits more multiply operations in the Tensor Core data path   │
    │  • ~8× speedup over FP32 CUDA cores for matmul                  │
    │  • Accumulation still done in full FP32                         │
    │  • Enabled by DEFAULT in PyTorch for matmul since 1.7           │
    │                                                                 │
    │  In PyTorch:                                                    │
    │  torch.matmul(A, B)  # Automatically uses TF32 on Ampere+       │
    │                                                                 │
    │  To disable:                                                    │
    │  torch.backends.cuda.matmul.allow_tf32 = False                  │
    └─────────────────────────────────────────────────────────────────┘
```

### 10.8 Mixed Precision — FP16 Compute, FP32 Accumulate

> **Ref:** ["Mixed Precision Training" (Micikevicius et al., ICLR 2018, arXiv:1710.03740)](https://arxiv.org/abs/1710.03740); [NVIDIA AMP documentation](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html); Tensor Core accumulator behavior from architecture whitepapers.

```
    The most common training pattern: multiply in FP16, accumulate in FP32.
    This is what AMP (Automatic Mixed Precision) does.

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  Inside a Tensor Core (FP16 input, FP32 accumulator):               │
    │                                                                     │
    │  Step 1: Load A and B fragments as FP16                             │
    │                                                                     │
    │  A (FP16)         B (FP16)         C (FP32 accumulator)             │
    │  ┌──┬──┬──┬──┐   ┌──┬──┬──┬──┐   ┌────┬────┬────┬────┐              │
    │  │2B│2B│2B│2B│ × │2B│2B│2B│2B│ + │ 4B │ 4B │ 4B │ 4B │              │
    │  └──┴──┴──┴──┘   └──┴──┴──┴──┘   └────┴────┴────┴────┘              │
    │   16 bits each     16 bits each     32 bits each                    │
    │                                                                     │
    │  Step 2: Multiply (in reduced precision)                            │
    │  products = A_elements × B_elements  (FP16 × FP16 → FP16)           │
    │                                                                     │
    │  Step 3: Accumulate (in full precision)                             │
    │  C_fp32 += convert_to_fp32(products)  (widen to FP32, then add)     │
    │                                                                     │
    │  Step 4: Output D (can be FP16 or FP32)                             │
    │  D = C  (keep FP32 for next accumulation or convert to FP16)        │
    │                                                                     │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │  WHY THIS WORKS:                                             │   │
    │  │                                                              │   │
    │  │  • FP16 multiply: individual products only need ~3 digits   │    │
    │  │  • FP32 accumulate: sum of many products needs more range   │    │
    │  │  • Without FP32 accum: errors compound → training diverges  │    │
    │  │  • With FP32 accum: matches FP32-only training accuracy     │    │
    │  │                                                              │   │
    │  │  Example:  sum of 1024 values around 0.001                  │    │
    │  │  FP16 accum:  loses precision → result ~0.98 (wrong)       │     │
    │  │  FP32 accum:  keeps precision → result ~1.024 (correct)    │     │
    │  └──────────────────────────────────────────────────────────────┘   │
    │                                                                     │
    │  In PyTorch:                                                        │
    │  with torch.autocast(device_type='cuda', dtype=torch.float16):      │
    │      output = model(input)   # Forward in FP16                      │
    │      loss = criterion(output, target)                               │
    │  scaler.scale(loss).backward()  # Backward with loss scaling        │
    │  scaler.step(optimizer)          # Update weights in FP32           │
    └─────────────────────────────────────────────────────────────────────┘
```

### 10.9 FP8 — The Two Flavors (E4M3 vs E5M2)

> **Ref:** ["FP8 Formats For Deep Learning" (arXiv:2209.05433)](https://arxiv.org/abs/2209.05433) by NVIDIA, Arm & Intel; [Hopper Whitepaper](https://resources.nvidia.com/en-us-tensor-core) Figure 6; [NVIDIA Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/) for E4M3/E5M2 automatic selection.

```
    FP8 has TWO formats for different purposes in training:

    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  E4M3 (4-bit exponent, 3-bit mantissa)                               │
    │  ████████                                                            │
    │  S EEEE MMM                                                          │
    │                                                                      │
    │  • Range: ±448                                                       │
    │  • More precision (3 mantissa bits)                                  │
    │  • NO infinities (all exponent patterns used for numbers)            │
    │  • Best for: FORWARD PASS (weights and activations)                  │
    │    → values are bounded, need precision                              │
    │                                                                      │
    │  ─────────────────────────────────────────────────────               │
    │                                                                      │
    │  E5M2 (5-bit exponent, 2-bit mantissa)                               │
    │  ████████                                                            │
    │  S EEEEE MM                                                          │
    │                                                                      │
    │  • Range: ±57,344  (128× wider than E4M3!)                           │
    │  • Less precision (2 mantissa bits)                                  │
    │  • HAS infinities and NaN (like IEEE formats)                        │
    │  • Best for: BACKWARD PASS (gradients)                               │
    │    → gradients can spike, need range more than precision             │
    │                                                                      │
    │  ─────────────────────────────────────────────────────               │
    │                                                                      │
    │  TRAINING DATA FLOW:                                                 │
    │                                                                      │
    │     Forward Pass           Backward Pass                             │
    │  ┌──────────────┐      ┌──────────────────┐                          │
    │  │ Weights: E4M3│ ───► │ Gradients: E5M2  │                          │
    │  │ Activns: E4M3│      │ (wider range for  │                         │
    │  │ (more precise│      │  gradient spikes) │                         │
    │  │  for values) │      └────────┬─────────┘                          │
    │  └──────────────┘               │                                    │
    │         ▲                        │                                   │
    │         └────────────────────────┘                                   │
    │              Weight update (in FP32)                                 │
    │                                                                      │
    │  Hopper Transformer Engine automates this E4M3/E5M2 selection.       │
    └──────────────────────────────────────────────────────────────────────┘
```

### 10.10 What Happens to Our 32×32 Matmul at Different Precisions

```
    Revisiting the concrete example from Section 13:
    C[32×32] = A[32×32] × B[32×32]

    ┌───────────┬────────┬──────────┬────────────┬──────────────────────┐
    │ Precision │ A+B+C  │ FLOPs    │ AI         │ Bound by             │
    │           │ size   │          │ (FLOP/byte)│                      │
    ├───────────┼────────┼──────────┼────────────┼──────────────────────┤
    │ FP64      │ 24 KB  │ 65,536   │ 2.7        │ Memory + overhead    │
    │ FP32      │ 12 KB  │ 65,536   │ 5.3        │ Memory + overhead    │
    │ FP16      │  6 KB  │ 65,536   │ 10.7       │ Memory + overhead    │
    │ FP8       │  3 KB  │ 65,536   │ 21.3       │ Memory + overhead    │
    └───────────┴────────┴──────────┴────────────┴──────────────────────┘
    All below ridge point (60) → all memory-bound at this size.
    All dominated by kernel launch overhead at this size.

    At 2048×2048 (where it actually matters):
    ┌───────────┬────────┬──────────┬────────────┬──────────────────────┐
    │ Precision │ A+B+C  │ FLOPs    │ AI         │ Hardware path        │
    ├───────────┼────────┼──────────┼────────────┼──────────────────────┤
    │ FP32 CUDA │ 48 MB  │ 17.2B    │ 341        │ 128 CUDA cores/SM    │
    │           │        │          │            │ → ~15 TFLOPS peak    │
    │           │        │          │            │                      │
    │ TF32 TC   │ 48 MB  │ 17.2B    │ 341        │ 4 Tensor Cores/SM    │
    │           │        │          │            │ → ~15 TFLOPS peak    │
    │           │        │          │            │ (same storage!)      │
    │           │        │          │            │                      │
    │ FP16 TC   │ 24 MB  │ 17.2B    │ 682        │ 4 Tensor Cores/SM    │
    │           │        │          │            │ → ~30 TFLOPS peak    │
    │           │        │          │            │ (2× compute, 2× BW)  │
    │           │        │          │            │                      │
    │ FP8 TC    │ 12 MB  │ 17.2B    │ 1,365      │ 4 Tensor Cores/SM    │
    │           │        │          │            │ → ~60 TFLOPS peak    │
    │           │        │          │            │ (4× compute, 4× BW)  │
    └───────────┴────────┴──────────┴────────────┴──────────────────────┘

    The 32×32 tiled kernel at FP16 — what changes:
    ┌───────────────────────────────────────────────────────────────────┐
    │  Shared memory:                                                   │
    │    As[32][32] of half (FP16) = 32×32×2 = 2,048 bytes = 2 KB       │
    │    Bs[32][32] of half (FP16) = 32×32×2 = 2,048 bytes = 2 KB       │
    │    Total: 4 KB  (vs 8 KB for FP32)  → 2× more tiles can fit       │
    │                                                                   │
    │  Global memory transactions:                                      │
    │    Warp loads 32 × 2 bytes = 64 bytes per transaction             │
    │    (vs 128 bytes for FP32)                                        │
    │    → Same number of transactions, but each is half the size       │
    │    → 2× effective bandwidth improvement                           │
    │                                                                   │
    │  Register usage:                                                  │
    │    Accumulator `sum` should still be FP32 (mixed precision)       │
    │    Input operands in FP16 → fewer registers for temporaries       │
    │                                                                   │
    │  If using Tensor Cores (WMMA) instead of CUDA core loop:          │
    │    Replace:  for(k) sum += As[ty][k] * Bs[k][tx];                 │
    │    With:     wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);      │
    │    The entire 32×32 matmul becomes ~2 WMMA operations             │
    │    (using m16n16k16 fragments: 2×2 = 4 fragments to cover 32×32)  │
    └───────────────────────────────────────────────────────────────────┘
```

### 10.11 2:4 Structured Sparsity — Double the Throughput

> **Ref:** [NVIDIA Ampere A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) p.42–44 "Fine-Grained Structured Sparsity" (first introduced in Ampere); [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) p.20.

```
    Ada Lovelace & Hopper Tensor Cores support 2:4 structured sparsity:
    Out of every 4 values, at least 2 must be zero.

    Dense matrix:                Sparse matrix (2:4 pattern):
    ┌───┬───┬───┬───┐           ┌───┬───┬───┬───┐
    │ 3 │ 0 │ 7 │ 2 │           │ 3 │ 0 │ 7 │ 0 │  ← 2 of 4 are zero
    │ 0 │ 5 │ 1 │ 4 │           │ 0 │ 5 │ 0 │ 4 │  ← 2 of 4 are zero
    │ 8 │ 6 │ 0 │ 3 │           │ 8 │ 0 │ 0 │ 3 │  ← 2 of 4 are zero
    └───┴───┴───┴───┘           └───┴───┴───┴───┘

    Storage: only store non-zero values + 2-bit index metadata
    ┌───┬───┐ ┌─────────────────────────────────┐
    │ 3 │ 7 │ │ 0, 2                            │  ← values + column indices
    │ 5 │ 4 │ │ 1, 3                            │
    │ 8 │ 3 │ │ 0, 3                            │
    └───┴───┘ └─────────────────────────────────┘
    50% less data → Tensor Core processes 2× as many elements/cycle

    RTX 4060 with sparsity:
    FP16:  30.2 TFLOPS (dense) → 60.5 TFLOPS (sparse) = 2× ✓
    FP8:   60.5 TFLOPS (dense) → 120.9 TFLOPS (sparse) = 2× ✓
```

### 10.12 When to Use Each Precision — Practical Guidelines

```
    ┌──────────┬──────────────────────────────────────────────────────────┐
    │ Format   │ Use when                                                 │
    ├──────────┼──────────────────────────────────────────────────────────┤
    │ FP64     │ Scientific computing, financial calculations.            │
    │          │ NOT for ML — 64× slower on consumer GPUs.                │
    │          │                                                          │
    │ FP32     │ Default safe choice. Weight master copy in training.     │
    │          │ ~15 TFLOPS on RTX 4060.                                  │
    │          │                                                          │
    │ TF32     │ "Free upgrade" — PyTorch uses automatically for matmul.  │
    │          │ Same code as FP32, Tensor Cores truncate internally.     │
    │          │ ~15 TFLOPS but uses Tensor Cores (more efficient).       │
    │          │                                                          │
    │ BF16     │ Training with wide range (same as FP32).                 │
    │          │ Better than FP16 for gradients (less overflow risk).     │
    │          │ ~30 TFLOPS on RTX 4060.                                  │
    │          │                                                          │
    │ FP16     │ Inference, training with loss scaling (AMP).             │
    │          │ More precision than BF16 but narrower range.             │
    │          │ ~30 TFLOPS on RTX 4060.                                  │
    │          │                                                          │
    │ FP8      │ Inference on latest models (Hopper/Ada).                 │
    │          │ Training with Transformer Engine (Hopper).               │
    │          │ E4M3 for forward, E5M2 for backward.                     │
    │          │ ~60 TFLOPS on RTX 4060.                                  │
    │          │                                                          │
    │ INT8     │ Quantized inference only. Not for training.              │
    │          │ Post-training quantization (PTQ) or QAT.                 │
    │          │ ~120 TOPS on RTX 4060.                                   │
    │          │                                                          │
    │ INT4     │ Aggressive quantized inference (GPTQ, AWQ).              │
    │          │ Significant quality loss without careful calibration.    │
    └──────────┴──────────────────────────────────────────────────────────┘
```

---

## 11. Data Flow — From CPU to GPU Computation

> **Ref:** [CUDA C++ Programming Guide §5.4 "Heterogeneous Programming"](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming), Figure 3 "Heterogeneous Programming"; API calls from §6.2 "CUDA Runtime".

```
    ┌─────────┐                         ┌─────────────────────────┐
    │   CPU   │                         │         GPU             │
    │  (Host) │                         │       (Device)          │
    └────┬────┘                         └──────────┬──────────────┘
         │                                                        │
    1. Allocate GPU memory ──────────────────────► cudaMalloc()
         │                                                        │
    2. Copy data to GPU ─────── PCIe/NVLink ──────► Global Memory
         │                                                        │
    3. Launch kernel ──────────────────────────────► GigaThread Engine
         │                                          │             │
         │                                                        │    ▼ Distributes
         │                                          │  ┌──────────┐
         │                                          │  │ GPC → SM │
         │                                          │  └──────┬───┘
         │                                          │             │
         │                                                        │    ▼ SM executes:
         │                                                        │  Load Global Mem → L2 → L1
         │                                                        │  Load L1 → Registers
         │                                                        │  CUDA Cores compute
         │                                                        │  Store Registers → L1 → L2
         │                                                        │  Write back → Global Mem
         │                                          │             │
    4. Synchronize ◄────────────────────────────── kernel complete
         │                                                        │
    5. Copy results back ◄── PCIe/NVLink ──────── Global Memory
         │                                                        │
    6. Free GPU memory ──────────────────────────► cudaFree()
```

---

## 12. Key Formulas and Metrics

> **Ref:** Roofline Model from ["Roofline: An Insightful Visual Performance Model" (Williams et al., 2009)](https://doi.org/10.1145/1498765.1498785); Arithmetic Intensity definition from [CUDA Best Practices Guide §8](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html); ridge point calculated from RTX 4060 spec sheet.

```
    Memory Bandwidth (GB/s)   = Bytes Transferred / Time(s) / 1e9
    FLOPS                      = Floating Point Operations
    TFLOPS                     = FLOPS / Time(s) / 1e12
    Arithmetic Intensity (AI)  = FLOPS / Bytes Accessed

    Roofline Model:
    ┌───────────────────────────────────────────────┐
    │                                               │
    │  Performance   ╱ Peak FLOPS ───────────────   │
    │  (TFLOPS)    ╱                                │
    │            ╱                                  │
    │          ╱ ← Ridge Point                      │
    │        ╱    (AI = Peak FLOPS / Peak BW)       │
    │      ╱      RTX 4060: ~60 FLOPs/byte          │
    │    ╱                                          │
    │  ╱  Memory-     │    Compute-                 │
    │╱    Bound       │    Bound                    │
    └──────────────────┴────────────────────────────┘
          Arithmetic Intensity (FLOPs/byte) →

    RTX 4060 Ridge Point = 15 TFLOPS / 256 GB/s ≈ 60 FLOPs/byte
    Below 60: memory-bound (optimize memory access)
    Above 60: compute-bound (optimize FLOPs, use Tensor Cores)
```

---

## 13. Concrete Example: 32×32 Matrix Multiplication Data Flow

Let's trace **exactly** what happens when you compute `C = A × B` where A and B are 32×32 float32 matrices on an RTX 4060 Laptop. We'll follow both the **naive** and **tiled** kernels from `kernels/matmul.cu`.

### The Numbers First

```
    Matrices: A[32×32], B[32×32], C[32×32]   (M=N=K=32)

    Memory:
      Each matrix = 32 × 32 × 4 bytes = 4,096 bytes = 4 KB
      Total memory = 3 × 4 KB = 12 KB  (A read, B read, C write)

    Compute:
      FLOPs = 2 × M × N × K = 2 × 32 × 32 × 32 = 65,536 FLOPs
      (each output element: 32 multiplies + 31 adds ≈ 2×32 = 64 FLOPs)

    Arithmetic Intensity:
      AI = 65,536 FLOPs / 12,288 bytes = 5.3 FLOPs/byte

      RTX 4060 ridge point = 60 FLOPs/byte
      5.3 << 60  →  THIS IS MEMORY-BOUND!  (even matmul, at small size)

    ⚠  At 32×32, the problem is TOO SMALL to saturate the GPU.
       The overhead of launching the kernel (~5-10 μs) likely exceeds
       the actual computation time (~0.01 μs at peak throughput).
       This is EXACTLY what Experiment 5 (overhead analysis) demonstrates.
```

### Step 0: CPU Side — Kernel Launch

```
    Python/PyTorch code:
    ┌──────────────────────────────────────────────────────────────┐
    │  A = torch.randn(32, 32, device='cuda')   # 4 KB on GPU      │
    │  B = torch.randn(32, 32, device='cuda')   # 4 KB on GPU      │
    │  C = matmul_tiled(A, B)                    # kernel launch   │
    └──────────────────────────────────────────────────────────────┘
                                                                   │
                            ▼
    CUDA Runtime:
    ┌──────────────────────────────────────────────────────────────┐
    │  C = torch::empty({32, 32}, A.options());  // allocate 4 KB  │
    │                                                              │
    │  dim3 threads(32, 32);   // 32×32 = 1,024 threads per block  │
    │  dim3 blocks(1, 1);      // only 1 block needed!             │
    │                          // (32/32=1 in each dimension)      │
    │                                                              │
    │  matmul_tiled_kernel<<<blocks(1,1), threads(32,32)>>>(       │
    │      A_ptr, B_ptr, C_ptr, 32, 32, 32                         │
    │  );                                                          │
    └──────────────────────────────────────────────────────────────┘
                                                                   │
                                                                   │  Command pushed to GPU command queue
                                                                   │  via PCIe / memory-mapped registers
                            ▼
```

### Step 1: GigaThread Engine — Work Distribution

```
    ┌──────────────────────────────────────────────────────────────┐
    │                    GigaThread Engine                         │
    │                                                              │
    │  Receives kernel launch:                                     │
    │    Grid:  1×1 blocks  (just 1 thread block total)            │
    │    Block: 32×32 threads (1,024 threads)                      │
    │                                                              │
    │  Thread block → assigned to ONE SM                           │
    │                                                              │
    │  ┌────────────────────────────────────────────────────┐      │
    │  │  Block (0,0) ──────────────────► SM #7 (arbitrary) │      │
    │  └────────────────────────────────────────────────────┘      │
    │                                                              │
    │  Only 1 of 24 SMs is used! (1/24 = 4.2% GPU utilization)     │
    │  This is why small matrices are inefficient on GPUs.         │
    │                                                              │
    │  The other 23 SMs sit completely idle.                       │
    └──────────────────────────────────────────────────────────────┘
```

### Step 2: SM Receives the Thread Block

```
    SM #7 receives Block (0,0) with 1,024 threads
                                                                   │
                          ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  SM #7 — Warp Assignment                                     │
    │                                                              │
    │  1,024 threads ÷ 32 threads/warp = 32 warps                  │
    │                                                              │
    │  Threads are assigned to warps by LINEARIZING threadIdx:     │
    │  linear_id = threadIdx.y * blockDim.x + threadIdx.x          │
    │            = threadIdx.y * 32 + threadIdx.x                  │
    │                                                              │
    │  Warp 0:  threads (y=0, x=0..31)  → linear IDs 0-31          │
    │  Warp 1:  threads (y=1, x=0..31)  → linear IDs 32-63         │
    │  Warp 2:  threads (y=2, x=0..31)  → linear IDs 64-95         │
    │  ...                                                         │
    │  Warp 31: threads (y=31, x=0..31) → linear IDs 992-1023      │
    │                                                              │
    │  ┌──────────────────────────────────────────────────────┐    │
    │  │  KEY INSIGHT: Each warp = one ROW of the 32×32 grid  │    │
    │  │  Warp 0 = Row 0 of thread block (ty=0, tx=0..31)    │     │
    │  │  Warp 5 = Row 5 of thread block (ty=5, tx=0..31)    │     │
    │  └──────────────────────────────────────────────────────┘    │
    │                                                              │
    │  4 Warp Schedulers handle 32 warps:                          │
    │    Scheduler 0: Warps 0, 4, 8, 12, 16, 20, 24, 28            │
    │    Scheduler 1: Warps 1, 5, 9, 13, 17, 21, 25, 29            │
    │    Scheduler 2: Warps 2, 6, 10, 14, 18, 22, 26, 30           │
    │    Scheduler 3: Warps 3, 7, 11, 15, 19, 23, 27, 31           │
    │                                                              │
    │  Each scheduler manages 8 warps (8 × 4 = 32 total)           │
    └──────────────────────────────────────────────────────────────┘
```

### Step 3: Register Allocation

```
    ┌──────────────────────────────────────────────────────────────┐
    │  Register Allocation (per thread, compiler-determined)       │
    │                                                              │
    │  For the tiled kernel, each thread needs approximately:      │
    │                                                              │
    │  float sum = 0.0f;        →  1 register  (accumulator)       │
    │  int row, col;            →  2 registers  (coordinates)      │
    │  int bx, by, tx, ty;     →  4 registers  (block/thread ID)   │
    │  int t;                   →  1 register  (tile loop index)   │
    │  temp values for loads    →  ~4 registers (addresses, etc.)  │
    │  ─────────────────────────────────────────                   │
    │  Total: ~12-16 registers per thread                          │
    │                                                              │
    │  Total register usage:                                       │
    │  1,024 threads × 16 registers × 4 bytes = 64 KB              │
    │  SM has 256 KB register file → 25% used                      │
    │                                                              │
    │  ✓ Plenty of room — could run more thread blocks if needed   │
    └──────────────────────────────────────────────────────────────┘
```

### Step 4: Shared Memory Allocation

```
    ┌──────────────────────────────────────────────────────────────┐
    │  Shared Memory Layout (for tiled kernel)                     │
    │                                                              │
    │  __shared__ float As[32][32];  →  32×32×4 = 4,096 bytes      │
    │  __shared__ float Bs[32][32];  →  32×32×4 = 4,096 bytes      │
    │  ─────────────────────────────────────────────               │
    │  Total shared memory: 8,192 bytes = 8 KB                     │
    │                                                              │
    │  SM has 128 KB configurable → only 6.25% used                │
    │                                                              │
    │  Memory layout in shared memory (32 banks, 4 bytes each):    │
    │                                                              │
    │  As[32][32] in shared memory:                                │
    │  Bank:   0    1    2    3   ...  31                          │
    │  ┌─────┬─────┬─────┬─────┬─────┬─────┐                       │
    │  │As   │As   │As   │As   │     │As   │  Row 0                │
    │  │[0][0]│[0][1]│[0][2]│[0][3]│ ... │[0][31]│                 │
    │  ├─────┼─────┼─────┼─────┼─────┼─────┤                       │
    │  │As   │As   │As   │As   │     │As   │  Row 1                │
    │  │[1][0]│[1][1]│[1][2]│[1][3]│ ... │[1][31]│                 │
    │  ├─────┼─────┼─────┼─────┼─────┼─────┤                       │
    │  │ ... │ ... │ ... │ ... │ ... │ ... │                       │
    │  └─────┴─────┴─────┴─────┴─────┴─────┘                       │
    │                                                              │
    │  ✓ As[ty][tx]: tx varies across warp → each thread hits      │
    │    a different bank → NO bank conflicts for row access!      │
    │                                                              │
    │  ⚠ As[ty][k] where k is same for all threads in warp:        │
    │    All 32 threads read same column → BROADCAST (1 conflict)  │
    │    Actually OK: shared memory supports broadcast when all    │
    │    threads in a warp read the SAME address in a bank.        │
    └──────────────────────────────────────────────────────────────┘
```

### Step 5: Execution — Naive Kernel (for comparison)

```
    NAIVE KERNEL: matmul_naive_kernel
    Each thread computes one element of C[row][col]

    Thread (ty=3, tx=5) computes C[3][5]:
    ┌───────────────────────────────────────────────────────────────┐
    │                                                               │
    │  sum = 0                                                      │
    │  for k = 0 to 31:                                             │
    │      sum += A[3][k] * B[k][5]                                 │
    │                                                               │
    │  C[3][5] = sum                                                │
    │                                                               │
    │         A                    B                   C            │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
    │  │             │    │     ▼col=5  │    │             │        │
    │  │             │    │     │       │    │             │        │
    │  │             │    │     │       │    │             │        │
    │  │►►►►►►►►►►►►►│ ×  │     │       │ =  │     ● C[3,5]│        │
    │  │  row=3      │    │     │       │    │             │        │
    │  │             │    │     │       │    │             │        │
    │  └─────────────┘    └─────────────┘    └─────────────┘        │
    │                                                               │
    │  Memory accesses per thread:                                  │
    │    Read A: 32 floats (row 3, all columns) = 128 bytes         │
    │    Read B: 32 floats (all rows, column 5) = 128 bytes         │
    │    Write C: 1 float = 4 bytes                                 │
    │                                                               │
    │  Total per thread: 260 bytes for 64 FLOPs                     │
    │                                                               │
    │  PROBLEM — Look at Warp 3 (threads with ty=3, tx=0..31):      │
    │                                                               │
    │  Thread tx=0: reads A[3][0], A[3][1], ..., A[3][31]  (same!)  │
    │  Thread tx=1: reads A[3][0], A[3][1], ..., A[3][31]  (same!)  │
    │  Thread tx=2: reads A[3][0], A[3][1], ..., A[3][31]  (same!)  │
    │  ...                                                          │
    │  All 32 threads read THE SAME ROW of A → L1 cache helps       │
    │                                                               │
    │  Thread tx=0: reads B[0][0], B[1][0], ..., B[31][0]           │
    │  Thread tx=1: reads B[0][1], B[1][1], ..., B[31][1]           │
    │  → At each k, threads access B[k][0..31] → COALESCED ✓        │
    │                                                               │
    │  Total global memory reads across ALL 1,024 threads:          │
    │    A: 1024 × 32 reads = 32,768 reads (but 32 threads          │
    │       share each row → 32 unique rows × 32 = 1,024 unique)    │
    │    B: 1024 × 32 reads = 32,768 reads (32 × 32 unique)         │
    │    C: 1024 writes = 1,024 writes                              │
    │                                                               │
    │  Without caching: 32,768 + 32,768 + 1,024 = 66,560 accesses   │
    │  With L1 cache: A rows cached → ~2,048 actual transactions    │
    └───────────────────────────────────────────────────────────────┘
```

### Step 6: Execution — Tiled Kernel (our actual implementation)

For 32×32 with TILE_SIZE=32, there is only **1 tile iteration** (the whole matrix fits in one tile). Let's trace it step by step.

```
    TILED KERNEL: matmul_tiled_kernel
    TILE_SIZE = 32,  K = 32  →  numTiles = 32/32 = 1 iteration

    ════════════════════════════════════════════════════════════════
    PHASE 1: Load Tile of A into Shared Memory
    ════════════════════════════════════════════════════════════════

    Each thread loads ONE element:
    Thread (ty, tx) loads: As[ty][tx] = A[row * 32 + t*32 + tx]
                                       = A[ty * 32 + tx]  (t=0)

    All 1,024 threads load in parallel → entire A loaded at once!

    Warp 0 (ty=0, tx=0..31):
    ┌────────────────────────────────────────────────────────────┐
    │  Global Memory Read:                                       │
    │  T0  reads A[0×32 + 0]  = A[0]    → As[0][0]               │
    │  T1  reads A[0×32 + 1]  = A[1]    → As[0][1]               │
    │  T2  reads A[0×32 + 2]  = A[2]    → As[0][2]               │
    │  ...                                                       │
    │  T31 reads A[0×32 + 31] = A[31]   → As[0][31]              │
    │                                                            │
    │  Addresses: A+0, A+4, A+8, ..., A+124  (consecutive!)      │
    │  → ONE 128-byte coalesced transaction  ✓                   │
    └────────────────────────────────────────────────────────────┘

    Warp 1 (ty=1, tx=0..31):  reads A[32..63]  → 1 transaction
    Warp 2 (ty=2, tx=0..31):  reads A[64..95]  → 1 transaction
    ...
    Warp 31 (ty=31, tx=0..31): reads A[992..1023] → 1 transaction

    Total for loading A: 32 coalesced transactions = 32 × 128 = 4 KB  ✓
    (matches: 32×32×4 bytes = 4 KB)

    ════════════════════════════════════════════════════════════════
    PHASE 2: Load Tile of B into Shared Memory (same pattern)
    ════════════════════════════════════════════════════════════════

    Thread (ty, tx) loads: Bs[ty][tx] = B[(t*32 + ty) * 32 + col]
                                       = B[ty * 32 + tx]  (t=0)

    Same coalesced pattern → 32 transactions → 4 KB loaded  ✓

    ════════════════════════════════════════════════════════════════
    PHASE 3: __syncthreads()  — Barrier
    ════════════════════════════════════════════════════════════════

    All 32 warps must reach this point before ANY can proceed.
    This ensures As[][] and Bs[][] are fully populated.

    ┌────────────────────────────────────────────────────────────┐
    │  Warp 0:  ████████████░░░░░░  (arrived, waiting)           │
    │  Warp 1:  ██████████████░░░░  (arrived, waiting)           │
    │  Warp 2:  ████████░░░░░░░░░░  (still loading B)            │
    │  ...                                                       │
    │  Warp 31: ██████████████████  (arrived, waiting)           │
    │                                                            │
    │  When ALL arrive → barrier releases → Phase 4 begins       │
    └────────────────────────────────────────────────────────────┘

    ════════════════════════════════════════════════════════════════
    PHASE 4: Compute — The Dot Products (from shared memory!)
    ════════════════════════════════════════════════════════════════

    for (k = 0; k < 32; k++) {
        sum += As[ty][k] * Bs[k][tx];
    }

    Let's trace Thread (ty=3, tx=5) computing C[3][5]:

    k=0:  sum += As[3][0] * Bs[0][5]    ← shared mem reads (~30 cycles)
    k=1:  sum += As[3][1] * Bs[1][5]    ← shared mem reads
    k=2:  sum += As[3][2] * Bs[2][5]    ← shared mem reads
    ...
    k=31: sum += As[3][31] * Bs[31][5]  ← shared mem reads

    ┌────────────────────────────────────────────────────────────┐
    │  Memory access pattern analysis for Warp 3 at step k=7:    │
    │                                                            │
    │  All 32 threads read As[3][7]:                             │
    │    → Same address! Shared memory BROADCAST (no conflict)   │
    │                                                            │
    │  Thread tx=0  reads Bs[7][0]   → Bank 0                    │
    │  Thread tx=1  reads Bs[7][1]   → Bank 1                    │
    │  Thread tx=2  reads Bs[7][2]   → Bank 2                    │
    │  ...                                                       │
    │  Thread tx=31 reads Bs[7][31]  → Bank 31                   │
    │    → All different banks! NO bank conflicts  ✓             │
    │                                                            │
    │  Per step: 1 broadcast + 32 conflict-free reads            │
    │  Per step: 1 FMA (multiply-add) per thread                 │
    │  32 steps × 32 threads = 1,024 FMAs per warp               │
    └────────────────────────────────────────────────────────────┘

    ════════════════════════════════════════════════════════════════
    PHASE 5: Write Results to Global Memory
    ════════════════════════════════════════════════════════════════

    C[row * 32 + col] = sum;

    Warp 0 (ty=0, tx=0..31):
      T0  writes C[0]   = sum
      T1  writes C[1]   = sum
      ...
      T31 writes C[31]  = sum
      → ONE 128-byte coalesced write transaction  ✓

    Total: 32 warps × 1 transaction = 32 transactions = 4 KB written
```

### Step 7: Pipeline View — What the Warp Schedulers See

```
    ┌─────────────────────────────────────────────────────────────────┐
    │  Cycle-level view of Scheduler 0 (managing Warps 0,4,8,...,28)  │
    │                                                                 │
    │  Cycle │ Warp │ Instruction          │ Notes                    │
    │  ──────┼──────┼──────────────────────┼───────────────────────── │
    │  0     │ W0   │ LD As[0][0..31]      │ Load A row 0 from glob   │
    │  1     │ W4   │ LD As[4][0..31]      │ Load A row 4 from glob   │
    │  2     │ W8   │ LD As[8][0..31]      │ Load A row 8 from glob   │
    │  3     │ W12  │ LD As[12][0..31]     │ Load A row 12            │
    │  ...   │      │                      │ (W0 still waiting for    │
    │        │      │                      │  memory ~500 cycles)     │
    │  4     │ W16  │ LD As[16][0..31]     │                          │
    │  5     │ W20  │ LD As[20][0..31]     │                          │
    │  6     │ W24  │ LD As[24][0..31]     │                          │
    │  7     │ W28  │ LD As[28][0..31]     │                          │
    │  ...   │      │ (all warps issued    │ Scheduler waits for      │
    │        │      │  their A loads)      │ memory returns           │
    │  ~500  │ W0   │ LD Bs[0][0..31]      │ W0's A load returned!    │
    │  ~501  │ W4   │ LD Bs[4][0..31]      │ Issue B loads            │
    │  ...   │      │                      │                          │
    │  ~1000 │      │ __syncthreads()      │ Barrier — all wait       │
    │  ~1001 │ W0   │ FMA sum+=As*Bs (k=0) │ Compute begins!          │
    │  ~1002 │ W4   │ FMA sum+=As*Bs (k=0) │ (shared mem: ~30 cyc)    │
    │  ~1003 │ W8   │ FMA sum+=As*Bs (k=0) │                          │
    │  ~1004 │ W12  │ FMA sum+=As*Bs (k=0) │                          │
    │  ...   │      │                      │                          │
    │  ~1030 │ W0   │ FMA sum+=As*Bs (k=1) │ W0's k=0 done, next k    │
    │  ...   │      │   (32 k-iterations)  │                          │
    │  ~2000 │ W0   │ ST C[0][0..31]       │ Write results            │
    │  ...   │      │                      │                          │
    │  ~2500 │      │ KERNEL COMPLETE      │                          │
    └─────────────────────────────────────────────────────────────────┘

    Note: cycle numbers are approximate and illustrative.
    Real execution overlaps across all 4 schedulers simultaneously.
```

### Step 8: Full Data Flow Through Memory Hierarchy

```
    ┌──────────────────────────────────────────────────────────────────┐
    │          Complete Memory Journey for 32×32 Matmul                │
    │                                                                  │
    │  GLOBAL MEMORY (GDDR6, 8 GB)                                     │
    │  ┌─────────┬─────────┬─────────┐                                 │
    │  │  A[32×32]│  B[32×32]│  C[32×32]│   12 KB total                │
    │  │  4 KB    │  4 KB    │  4 KB    │                              │
    │  └────┬─────┴────┬─────┴────┬─────┘                              │
    │       │ read     │ read     │ write                              │
    │       │ 32 txns  │ 32 txns  │ 32 txns                            │
    │       ▼          ▼          ▲                                    │
    │  L2 CACHE (24 MB) ── 12 KB easily fits, likely stays cached      │
    │       │          │          │                                    │
    │       ▼          ▼          ▲                                    │
    │  L1 CACHE (128 KB per SM) ── 12 KB fits entirely                 │
    │       │          │          │                                    │
    │       ▼          ▼          ▲                                    │
    │  SHARED MEMORY                                                   │
    │  ┌──────────┬──────────┐                                         │
    │  │As[32][32]│Bs[32][32]│   8 KB (programmer-managed)             │
    │  │  4 KB    │  4 KB    │                                         │
    │  └────┬─────┴────┬─────┘                                         │
    │       │          │         (32 reads per thread per k-step)      │
    │       ▼          ▼                                               │
    │  REGISTERS (per thread)                                          │
    │  ┌──────┐                                                        │
    │  │ sum  │  1 register — accumulates dot product                  │
    │  │ temp │  temporaries for FMA                                   │
    │  └──┬───┘                                                        │
    │     │                                                            │
    │     ▼                                                            │
    │  CUDA CORE — FMA unit                                            │
    │  sum = sum + As[ty][k] * Bs[k][tx]                               │
    │  (1 fused multiply-add = 2 FLOPs)                                │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
```

### Naive vs Tiled — Memory Traffic Comparison

```
    ┌────────────────────────┬──────────────────┬──────────────────┐
    │                        │  NAIVE KERNEL    │  TILED KERNEL    │
    ├────────────────────────┼──────────────────┼──────────────────┤
    │ Global memory reads    │                  │                  │
    │   A                    │ 32K reads*       │ 1,024 reads      │
    │   B                    │ 32K reads*       │ 1,024 reads      │
    │   Total reads          │ 64K reads*       │ 2,048 reads      │
    │                        │                  │                  │
    │ * without L1 caching   │ * L1 helps a lot │ shared mem       │
    │   (theoretical worst)  │   in practice    │ guarantees reuse │
    │                        │                  │                  │
    │ Shared memory reads    │ 0                │ 64K reads        │
    │ (per thread: 2×32×32)  │                  │ (but ~30 cycles!)│
    │                        │                  │                  │
    │ Global memory writes   │ 1,024            │ 1,024            │
    │ (same for both)        │                  │                  │
    │                        │                  │                  │
    │ Sync barriers          │ 0                │ 2 (load + end)   │
    │                        │                  │                  │
    │ Compute per thread     │ 32 FMA = 64 FLOP │ 32 FMA = 64 FLOP │
    ├────────────────────────┼──────────────────┼──────────────────┤
    │ For 32×32:             │ Similar speed    │ Similar speed    │
    │                        │ (L1 cache saves  │ (shared mem is   │
    │                        │  the naive case) │  explicit cache) │
    │                        │                  │                  │
    │ For 2048×2048:         │ ~0.1 TFLOPS      │ ~2 TFLOPS        │
    │ (where tiling wins)    │ (L1 can't hold   │ (tiles fit in    │
    │                        │  entire rows)    │  shared mem)     │
    └────────────────────────┴──────────────────┴──────────────────┘

    WHY TILING DOESN'T HELP MUCH AT 32×32:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    The entire 4 KB matrix A fits in L1 cache (128 KB).
    The naive kernel benefits from implicit caching.
    Tiling shines when matrices EXCEED the L1 cache size:

    32×32:   4 KB per matrix   → fits in L1 (128 KB)   → tiling ≈ naive
    256×256: 256 KB per matrix → EXCEEDS L1 (128 KB)   → tiling >> naive
    2048×2048: 16 MB per matrix → far exceeds L1        → tiling >>> naive
```

### Why 32×32 Is a Terrible GPU Workload

```
    ┌──────────────────────────────────────────────────────────────┐
    │  RTX 4060 Resources vs What 32×32 Matmul Uses                │
    │                                                              │
    │  SMs:          24 available, 1 used          (4.2%)          │
    │  CUDA Cores:   3,072 available, 128 used     (4.2%)          │
    │  Warps:        1,536 max, 32 active          (2.1%)          │
    │  Registers:    6 MB total, ~64 KB used       (1.0%)          │
    │  Shared Mem:   3 MB total, 8 KB used         (0.3%)          │
    │  L2 Cache:     24 MB total, 12 KB used       (0.05%)         │
    │  Memory BW:    256 GB/s peak, ~0.01 GB/s used (0.004%)       │
    │                                                              │
    │  Kernel launch overhead: ~5-10 μs                            │
    │  Actual compute time:    ~0.01 μs (at peak throughput)       │
    │  Overhead is 500-1000× the useful work!                      │
    │                                                              │
    │  ┌──────────────────────────────────────────────────────┐    │
    │  │  LESSON: GPUs need MASSIVE parallelism to saturate.  │    │
    │  │                                                      │    │
    │  │  For matmul, you need ~512×512+ to start seeing     │     │
    │  │  meaningful GPU utilization, and 2048×2048+ to       │    │
    │  │  approach peak throughput.                            │   │
    │  │                                                      │    │
    │  │  For 32×32? The CPU is probably faster due to        │    │
    │  │  zero launch overhead + data already in CPU cache.   │    │
    │  └──────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────┘
```

### Scaling Up: What Changes at 2048×2048

```
    ┌────────────────────────────┬───────────────┬────────────────┐
    │ Property                   │  32×32        │  2048×2048     │
    ├────────────────────────────┼───────────────┼────────────────┤
    │ Elements per matrix        │ 1,024         │ 4,194,304      │
    │ Memory per matrix          │ 4 KB          │ 16 MB          │
    │ FLOPs                      │ 65,536        │ 17.2 billion   │
    │ Arithmetic Intensity       │ 5.3 FLOP/B    │ 341 FLOP/B     │
    │ Bound by                   │ Memory (& OH) │ Compute ✓      │
    │                            │               │                │
    │ Thread blocks              │ 1             │ 4,096          │
    │ (at TILE_SIZE=32)          │ (1×1)         │ (64×64)        │
    │                            │               │                │
    │ SMs used                   │ 1 of 24       │ 24 of 24       │
    │                            │               │ (blocks rotate)│
    │                            │               │                │
    │ Tile iterations per block  │ 1             │ 64             │
    │ (K / TILE_SIZE)            │               │                │
    │                            │               │                │
    │ Global → Shared loads      │ 2 × 4 KB     │ 2×64 × 4 KB     │
    │ per block                  │ = 8 KB        │ = 512 KB       │
    │                            │               │                │
    │ Shared mem reuse factor    │ 32× (= K)    │ 32× (= TILE)    │
    │                            │               │                │
    │ Expected perf (tiled)      │ ~meaningless  │ ~2 TFLOPS      │
    │ Expected perf (cuBLAS)     │ ~meaningless  │ ~12 TFLOPS     │
    └────────────────────────────┴───────────────┴────────────────┘
```

---

## 14. Professional & Data Center GPU Variants

> **Ref:** Die comparisons from [Ampere GA102 Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) and [Ampere A100 (GA100) Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf); professional specs from [RTX A6000 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web.pdf).

NVIDIA ships **different dies** and **different configurations** across consumer, professional, and data center product lines. This is one of the most misunderstood aspects of their lineup.

### The Two Ampere Dies: GA100 vs GA102

A critical insight: the A6000 and A100 are **completely different architectures** despite both being "Ampere."

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AMPERE: Two Different Dies                           │
│                                                                         │
│  ┌──────────────────────────┐    ┌──────────────────────────────────┐   │
│  │     GA100 (Data Center)  │    │     GA102 (Consumer / Pro Vis)   │   │
│  │                          │    │                                  │   │
│  │  TSMC 7nm, 826 mm²      │    │  Samsung 8nm, 628 mm²           │     │
│  │  54.2B transistors       │    │  28.3B transistors              │    │
│  │                          │    │                                  │   │
│  │  108 SMs                 │    │  84 SMs                         │    │
│  │  64 FP32 cores/SM        │    │  128 FP32 cores/SM              │    │
│  │  = 6,912 CUDA cores      │    │  = 10,752 CUDA cores            │    │
│  │                          │    │                                  │   │
│  │  FULL FP64 (1:2 ratio)  │    │  THROTTLED FP64 (1:64 ratio)   │      │
│  │  19.5 TFLOPS FP64       │    │  ~0.3 TFLOPS FP64              │      │
│  │                          │    │                                  │   │
│  │  4× Tensor perf vs Volta│    │  2× Tensor perf vs Volta       │      │
│  │                          │    │                                  │   │
│  │  HBM2e memory            │    │  GDDR6 / GDDR6X memory         │     │
│  │  Up to 2 TB/s BW         │    │  Up to 768 GB/s BW             │     │
│  │                          │    │                                  │   │
│  │  NO RT Cores             │    │  84 RT Cores (Gen 2)           │     │
│  │  NO display outputs      │    │  Full graphics pipeline         │    │
│  │  NO NVENC/NVDEC          │    │  NVENC + NVDEC                  │    │
│  │                          │    │                                  │   │
│  │  MIG support (7 slices)  │    │  No MIG                         │    │
│  │                          │    │                                  │   │
│  │  Products:               │    │  Products:                      │    │
│  │    A100 (40/80 GB)       │    │    RTX 3090/3090 Ti (consumer) │     │
│  │    A30 (24 GB, 4 MIG)   │    │    A6000 (professional)        │      │
│  │                          │    │    A40 (data center vis)        │    │
│  └──────────────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### NVIDIA A6000 (Ampere Professional)

> **Ref:** [NVIDIA RTX A6000 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web.pdf); NVLink and ECC details from [NVIDIA Professional Visualization page](https://www.nvidia.com/en-us/design-visualization/).

The A6000 uses the **full GA102 die** — same silicon as RTX 3090, but fully enabled with professional features.

```
┌───────────────────────────────────────────────────────────┐
│                 NVIDIA A6000 (GA102)                      │
│              Ampere Professional Visualization            │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  84 SMs (full GA102 — RTX 3090 has only 82)               │
│  10,752 CUDA Cores │ 336 Tensor Cores │ 84 RT Cores       │
│                                                           │
│  Register File:   84 × 256 KB = 21 MB total               │
│  L1/Shared Mem:   84 × 128 KB = 10.5 MB total             │
│  L2 Cache:        6 MB                                    │
│                                                           │
│  Memory: 48 GB GDDR6 with ECC                             │
│  Memory Bus: 384-bit                                      │
│  Memory BW: 768 GB/s                                      │
│  FP32 Perf: ~38.7 TFLOPS                                  │
│  Power: 300W                                              │
│                                                           │
│  ┌──────────────────────────────────────────────────┐     │
│  │  PROFESSIONAL FEATURES (not on RTX 3090):        │     │
│  │  ✓ NVLink (112.5 GB/s, 2-way → 96 GB combined)  │      │
│  │  ✓ ECC Memory (error-correcting)                 │     │
│  │  ✓ 48 GB VRAM (2× RTX 3090's 24 GB)            │       │
│  │  ✓ ISV-certified professional drivers            │     │
│  │  ✓ Passive cooling option, 24/7 rated           │      │
│  └──────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────┘
```

### A6000 vs RTX 3090 — Same Die, Different Config

> **Ref:** SM/CUDA core counts from [Ampere GA102 Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) Table 1; pricing and feature comparison from NVIDIA product pages and [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/).

```
                    GA102 Die (84 SMs max)
    ┌────────────────────────────────────────────────────┐
    │ SM SM SM SM SM SM SM SM SM SM SM SM │ SM SM │      │
    │ SM SM SM SM SM SM SM SM SM SM SM SM │ SM SM │      │
    │ SM SM SM SM SM SM SM SM SM SM SM SM │ SM SM │      │
    │ SM SM SM SM SM SM SM SM SM SM SM SM │ SM SM │      │
    │ SM SM SM SM SM SM SM SM SM SM SM SM │ SM SM │      │
    │ SM SM SM SM SM SM SM SM SM SM SM SM │ SM SM │      │
    │ SM SM SM SM SM SM SM SM SM SM SM SM │       │      │
    └────────────────────────────────────────────────────┘
    │◄──────── 80 SMs ─────────────────►│◄─ +4 ─►        │

    RTX 3090:    82 of 84 SMs enabled (2 disabled for yield)
    RTX 3090 Ti: 84 of 84 SMs enabled (full die)
    A6000:       84 of 84 SMs enabled (full die)
    A40:         84 of 84 SMs enabled (full die, data center form)

    ┌───────────────┬──────────┬──────────────┬──────────┐
    │ Feature       │ RTX 3090 │ RTX 3090 Ti  │  A6000   │
    ├───────────────┼──────────┼──────────────┼──────────┤
    │ SMs           │ 82       │ 84           │ 84       │
    │ CUDA Cores    │ 10,496   │ 10,752       │ 10,752   │
    │ VRAM          │ 24 GB    │ 24 GB        │ 48 GB    │
    │ Memory Type   │ GDDR6X   │ GDDR6X       │ GDDR6    │
    │ ECC           │ No       │ No           │ Yes      │
    │ NVLink        │ No       │ No           │ Yes      │
    │ MIG           │ No       │ No           │ No       │
    │ Power         │ 350W     │ 450W         │ 300W     │
    │ Price         │ $1,499   │ $1,999       │ $4,650+  │
    └───────────────┴──────────┴──────────────┴──────────┘
```

### Ada Lovelace Variants — Same Pattern, Different Configs

> **Ref:** [Ada Lovelace Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) Table 1–2; [NVIDIA L40S Datasheet](https://www.nvidia.com/en-us/data-center/l40s/); [NVIDIA RTX 6000 Ada Datasheet](https://www.nvidia.com/en-us/design-visualization/rtx-6000/).

```
                    AD102 Die (144 SMs max)

    ┌───────────────┬──────────┬──────────────┬──────────┐
    │ Feature       │ RTX 4090 │ RTX 6000 Ada │  L40S    │
    ├───────────────┼──────────┼──────────────┼──────────┤
    │ SMs           │ 128      │ 142          │ 142      │
    │ CUDA Cores    │ 16,384   │ 18,176       │ 18,176   │
    │ Tensor Cores  │ 512      │ 568          │ 568      │
    │ RT Cores      │ 128      │ 142          │ 142      │
    │ L2 Cache      │ 96 MB    │ 96 MB        │ 96 MB    │
    │ VRAM          │ 24 GB    │ 48 GB        │ 48 GB    │
    │ Memory Type   │ GDDR6X   │ GDDR6 ECC    │ GDDR6 ECC│
    │ Memory BW     │ ~1 TB/s  │ 864 GB/s     │ 864 GB/s │
    │ NVLink        │ No       │ Yes (8-way)  │ No       │
    │ Power         │ 450W     │ 300W         │ 300W     │
    │ Cooling       │ Active   │ Passive/Act. │ Passive  │
    │ Target        │ Gaming   │ Workstation  │ DC Infer.│
    └───────────────┴──────────┴──────────────┴──────────┘

    ⚠  RTX 4090 often OUTPERFORMS RTX 6000 Ada by 3-25%
       despite fewer SMs — because it has 50% more power
       budget (450W vs 300W) allowing much higher clocks!
```

### Data Center Lineage — Compute-Only Dies

> **Ref:** [Ampere A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf); [Hopper Whitepaper](https://resources.nvidia.com/en-us-tensor-core); [Blackwell Architecture Overview](https://resources.nvidia.com/en-us-blackwell-architecture); [NVIDIA B200 Datasheet](https://www.nvidia.com/en-us/data-center/b200/).

```
    ┌─────────────────────────────────────────────────────────────────┐
    │              DATA CENTER COMPUTE EVOLUTION                      │
    │                                                                 │
    │   Ampere             Hopper              Blackwell              │
    │   ┌───────┐          ┌───────┐          ┌───────────┐           │
    │   │ GA100 │          │ GH100 │          │   GB200   │           │
    │   │       │ ───────► │       │ ───────► │           │           │
    │   │ 7nm   │          │ 4nm   │          │   4nm     │           │
    │   │ 826mm²│          │ 814mm²│          │  2×die    │           │
    │   │ 54.2B │          │  80B  │          │   208B    │           │
    │   └───┬───┘          └───┬───┘          └─────┬─────┘           │
    │       │                  │                     │                │
    │       ▼                  ▼                     ▼                │
    │   A100                H100/H200           B100/B200             │
    │   80GB HBM2e          80GB HBM3           192GB HBM3e           │
    │   2 TB/s              3.35 TB/s           8 TB/s                │
    │   312 TFLOPS(FP16)    990 TFLOPS(FP16)    ~4.5 PFLOPS(FP8)      │
    │                                                                 │
    │   ┌─────────────────────────────────────────────┐               │
    │   │  FEATURES ONLY ON DATA CENTER DIES:         │               │
    │   │  ✓ Full-rate FP64 (1:2 vs 1:64 on consumer)│                │
    │   │  ✓ MIG (Multi-Instance GPU) — 7 partitions  │               │
    │   │  ✓ HBM memory (3-10× bandwidth vs GDDR6)   │                │
    │   │  ✓ Enhanced Tensor Cores (4× vs 2× Volta)  │                │
    │   │  ✗ No RT Cores (compute only)               │               │
    │   │  ✗ No display outputs                       │               │
    │   │  ✗ No video encode/decode                   │               │
    │   └─────────────────────────────────────────────┘               │
    └─────────────────────────────────────────────────────────────────┘
```

### Hopper & Blackwell — Latest Data Center Architectures

> **Ref:** H100 specs from [Hopper Whitepaper](https://resources.nvidia.com/en-us-tensor-core), Table 1 & Figure 3; H200/GH200 from [NVIDIA H200 Datasheet](https://www.nvidia.com/en-us/data-center/h200/); B200/GB200 from [Blackwell Architecture Overview](https://resources.nvidia.com/en-us-blackwell-architecture).

```
┌─────────────────────────────────────────────────────────────────────┐
│                          H100 (Hopper)                              │
├─────────────────────────────────────────────────────────────────────┤
│  132 SMs (of 144) │ 16,896 CUDA Cores │ 528 Tensor Cores            │
│  80 GB HBM3 @ 3.35 TB/s │ 50 MB L2 Cache                            │
│  Up to 228 KB shared memory per SM (1.8× Ada)                       │
│  FP8 Tensor: ~1,979 TFLOPS │ FP64: ~34 TFLOPS                       │
│  MIG: 7 instances │ NVLink: 900 GB/s (NVSwitch)                     │
│  New: TMA (Tensor Memory Accelerator) — async bulk data movement    │
│  New: DPX instructions — dynamic programming acceleration           │
│  New: Thread Block Clusters — cooperative groups across SMs         │
│  Power: 350W (PCIe) / 700W (SXM)                                    │
├─────────────────────────────────────────────────────────────────────┤
│  H200 = same GH100 die + 141 GB HBM3e @ 4.8 TB/s (1.8× BW)          │
│  GH200 = Grace CPU (72 ARM cores) + H100/H200 via NVLink-C2C        │
│          900 GB/s coherent CPU↔GPU link (7× faster than PCIe 5)     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         B200 (Blackwell)                            │
├─────────────────────────────────────────────────────────────────────┤
│  208B transistors │ TSMC 4NP                                        │
│  192 GB HBM3e @ 8 TB/s                                              │
│  FP4: 9 PFLOPS │ FP8: 4.5 PFLOPS │ FP16: ~2.25 PFLOPS               │
│  2:4 Sparsity doubles throughput → FP4 Sparse: 18 PFLOPS            │
│  New: FP4 & FP6 precision for inference                             │
│  New: 2nd-gen Transformer Engine                                    │
│  Power: 1.0-1.2 kW                                                  │
│                                                                     │
│  vs H100: 3× training, 15× inference (LLM workloads)                │
│  GB200 = Grace CPU + B200 GPU (like GH200 pattern)                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Multi-Instance GPU (MIG) — Data Center Only

> **Ref:** [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/); [Ampere A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) p.47–50 "Multi-Instance GPU"; [Hopper Whitepaper](https://resources.nvidia.com/en-us-tensor-core) MIG section.

```
    ┌──────────────────────────────────────────────────────────┐
    │                    A100 / H100 with MIG                  │
    │                                                          │
    │  Full GPU (e.g., H100 SXM: 132 SMs, 80 GB HBM3)          │
    │                                                          │
    │  Partitioned into up to 7 isolated instances:            │
    │                                                          │
    │  ┌────────┐┌────────┐┌────────┐┌────────┐                │
    │  │ MIG 0  ││ MIG 1  ││ MIG 2  ││ MIG 3  │  ...           │
    │  │ 1/7 SM ││ 1/7 SM ││ 1/7 SM ││ 1/7 SM │                │
    │  │ 1/7 Mem││ 1/7 Mem││ 1/7 Mem││ 1/7 Mem│                │
    │  │ Own L2 ││ Own L2 ││ Own L2 ││ Own L2 │                │
    │  └────────┘└────────┘└────────┘└────────┘                │
    │                                                          │
    │  Each instance has:                                      │
    │  ✓ Dedicated SMs (not shared)                            │
    │  ✓ Dedicated memory bandwidth                            │
    │  ✓ Dedicated L2 cache partition                          │
    │  ✓ Hardware-level fault isolation                        │
    │  ✓ Independent CUDA contexts                             │
    │                                                          │
    │  NOT available on: A6000, A40, RTX 4090, RTX 6000 Ada    │
    │  (requires GA100 or GH100 die)                           │
    └──────────────────────────────────────────────────────────┘
```

### Complete Product Map — Die to Product

> **Ref:** Compiled from all architecture whitepapers (Ada, Ampere GA100/GA102, Hopper, Blackwell), official NVIDIA product pages, and [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/).

```
    ┌──────────────────────────────────────────────────────────────────┐
    │                    NVIDIA GPU Product Map                        │
    │                                                                  │
    │  DIE          CONSUMER         PROFESSIONAL      DATA CENTER     │
    │  ─────────    ────────────     ──────────────    ──────────────  │
    │                                                                  │
    │  GA102        RTX 3090 (82SM)  A6000 (84SM)     A40 (84SM)       │
    │  (Ampere)     RTX 3090Ti(84SM)                                   │
    │  Samsung 8nm  RTX 3080 (68SM)                                    │
    │               RTX 3070 Ti(48SM)                                  │
    │                                                                  │
    │  GA100        —                —                 A100 (108SM)    │
    │  (Ampere DC)                                     A30  (reduced)  │
    │  TSMC 7nm                                                        │
    │                                                                  │
    │  AD102        RTX 4090 (128SM) RTX 6000 Ada     L40  (142SM)     │
    │  (Ada)                         (142SM)           L40S (142SM)    │
    │  TSMC 4N      RTX 4080 (76SM)                                    │
    │                                                                  │
    │  AD107        RTX 4060 (24SM)  —                 —               │
    │  (Ada)                                                           │
    │                                                                  │
    │  GH100        —                —                 H100 (132SM)    │
    │  (Hopper)                                        H200 (132SM)    │
    │  TSMC 4N                                         GH200 (w/Grace) │
    │                                                                  │
    │  GB200        —                —                 B200            │
    │  (Blackwell)                                     GB200 (w/Grace) │
    │  TSMC 4NP                                                        │
    └──────────────────────────────────────────────────────────────────┘
```

---

## Sources

### Architecture Whitepapers
- [NVIDIA Ada Lovelace Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [NVIDIA Ampere GA102 Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
- [NVIDIA Ampere GA100 (A100) Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA Hopper H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [NVIDIA Blackwell Architecture Overview](https://resources.nvidia.com/en-us-blackwell-architecture)

### Programming & Tuning Guides
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)

### Product Datasheets
- [NVIDIA RTX A6000 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web.pdf)
- [NVIDIA L40S Datasheet](https://www.nvidia.com/en-us/data-center/l40s/)
- [NVIDIA A30 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-datasheet.pdf)

---

*This document is part of the [Prometheus GPU Architecture Learning Lab](/home/sixigma/Prometheus). See `EXPERIMENTS.md` for hands-on experiments that demonstrate these concepts.*
