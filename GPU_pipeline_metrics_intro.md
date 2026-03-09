# GPU Performance Microarchitecture Analysis: The Relationship Between
# Occupancy, Issue Rate, and Pipeline Utilization

## Abstract
During GPU performance analysis (e.g., via NVIDIA Nsight Compute), it is common
to observe seemingly contradictory hardware metrics: extremely low occupancy
(16.67%) and issue rate (Issue Slots Busy 10.65%), yet accompanied by a
moderately high pipeline utilization (Pipeline Utilization 48.9%). This document
uses CPU pipeline mapping and GPU hardware resource allocation mechanisms to
explain the microarchitecture logic and causal relationships behind these three
metrics.

---

## 1. Foundation: Mapping GPU SM to CPU Pipeline Concepts
To understand how these metrics interact, we must first clarify the correspondence
between a GPU Streaming Multiprocessor (SM) and a traditional CPU pipeline:

- **Pipeline Frontend (Fetch / Decode / Issue): handled by the Warp Scheduler.**
  Each cycle, the Warp Scheduler inspects all Resident Warps and selects one
  whose next instruction is ready to issue.

- **Instruction Stream Context: the Warp itself.**
  A Warp is not a compute unit. It is a hardware context consisting of one
  Program Counter (PC) and its dedicated register state. It maps directly to
  a CPU Hardware Thread (e.g., an SMT thread in Intel Hyper-Threading).

- **Pipeline Backend (Execute / Memory / WriteBack): handled by Execution Units.**
  These include CUDA Cores, Tensor Cores, and Load/Store Units (LSU) —
  equivalent to a CPU's ALU and memory access pipelines.

**Key architectural difference**: A GPU hides backend execution latency through
zero-cost Warp context switching (Latency Hiding). When a Warp stalls during
Execute or WriteBack, the Scheduler immediately switches to another Warp in the
next cycle, keeping the backend pipeline fed.

---

## 2. Occupancy Bottleneck (16.67%): Static Resource Exhaustion
**Achieved Occupancy** is defined as the ratio of currently resident Warps to the
hardware's theoretical maximum. When this metric appears as a flat bottom line at
16.67%, it is almost always caused by **static resource exhaustion at launch time**,
specifically the Register File.

Using the NVIDIA Ampere architecture (e.g., RTX A6000) as reference, the hardware
allocation math is as follows:

1. **Resource demand**: If the Kernel logic is complex enough that each software
   Thread requires the maximum of 255 registers, then a single Warp (32 Threads)
   demands `255 × 32 = 8,160` registers.

2. **Allocation granularity**: The GPU allocates registers in units of 256. The
   demand of 8,160 is therefore rounded up to **8,192** physical registers per Warp.

3. **Capacity ceiling**: The A6000's SM has a total Register File of 64K (65,536)
   physical registers.
   - `65,536 / 8,192 = 8` Resident Warps maximum.

4. **Metric result**: Ampere's theoretical SM maximum is 48 Warps. With only 8
   actually resident: `8 / 48 = 16.67%`.

**Architectural implication**: The SM's register pool has been entirely consumed
by these 8 large Warps. All other Warps cannot be assigned a PC or register space,
and are blocked from entering the SM.

> **Gemini Comment:** The mathematical analysis here is perfectly accurate for Ampere SM86 (which has 64K physical registers and a max of 48 warps per SM). However, it's crucial to understand that for highly optimized Tensor Core kernels (like cuBLAS/cuSPARSELt GEMMs), this 16.67% occupancy is usually **intentional**. Instead of Thread-Level Parallelism (TLP) with many warps, these kernels rely on Instruction-Level Parallelism (ILP). The 255 registers are deliberately consumed to keep massive matrix tiles as close to the math units as possible, avoiding slow roundtrips to Shared Memory.

> **Claude Comment:** Two practical additions. First, the 255-register figure is the ISA hard maximum — real kernels rarely hit it exactly. Run `nvcc` with `--ptxas-options=-v` to see the actual register count; cuBLAS GEMM kernels on Ampere typically land in the 232–250 range, which still rounds up to 8,192 per warp and produces the same 16.67% result. The "flat bottom" value is a reliable diagnostic signature: 16.67% (= 8/48) specifically points to register exhaustion, while 20.83% (= 10/48) or 25% (= 12/48) would instead suggest a shared memory or block-count ceiling. Second, if the block size is also 256 threads (8 warps), the register limit and the block-resident-warp limit coincide, making it impossible to distinguish from the metric alone which constraint is actually binding — cross-check with the "Theoretical Occupancy" breakdown in Nsight Compute's "Launch Statistics" section.

---

## 3. Issue Slots Busy Analysis (10.65%): Frontend Pipeline Starvation
**Issue Slots Busy** represents the percentage of cycles in which the Warp
Scheduler successfully issued at least one instruction. With only 8 Resident
Warps available, the frontend suffers severe **pipeline starvation**.

- **Data Dependency Stall**: After a Warp issues an instruction (e.g., a Tensor
  MMA or Global Memory Load), the next instruction typically depends on that
  result. The hardware Scoreboard marks the Warp as **Stalled** until the result
  is written back to the register file (Register Write-back).

- **Frontend Bubble**: With only 8 Resident Warps, when all 8 enter a long-cycle
  Stall simultaneously, the Warp Scheduler has no ready Warp to issue from —
  it is left with nothing to do.

- **Metric result**: In a window of 100 cycles, the Scheduler only has active
  work during approximately 10–11 cycles (issuing one instruction per ready Warp).
  The remaining ~89 cycles produce no issues, as there are no backup Warps to
  fill the gap — yielding a metric of 10.65%.

> **Claude Comment:** The document treats "Data Dependency Stall" as a single cause, but Nsight Compute's **Warp State Statistics** disaggregates stalls into distinct hardware reasons: `Long Scoreboard` (waiting on slow memory ops), `Short Scoreboard` (register-to-register latency), `MIO Throttle` (shared memory / texture pipe saturated), `Wait` (explicit `bar.sync` or `cp.async` fence), and others. For a cuSPARSELt kernel, the dominant stall is almost certainly `Long Scoreboard` (waiting for Global→Shared async copies to land) or `Wait` (cp.async barriers between pipeline stages) — *not* a generic dependency stall. This distinction matters enormously for optimization: `Long Scoreboard` means the async copy pipeline needs more stages; `MIO Throttle` means shared memory bank conflicts or pipe saturation; `Wait` means the compute-to-copy overlap ratio is wrong. Knowing the issue rate is 10.65% tells you *that* the frontend is starved; the Warp State histogram tells you *why*.

---

## 4. The Pipeline Utilization Paradox (48.9%): Physical Decoupling of
##    Throughput and Latency
Why does an Issue Rate of only 10.65% coexist with a Pipeline Utilization of
48.9%? This arises from the asymmetry between a pipeline instruction's
**Initiation Interval (II)** and its **Instruction Latency**.

### Parameter Definitions

- **Issue Slot cost**: Issuing one instruction from the Warp Scheduler consumes
  **1 cycle** of frontend time.

- **Pipeline Throughput / Initiation Interval (II)**: A heavy instruction such
  as a Tensor MMA, once issued, occupies the pipeline's intake port for several
  consecutive cycles (e.g., **~4.5 cycles**). This means the pipeline's active
  duration is a multiple of the time spent issuing it.

- **Instruction Latency**: The total pipeline depth from when an instruction
  enters the pipeline to when the final result is written back to the physical
  register file (e.g., **~40 cycles**).

### Mathematical Model

Within a 100-cycle window:

1. **Issue phase**: The Scheduler issued approximately 11 Tensor instructions
   (Issue Slots Busy ≈ 11%).

2. **Execute phase**: Each of those 11 instructions kept the Tensor pipeline
   active for 4.5 cycles.
   - Calculation: `11 instructions × 4.5 cycles = 49.5 cycles`
   - Result: Tensor Pipeline active time ≈ **~49%**.

### Why the Warp Must Wait 40 Cycles Despite the Pipeline Being Free After 4.5

- **Pipeline intake released**: After 4.5 cycles, the Tensor pipeline's intake
  port is free and can accept an instruction from **a different Warp**.

- **Register Write-back still pending**: However, for the Warp that originally
  issued the instruction, its data is still traveling through a pipeline that is
  40 cycles deep. Until Write-back completes, the Warp cannot execute the next
  instruction that depends on that result.

- **The cost of the resource mismatch**: To fully hide this 40-cycle latency, the
  Scheduler needs approximately `40 / 4.5 ≈ 9` Ready Warps rotating through
  issue slots continuously. The system only has 8 Resident Warps — just short of
  what is needed to fully cover the latency — which leaves the remaining ~51% of
  pipeline cycles idle.

> **Gemini Comment:** The math effectively demonstrates the TLP (Thread-Level Parallelism) latency hiding model, but it overlooks **ILP (Instruction-Level Parallelism)**. A single large warp holding 255 registers can hide its *own* latency if it has enough independent instructions unrolled. For example, while waiting 40 cycles for an instruction to complete, the *same* warp can issue multiple other independent instructions in subsequent cycles, since they operate on different register tiles. If pipeline utilization stalls at 48.9%, the warp likely hit a barrier (e.g., waiting for the next data tile to load from Shared Memory) rather than purely lacking a 9th warp to switch to.

> **Claude Comment:** Two clarifications on the numbers. First, the ~4.5-cycle Initiation Interval is instruction- and architecture-specific. For Ampere FP16 `mma.sync.aligned.m16n8k16`, NVIDIA documents 256 FP16 FMA ops/cycle/SM, which back-calculates to a specific II for that shape. TF32 and FP8 shapes have different IIs, and on Ada Lovelace (SM89) the new `wgmma.mma_async` instruction is *asynchronous at the warp level* — the warp issues the wgmma and immediately proceeds to the next instruction without stalling on the 40-cycle write-back at all, making the II-vs-latency model here largely obsolete for Ada/Hopper. Second, "Pipeline Utilization" in Nsight Compute is not a single number — it is reported per execution pipeline. The 48.9% figure needs a qualifier: is it `sm__pipe_tensor_op_hmma_cycles_active` (Tensor HMMA), `sm__inst_executed_pipe_lsu` (Load/Store), or the aggregate SM utilization? A 48.9% Tensor pipeline utilization implies a different root cause and fix than 48.9% LSU utilization. Always confirm which pipeline the metric refers to before drawing conclusions.

> **Codex Comment:** The explanatory model is good, but I would soften the sentence "`40 / 4.5 ≈ 9` Ready Warps" from a conclusion into a conditional. That estimate only holds if latency hiding comes primarily from warp rotation and each warp exposes limited ILP. In real Tensor Core kernels, part of that 40-cycle gap is often hidden by instruction overlap inside the same warp, and part is exposed again at explicit stage boundaries (`cp.async.wait_group`, tile exchange, barrier points). A safer wording is: **"If the kernel cannot extract enough ILP within each warp, roughly 9 ready warps would be needed to cover a 40-cycle dependency chain with a 4.5-cycle initiation interval."** That keeps the math while avoiding the impression that an 8-warp kernel is automatically underprovisioned.

---

## 5. Summary and Optimization Direction

These three metrics form a tight causal chain:
High register pressure
→ Static resource exhaustion
→ Occupancy locked at 16.67%
→ Insufficient switchable Warps
→ Scheduler starved during long-latency stalls
→ Issue Rate drops to 10.65%
→ Backend pipeline cannot be continuously fed
→ Pipeline Utilization stalls at 48.9%


**Optimization implication**:  
In a compute-bound workload, improving Pipeline Utilization requires breaking the
frontend starvation. The concrete approach is to reduce per-Thread register
consumption — for example, by enforcing a register cap via `-maxrregcount` at
compile time, or by adjusting the Block Tiling size in the Kernel. By reducing
the "weight" of each Warp, more Warps can reside simultaneously in the SM. A
sufficient number of Resident Warps enables effective latency hiding across the
deep execution pipeline, ultimately driving Pipeline Utilization toward full
saturation.

> **Gemini Comment:** I **strongly disagree** with the recommendation to force higher occupancy by using `-maxrregcount` on Tensor Core GEMM kernels.
>
> Artificially capping a highly tuned kernel from 255 down to 128 or 64 registers will trigger massive **Register Spilling**. The compiler will be forced to dump the active matrix fragments to Local Memory (which physically resides in the L2 Cache / Global Memory). This replaces register-to-register latency (~1 cycle) with memory latency (~150+ cycles), which will instantly plummet your Pipeline Utilization and overall TFLOPS.
>
> **Alternative Optimization Direction:** For GEMMs stuck at a "low" issue rate but ~50% pipeline utilization, the primary bottleneck is usually **memory-feed starvation**, not warp starvation. Instead of sacrificing registers for warps, focus on:
> 1. **Data Movement Efficiency:** Maximize the bandwidth of data moving from Shared Memory into Registers using `ldmatrix` instructions and resolving Shared Memory bank conflicts.
> 2. **Software Pipelining:** Ensure the async copy pipeline (Global → Shared) overlapping with the compute pipeline (Shared → Register → MMA) is deep enough (e.g., 3-stage or 4-stage pipelines) so that compute units are never waiting for data.
> 3. **Block Tile Tuning:** Adjust the tile dimensions (e.g., from 128x128x32 to 256x128x64) within the register budget to find the mathematical sweet spot of ILP, rather than blindly forcing occupancy.

> **Claude Comment:** Agreeing with Gemini's objection, and adding one meta-point: the causal chain in Section 5 is a useful teaching model, but it presents a single linear path when the actual diagnosis should be driven by data. Before choosing between any of these strategies, the correct first step is to open Nsight Compute's **"Top Stall Reasons"** (under Warp State Statistics) and **"Memory Workload Analysis"** sections for the specific kernel. The chain `register pressure → low occupancy → low issue rate → low pipeline utilization` is one possible path, but a kernel could show the same three numbers for entirely different reasons — e.g., a poorly staged `cp.async` pipeline produces low issue rate and ~50% tensor utilization even with adequate warp count. Treating the causal chain as diagnostic ground truth risks optimizing the wrong bottleneck. The metrics are effects; the profiler's stall breakdown is the cause.

> **Codex Comment:** I would change the optimization paragraph from prescriptive to conditional. Right now it reads as if lower register count is the default fix, but for modern GEMM kernels that is usually the *last* lever to pull because spilling destroys the very locality the kernel was designed to buy with registers. A better framing would be: **"If profiler data shows the dominant limiter is insufficient eligible warps, then reducing register footprint or retuning tile shape may help. If the dominant limiter is async-copy staging, shared-memory throughput, or synchronization, higher occupancy alone will not recover the missing pipeline utilization."** That wording keeps the causal lesson without turning it into a one-size-fits-all recommendation.
