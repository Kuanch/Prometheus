# cuSPARSELt Optimization Plan for A6000 (Ampere SM86)

This document outlines a detailed optimization plan for achieving peak theoretical FLOPs on an A6000 GPU using the `cuSPARSELt` library on gemm_sparse_fp16.cu for 2:4 structured sparsity.

## 1. The "Low Occupancy" Red Herring

**Observation:** Profiling with Nsight Compute (`ncu`) may show low occupancy (e.g., only 2 warps per warp scheduler, or 8 warps per SM).
**Analysis:** This is intentional and expected for highly optimized Tensor Core GEMM kernels. To maximize Instruction-Level Parallelism (ILP), these kernels cache massive tiles of matrices directly in thread registers (using up to ~255 registers per thread) to eliminate Shared Memory bottlenecks.
**Action:** Do not artificially attempt to increase occupancy. Doing so will lead to register spilling to local memory, thrashing the L1 cache, and severely degrading performance. The true bottlenecks lie elsewhere.

## 2. Fix the Algorithm Search Workspace (Critical Bug)

**Observation:** In the current implementation, `cusparseLtMatmulSearch` is constrained by a minimal workspace allocation.
**Analysis:** The `cusparseLtMatmulGetWorkspace` function is called with only the `DEFAULT` fallback configuration. As a result, the internal search function fails or skips the most highly tuned algorithms (like Split-K implementations) that require significantly larger workspaces to achieve peak performance.
**Action:** Query the maximum workspace size across *all* available algorithm configurations before allocating and running the search.

**Implementation Steps:**
1. Find the maximum configuration ID using `CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID`.
2. Loop through all algorithms from `0` to `alg_id_max`.
3. For each algorithm, set the attribute and query its required workspace size.
4. Keep track of the maximum workspace size found.
5. Allocate this `max_ws_size` to `d_workspace`.
6. Run the search with this adequately sized workspace.

## 3. Mitigate the "Tail Effect" (Wave Quantization)

**Observation:** Matrix dimensions `M=4096, N=4096, K=4096` may lead to poor hardware utilization near the end of the compute cycle.
**Analysis:** cuSPARSELt breaks the matrix into Thread Block tiles (e.g., `128x128`). This translates to 1024 total Thread Blocks. The A6000 has 84 SMs, so `1024 / 84 ≈ 12.19` waves of execution. For the first 12 waves, all SMs are fully saturated. For the final 0.19 wave, only 16 SMs are working while the remaining 68 SMs sit idle, significantly pulling down the average TFLOP/s.
**Action:**
- **Short-term:** Unlock Split-K algorithms (via step 2), which typically fragment the compute into more blocks, better distributing the load.
- **Long-term:** When integrating into real workloads, consider padding the M or N dimensions so that the total number of blocks is closer to a clean multiple of 84.

## 4. Optimize Memory Layouts for Tensor Cores

**Observation:** The matrices A, B, and C are all initialized using `CUSPARSE_ORDER_ROW` (Row-Major).
**Analysis:** Hardware Tensor Cores are designed to ingest matrix A in row slices and matrix B in column slices to perform dot products most efficiently. If matrix B is provided in Row-Major order, the kernel is forced to manually transpose it in Shared Memory on the fly, consuming valuable cycles and bandwidth.
**Action:** Switch the descriptor for matrix B to `CUSPARSE_ORDER_COL` (Column-Major).
*Note: Depending on how the data is produced upstream, consider doing this transposition offline or fusing it with previous operations.*

## 5. Account for Power Throttling and Clock Drops

**Observation:** The theoretical peak of 624 Tensor TFLOPS assumes the GPU sustains its maximum boost clock (~1800 MHz).
**Analysis:** Heavy 4096-dimension GEMM operations rapidly consume the 300W peak power budget. The GPU will downclock to base frequencies (~1200 MHz) to stay within thermal limits. If this happens, your theoretical ceiling also drops proportionally (e.g., to ~415 TFLOPS). Achieving 220–280 TFLOPS under these conditions implies >60% hardware saturation, which is actually highly efficient.
**Action:** Use `nvidia-smi dmon -s pctu` to monitor the `sm_clk` (clock speed) and `pwr` (power draw) in real-time during the benchmark. If the clock drops below 1500 MHz, your performance is physically bound by the cooling/power constraints, rather than suboptimal CUDA code.

---

## Review: Claude (Sonnet 4.6)

**Section 1 — Low Occupancy:** Correct in principle. The register-heavy ILP trade-off is real for cuBLAS/cuSPARSELt kernels. The phrasing "do not try to increase occupancy" is directionally right, but occupancy itself isn't the issue — tensor core pipe utilization (`sm__pipe_tensor_op_hmma_cycles_active`) is the real metric to watch. A kernel can have low occupancy and still be underperforming if the tensor pipes are idle.

**Section 2 — Algorithm Search Workspace:** The bug is real. Looking at the code (line 318), `GetWorkspace` is called on the default plan before the search, so the workspace may be undersized for higher-performing algorithms. However, codifying a `0..alg_id_max` sweep needs care — check your installed cuSPARSELt header for the exact `CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID` attribute enum, as the API is versioned and not all IDs may be valid.

**Section 3 — Tail Effect:** The math is correct: `1024 / 84 ≈ 12.19`, and the last partial wave has only 16/84 ≈ 19% of SMs working. But the throughput penalty is `0.19 / 12.19 ≈ 1.6%` — minor at this problem size. The real value of unlocking Split-K is more about exposing more compute parallelism along K, not just tail mitigation.

**Section 4 — Memory Layout:** Valid concern, but the fix is not just a descriptor flag change. If you set `matB` to `CUSPARSE_ORDER_COL`, the physical buffer must also be in column-major layout and the leading dimension must be updated to `K` (not `N`). Otherwise the addresses will be wrong. Test, don't assume it's a free win.

**Section 5 — Power Throttling:** Solid and practical. One addition: lock clocks before benchmarking for reproducibility with `nvidia-smi -i 0 -lgc 1800,1800` (lock graphics clock). This eliminates clock noise across runs and makes comparisons meaningful. Remember to reset with `-rgc` afterward.

**Missing from the plan:**
- Use `cublasLtMatmul` with the same FP16/FP32-acc config as a baseline before attributing any gap to cuSPARSELt specifically.
- Separate one-time setup costs (`prune`, `compress`, `search`) from the steady-state benchmark loop — the current code already does this, but the plan doesn't call it out explicitly as a requirement.

---

## Review: Codex (gpt-5.3-codex)

**Section 1 — Low Occupancy:** Correct that low occupancy is normal for register-heavy Tensor Core kernels. Concern: "Do not try to increase occupancy" is too absolute. The right rule is: do not trade away register locality *unless profiling shows register pressure is the limiter*. A low-warp kernel can still underperform if `smsp__issue_active`, tensor pipe utilization, or eligible warps are poor.

**Section 2 — Algorithm Search Workspace:** The search limitation is confirmed in the code (lines 316 and 330). However, this is a performance limitation, not a "critical bug" — correctness is fine because the code re-queries workspace after search and grows it for the chosen kernel. The risk is only that search never considered larger-workspace candidates. The `0..alg_id_max` sweep is too hand-wavy; cuSPARSELt's alg APIs are version-specific and not always exposed as a simple dense ID range.

**Section 3 — Tail Effect:** Concept is real, but impact is overstated. The last partial wave is `0.19 / 12.19 ≈ 1.5%` idealized efficiency loss. The 1024-block estimate also assumes a specific `128×128` tile shape — cuSPARSELt may pick different tiles, Split-K, or persistent-style kernels, so the math should be verified against the actual kernel launch.

**Section 4 — Memory Layout:** Switching `B` to column-major is not just a descriptor tweak. The physical storage and leading dimension must match the declared layout (the current code uses `ld = N` for row-major). "Tensor Cores want B in column-major" is too simplistic — cuSPARSELt can handle multiple layout/op combinations. Frame this as "test alternative layout/op pairs" rather than a guaranteed optimization.

**Section 5 — Power Throttling:** Correct. Additional recommendation: also lock clocks and record throttle reasons (via `nvidia-smi -q -d PERFORMANCE`) for stable run-to-run comparisons. The "624 TFLOPS" headline is a rough upper bound that depends on clock, counting convention, and sparse-throughput assumptions.

**Missing from the plan (Codex additions):**
- Separate one-time costs (`prune`, `compress`, `search`) from the timed steady-state matmul — the sample already does this reasonably well.
- Establish a fair dense baseline (`cublasLt`, same datatype/accumulation/layout policy) before attributing performance gaps purely to cuSPARSELt behavior.
