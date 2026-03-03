// gemm_sparse_fp16.cu — FP16 2:4 Structured Sparse GEMM benchmark
//
// Compares three paths for C = A * B  (M=N=K=4096, FP16):
//   1. cuBLAS  cublasHgemm       — dense FP16 Tensor Cores
//   2. cuBLAS  cublasGemmEx      — dense FP16 with FP32 accumulation (TF32 path)
//   3. cuSPARSELt cusparseLtMatmul — 2:4 Sparse Tensor Cores (~2x peak)
//
// 2:4 sparsity rule: in every 4 *consecutive* values along the K dimension
// of matrix A, at least 2 must be zero.  cuSPARSELt's prune step enforces
// this automatically (zeroes the 2 smallest-magnitude values per group of 4).
// The compressed format stores the non-zero values + a 2-bit metadata mask,
// halving the memory traffic and enabling the dedicated sparse path on
// Ampere / Ada Lovelace Sparse Tensor Core units.
//
// Target peak (theoretical):
//   A6000  (Ampere SM86): ~312 TFLOPS FP16 dense → ~624 TFLOPS 2:4 sparse
//   RTX 4060L (Ada SM89):  ~33 TFLOPS FP16 dense →  ~66 TFLOPS 2:4 sparse
//
// Compile (inside Docker):
//   nvcc -o /output/gemm_sparse_fp16 \
//        /workspace/kernels/gemm/gemm_sparse_fp16.cu \
//        -lcublas -lcusparseLt -O2 -arch=sm_80
//
//   For Ada (RTX 40xx):  -arch=sm_89
//   For Ampere (A6000):  -arch=sm_86
//
// Run:
//   /output/gemm_sparse_fp16
//
// Profile:
//   nsys profile -o /output/gemm_sparse_fp16 /output/gemm_sparse_fp16
//   ncu --set basic -o /output/gemm_sparse_fp16_ncu /output/gemm_sparse_fp16

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cusparseLt.h>

// ── Error checking ────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d — status %d\n",                \
                    __FILE__, __LINE__, (int)_s);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUSPARSELT_CHECK(call)                                                  \
    do {                                                                        \
        cusparseLtStatus_t _s = (call);                                         \
        if (_s != CUSPARSE_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuSPARSELt error %s:%d — status %d\n",            \
                    __FILE__, __LINE__, (int)_s);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── Helpers ───────────────────────────────────────────────────────────────────

static void fill_random_fp16(__half *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
}

// Report sparsity ratio of an FP16 device buffer (informational only)
static float device_sparsity(const __half *d_ptr, size_t n) {
    __half *h = (__half *)malloc(n * sizeof(__half));
    cudaMemcpy(h, d_ptr, n * sizeof(__half), cudaMemcpyDeviceToHost);
    size_t zeros = 0;
    for (size_t i = 0; i < n; i++)
        if (__half2float(h[i]) == 0.0f) zeros++;
    free(h);
    return (float)zeros / (float)n * 100.0f;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(void) {
    const int M = 4096;   // rows of A, rows of C
    const int N = 4096;   // cols of B, cols of C
    const int K = 4096;   // cols of A, rows of B
    const int WARMUP = 10;   // enough warmup iterations to reach boost-clock steady state
    const int ITERS  = 50;   // 50 iters reduces timing noise from thermal jitter

    // 2 * M * N * K multiply-adds — same formula as dense, used for effective TFLOPS
    const double FLOPS = 2.0 * M * N * K;

    printf("=== FP16 GEMM Benchmark  M=%d  N=%d  K=%d ===\n\n", M, N, K);
    printf("Paths compared:\n");
    printf("  A. cuBLAS  cublasHgemm     — dense FP16, FP16 accumulate\n");
    printf("  B. cuBLAS  cublasGemmEx    — dense FP16, FP32 accumulate (default)\n");
    printf("  C. cuSPARSELt              — 2:4 sparse FP16, Sparse Tensor Cores\n\n");

    // ── Host allocation & fill ────────────────────────────────────────────────
    size_t bytesA = (size_t)M * K * sizeof(__half);
    size_t bytesB = (size_t)K * N * sizeof(__half);
    size_t bytesC = (size_t)M * N * sizeof(__half);

    __half *hA = (__half *)malloc(bytesA);
    __half *hB = (__half *)malloc(bytesB);
    srand(42);
    fill_random_fp16(hA, (size_t)M * K);
    fill_random_fp16(hB, (size_t)K * N);

    // ── Device allocation ─────────────────────────────────────────────────────
    __half *dA, *dB, *dC_dense, *dC_sparse;
    CUDA_CHECK(cudaMalloc(&dA,        bytesA));
    CUDA_CHECK(cudaMalloc(&dB,        bytesB));
    CUDA_CHECK(cudaMalloc(&dC_dense,  bytesC));
    CUDA_CHECK(cudaMalloc(&dC_sparse, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC_dense,  0, bytesC));
    CUDA_CHECK(cudaMemset(dC_sparse, 0, bytesC));

    // ── CUDA timing events ────────────────────────────────────────────────────
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

#define TIME_ITERS(record_start, body, record_end, out_ms)              \
    do {                                                                \
        /* warmup */                                                    \
        for (int _w = 0; _w < WARMUP; _w++) { body; }                  \
        CUDA_CHECK(cudaDeviceSynchronize());                            \
        CUDA_CHECK(cudaEventRecord(record_start));                      \
        for (int _i = 0; _i < ITERS; _i++) { body; }                   \
        CUDA_CHECK(cudaEventRecord(record_end));                        \
        CUDA_CHECK(cudaEventSynchronize(record_end));                   \
        float _ms; CUDA_CHECK(cudaEventElapsedTime(&_ms, t0, t1));     \
        (out_ms) = _ms / ITERS;                                        \
    } while (0)

    // ── A. cuBLAS dense FP16 — FP16 accumulation (cublasHgemm) ──────────────
    {
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        // cuBLAS is column-major: C = A*B (row-major) → C^T = B^T * A^T (col-major)
        // Swap A↔B, swap M↔N in the call.
        __half alpha = __float2half(1.0f);
        __half beta  = __float2half(0.0f);

        float ms;
        TIME_ITERS(t0,
            CUBLAS_CHECK(cublasHgemm(h,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha, dB, N, dA, K,
                &beta,  dC_dense, N)),
            t1, ms);

        printf("A. cublasHgemm (FP16 accum)\n");
        printf("   latency : %.3f ms\n", ms);
        printf("   TFLOPS  : %.2f\n\n", FLOPS / (ms * 1e-3) / 1e12);
        cublasDestroy(h);
    }

    // ── B. cuBLAS dense FP16 — FP32 accumulation (cublasGemmEx) ─────────────
    {
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        float alpha = 1.0f, beta = 0.0f;

        float ms;
        TIME_ITERS(t0,
            CUBLAS_CHECK(cublasGemmEx(h,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB, CUDA_R_16F, N,
                dA, CUDA_R_16F, K,
                &beta,
                dC_dense, CUDA_R_16F, N,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP)),
            t1, ms);

        printf("B. cublasGemmEx (FP32 accum, tensor ops)\n");
        printf("   latency : %.3f ms\n", ms);
        printf("   TFLOPS  : %.2f\n\n", FLOPS / (ms * 1e-3) / 1e12);
        cublasDestroy(h);
    }

    // ── C. cuSPARSELt 2:4 Sparse FP16 ───────────────────────────────────────
    {
        cusparseLtHandle_t            lt;
        cusparseLtMatDescriptor_t     matA, matB, matC;
        cusparseLtMatmulDescriptor_t  matmul;
        cusparseLtMatmulAlgSelection_t alg_sel;
        cusparseLtMatmulPlan_t        plan;

        CUSPARSELT_CHECK(cusparseLtInit(&lt));

        // ── Matrix descriptors (row-major, 16-byte alignment for FP16) ──────
        // A: M×K  sparse (structured 50% — 2:4 per row of 4 along K)
        // B: K×N  dense
        // C: M×N  dense  (both input accumulator and output D)
        const uint32_t align = 16;

        CUSPARSELT_CHECK(cusparseLtStructuredDescriptorInit(
            &lt, &matA,
            M, K, K,       // rows, cols, ld  (row-major → ld = cols)
            align, CUDA_R_16F,
            CUSPARSE_ORDER_ROW,
            CUSPARSELT_SPARSITY_50_PERCENT));

        CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(
            &lt, &matB,
            K, N, N,
            align, CUDA_R_16F,
            CUSPARSE_ORDER_ROW));

        CUSPARSELT_CHECK(cusparseLtDenseDescriptorInit(
            &lt, &matC,
            M, N, N,
            align, CUDA_R_16F,
            CUSPARSE_ORDER_ROW));

        // ── Matmul descriptor ────────────────────────────────────────────────
        // CUSPARSE_COMPUTE_16F: FP16 inputs + FP16 accumulation — closest to
        // the hardware's headline Sparse Tensor Core peak.  Matches cublasHgemm
        // (path A), making the dense-vs-sparse comparison apples-to-apples.
        // Use CUSPARSE_COMPUTE_32F for FP32 accumulation if accuracy matters.
        CUSPARSELT_CHECK(cusparseLtMatmulDescriptorInit(
            &lt, &matmul,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &matA, &matB, &matC, &matC,
            CUSPARSE_COMPUTE_16F));

        // ── Prune A to 2:4 structured sparsity ──────────────────────────────
        // CUSPARSELT_PRUNE_SPMMA_STRIP: for each group of 4 consecutive values
        // along K, the 2 with the smallest absolute value are set to zero.
        // This is done in-place on the device.
        printf("C. cuSPARSELt 2:4 sparse FP16 (FP16 accum — peak path, matches A)\n");
        printf("   Pruning A to 2:4 structured sparsity ...\n");

        CUSPARSELT_CHECK(cusparseLtSpMMAPrune(
            &lt, &matmul,
            dA, dA,        // in-place: prune dA to 2:4
            CUSPARSELT_PRUNE_SPMMA_STRIP,
            /*stream=*/0));
        CUDA_CHECK(cudaDeviceSynchronize());

#ifdef VERBOSE
        // device_sparsity does a full 32 MB D2H copy — gating it avoids
        // polluting GPU L2 cache state before the benchmark.
        float actual_sparsity = device_sparsity(dA, (size_t)M * K);
        printf("   Sparsity of A after pruning: %.1f%%  (target ≥50%%)\n",
               actual_sparsity);

        // pruneCheck adds another sync + D2H — useful for debug, not for perf runs.
        int *d_valid;
        CUDA_CHECK(cudaMalloc(&d_valid, sizeof(int)));
        CUSPARSELT_CHECK(cusparseLtSpMMAPruneCheck(
            &lt, &matmul, dA, d_valid, /*stream=*/0));
        CUDA_CHECK(cudaDeviceSynchronize());
        int h_valid = 0;
        CUDA_CHECK(cudaMemcpy(&h_valid, d_valid, sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_valid));
        // d_valid == 0  means the matrix IS valid 2:4 sparse
        printf("   2:4 validity check: %s\n",
               h_valid == 0 ? "PASSED" : "FAILED (unexpected)");
#endif

        // ── Algorithm selection & plan ───────────────────────────────────────
        CUSPARSELT_CHECK(cusparseLtMatmulAlgSelectionInit(
            &lt, &alg_sel, &matmul,
            CUSPARSELT_MATMUL_ALG_DEFAULT));

        CUSPARSELT_CHECK(cusparseLtMatmulPlanInit(
            &lt, &plan, &matmul, &alg_sel));

        // ── Compress A into 2:4 packed format ────────────────────────────────
        // Compressed layout stores only the 2 non-zeros per group of 4 plus
        // a 2-bit metadata mask — halves the memory footprint of A.
        size_t compressed_size     = 0;
        size_t compressed_buf_size = 0;
        CUSPARSELT_CHECK(cusparseLtSpMMACompressedSize(
            &lt, &plan, &compressed_size, &compressed_buf_size));

        void *dA_compressed, *dA_compress_buf;
        CUDA_CHECK(cudaMalloc(&dA_compressed,  compressed_size));
        CUDA_CHECK(cudaMalloc(&dA_compress_buf, compressed_buf_size));

        CUSPARSELT_CHECK(cusparseLtSpMMACompress(
            &lt, &plan,
            dA,              // pruned dense A
            dA_compressed,   // output: values + metadata
            dA_compress_buf, // scratch
            /*stream=*/0));
        CUDA_CHECK(cudaFree(dA_compress_buf));

        printf("   Compressed A size: %.2f MB  (dense A: %.2f MB)\n",
               (double)compressed_size / (1024 * 1024),
               (double)bytesA          / (1024 * 1024));

        // ── Workspace (initial, for DEFAULT algorithm) ────────────────────────
        size_t ws_size = 0;
        CUSPARSELT_CHECK(cusparseLtMatmulGetWorkspace(&lt, &plan, &ws_size));
        void *d_workspace = nullptr;
        if (ws_size > 0)
            CUDA_CHECK(cudaMalloc(&d_workspace, ws_size));

        // ── Algorithm search ─────────────────────────────────────────────────
        // Tries every available tile/stage/split-K combo for this exact
        // (M,N,K,FP16,ROW) shape and updates the plan with the fastest one.
        // This is the single biggest sparse-path win on Ampere/Ada.
        // Compressed A must already exist before calling this.
        printf("   Running algorithm search ...\n");
        float alpha_f = 1.0f, beta_f = 0.0f;
        CUSPARSELT_CHECK(cusparseLtMatmulSearch(
            &lt, &plan,
            &alpha_f,
            dA_compressed, dB,
            &beta_f,
            dC_sparse, dC_sparse,
            d_workspace,
            /*streams=*/nullptr, /*numStreams=*/0));

        // Re-query workspace: the winning algorithm may need more space
        size_t ws_size_best = 0;
        CUSPARSELT_CHECK(cusparseLtMatmulGetWorkspace(&lt, &plan, &ws_size_best));
        if (ws_size_best > ws_size) {
            if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
            CUDA_CHECK(cudaMalloc(&d_workspace, ws_size_best));
            printf("   Workspace grown: %zu → %zu bytes\n", ws_size, ws_size_best);
        }

        // ── Benchmark ────────────────────────────────────────────────────────
        float ms;
        TIME_ITERS(t0,
            CUSPARSELT_CHECK(cusparseLtMatmul(
                &lt, &plan,
                &alpha_f,
                dA_compressed, dB,
                &beta_f,
                dC_sparse,     // C input  (unused when beta=0)
                dC_sparse,     // D output
                d_workspace,
                /*streams=*/nullptr, /*numStreams=*/0)),
            t1, ms);

        // Dense-equivalent TFLOPS (2*M*N*K): apples-to-apples vs cuBLAS paths.
        // Actual sparse TFLOPS (M*N*K): real MACs executed — 50% zeros skipped.
        double tflops_dense_equiv  = FLOPS        / (ms * 1e-3) / 1e12;
        double tflops_actual_sparse = (FLOPS/2.0) / (ms * 1e-3) / 1e12;
        printf("   latency : %.3f ms\n", ms);
        printf("   TFLOPS  : %.2f  (dense-equiv 2*M*N*K — compare vs A and B)\n",
               tflops_dense_equiv);
        printf("   TFLOPS  : %.2f  (actual sparse MACs, M*N*K — hw saturation check)\n",
               tflops_actual_sparse);

        // ── Cleanup ──────────────────────────────────────────────────────────
        if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
        CUDA_CHECK(cudaFree(dA_compressed));
        CUSPARSELT_CHECK(cusparseLtMatmulPlanDestroy(&plan));
        CUSPARSELT_CHECK(cusparseLtDestroy(&lt));
    }

    // ── Final notes ───────────────────────────────────────────────────────────
    printf("\n=== Notes ===\n");
    printf("Path A (cublasHgemm FP16 accum) and path C (cuSPARSELt FP16 accum)\n");
    printf("use the same accumulation type — true apples-to-apples comparison.\n");
    printf("Dense-equiv TFLOPS (2*M*N*K) is the standard marketing metric.\n");
    printf("Actual sparse TFLOPS (M*N*K) tells you if you saturated the sparse\n");
    printf("Tensor Core path — if actual ≈ 0.5 * dense FP16 peak, you're there.\n");
    printf("\n");
    printf("2:4 sparsity trade-off:\n");
    printf("  + Up to 2x throughput over dense FP16 Tensor Cores\n");
    printf("  + 50%% compressed — smaller A footprint, lower DRAM BW\n");
    printf("  - Requires pruning A; small accuracy loss from zeroed weights\n");
    printf("  - Only A can be sparse (not B); requires SM80+ (Ampere/Ada)\n");
    printf("  - Build with -DVERBOSE to enable sparsity check + pruneCheck\n");

    // ── Global cleanup ────────────────────────────────────────────────────────
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC_dense));
    CUDA_CHECK(cudaFree(dC_sparse));
    free(hA);
    free(hB);

    return 0;
}
