// benchmark.cu — cuBLAS vs hand-written CUDA SGEMM comparison
//
// Compiles: nvcc -o /output/benchmark benchmark.cu gemm_cuda.cu -lcublas -O2
// Run:      /output/benchmark
// Profile:  nsys profile -o /output/gemm_cuda /output/benchmark
//
// Tests four kernels at M=N=K=4096 FP32:
//   1. Naive       — 1 thread per output, global memory only
//   2. Tiled       — 32×32 shared memory tiles
//   3. Reg-tiled   — 64×64 block, 4×4 register tile per thread
//   4. cuBLAS      — NVIDIA's autotuned implementation
//
// Also verifies correctness against cuBLAS result.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gemm_cuda.cuh"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                        \
                    __FILE__, __LINE__, (int)st);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ── Helpers ─────────────────────────────────────────────────────────────────

static void fill_random(float *p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = (float)rand() / RAND_MAX - 0.5f;
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float maxd = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// Benchmark one kernel: warmup + timed iterations, return ms/iter
typedef void (*launch_fn)(const float*, const float*, float*, int, int, int);

static float bench_kernel(launch_fn fn, const float* dA, const float* dB, float* dC,
                          int M, int N, int K, int warmup, int iters) {
    for (int i = 0; i < warmup; i++)
        fn(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++)
        fn(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms / iters;
}

// cuBLAS wrapper matching the launch_fn signature (uses a global handle)
static cublasHandle_t g_handle;

static void launch_cublas(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major trick: compute C^T = B^T * A^T
    CUBLAS_CHECK(cublasSgemm(g_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, B, N, A, K,
                             &beta, C, N));
}

// ── Main ────────────────────────────────────────────────────────────────────

int main() {
    const int M = 4096, N = 4096, K = 4096;
    const int WARMUP = 5, ITERS = 20;

    printf("=== SGEMM Benchmark  M=%d  N=%d  K=%d  FP32 ===\n\n", M, N, K);

    // Allocate host
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hRef = (float*)malloc(bytesC);   // cuBLAS reference result
    float *hTest = (float*)malloc(bytesC);  // test kernel result
    srand(42);
    fill_random(hA, (size_t)M * K);
    fill_random(hB, (size_t)K * N);

    // Allocate device
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    CUBLAS_CHECK(cublasCreate(&g_handle));

    double flops = 2.0 * M * N * K;

    // Kernel list
    struct { const char* name; launch_fn fn; } kernels[] = {
        {"1. Naive (global mem)",   launch_sgemm_naive},
        {"2. Tiled (shared mem)",   launch_sgemm_tiled},
        {"3. Reg-tiled (4x4)",     launch_sgemm_reg_tiled},
        {"4. cuBLAS",              launch_cublas},
    };
    int nkernels = sizeof(kernels) / sizeof(kernels[0]);

    // Get cuBLAS reference result for correctness check
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));
    launch_cublas(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef, dC, bytesC, cudaMemcpyDeviceToHost));

    // Print header
    printf("%-25s %10s %10s %10s %12s\n",
           "Kernel", "ms/iter", "TFLOPS", "vs cuBLAS", "max |err|");
    printf("%-25s %10s %10s %10s %12s\n",
           "-------------------------", "----------", "----------", "----------", "------------");

    float cublas_ms = 0.0f;

    for (int i = 0; i < nkernels; i++) {
        // Clear C
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));

        // Benchmark
        float ms = bench_kernel(kernels[i].fn, dA, dB, dC, M, N, K, WARMUP, ITERS);
        double tflops = flops / (ms * 1e-3) / 1e12;

        if (i == nkernels - 1) cublas_ms = ms;

        // Correctness: copy result back and compare to cuBLAS
        CUDA_CHECK(cudaMemcpy(hTest, dC, bytesC, cudaMemcpyDeviceToHost));
        float err = max_abs_diff(hRef, hTest, (size_t)M * N);

        // Speed relative to cuBLAS (printed after cuBLAS runs)
        char speedup_buf[32];
        if (i == nkernels - 1) {
            // cuBLAS itself: store its time, label as "baseline"
            snprintf(speedup_buf, sizeof(speedup_buf), "baseline");
        } else {
            snprintf(speedup_buf, sizeof(speedup_buf), "—");
        }

        printf("%-25s %8.3f ms %8.2f T %10s %12.6f\n",
               kernels[i].name, ms, tflops, speedup_buf, err);
    }

    // Print speedup column now that we have cuBLAS time
    printf("\n--- Relative to cuBLAS (%.3f ms) ---\n", cublas_ms);
    for (int i = 0; i < nkernels - 1; i++) {
        CUDA_CHECK(cudaMemset(dC, 0, bytesC));
        float ms = bench_kernel(kernels[i].fn, dA, dB, dC, M, N, K, WARMUP, ITERS);
        printf("  %-25s %.1fx slower\n", kernels[i].name, ms / cublas_ms);
    }

    // Performance analysis
    printf("\n=== Why cuBLAS Wins ===\n");
    printf("\n");
    printf("Naive → Tiled:\n");
    printf("  Shared memory eliminates redundant global loads.\n");
    printf("  Each element loaded once per tile vs once per output.\n");
    printf("  Global memory traffic: reduced ~32x (TILE_SIZE).\n");
    printf("\n");
    printf("Tiled → Reg-tiled:\n");
    printf("  Each thread computes 4x4 outputs instead of 1x1.\n");
    printf("  More FMAs per shared memory load (higher arithmetic intensity).\n");
    printf("  Better instruction-level parallelism (16 independent accumulators).\n");
    printf("\n");
    printf("Reg-tiled → cuBLAS (remaining gap):\n");
    printf("  - Vectorized loads (float4/LDG.128): 4x fewer memory transactions\n");
    printf("  - Double-buffered shared memory: overlaps loads with compute\n");
    printf("  - Software pipelining / async copies (cp.async on SM80+)\n");
    printf("  - Auto-tuned tile sizes per GPU arch (Ada: different from Ampere)\n");
    printf("  - Warp-level primitives (wmma/mma.sync for tensor cores)\n");
    printf("  - Bank-conflict-free shared memory layout (swizzling)\n");
    printf("  - TF32 tensor core path (19-bit mantissa, ~2x FP32 throughput)\n");

    // Cleanup
    cublasDestroy(g_handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hRef); free(hTest);

    return 0;
}
