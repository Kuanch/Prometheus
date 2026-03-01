// gemm.cu — cuBLAS SGEMM benchmark
// Computes: C = alpha * A * B + beta * C
// Matrix dims: M x N = (M x K) * (K x N)
//
// Compile: nvcc -o gemm gemm.cu -lcublas -O2
// Run:     ./gemm
// Profile: nsys profile -o /output/gemm ./gemm
//          ncu --set full --export /output/gemm ./gemm

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ── helpers ─────────────────────────────────────────────────────────────────

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

static void fill_random(float *p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        p[i] = (float)rand() / RAND_MAX - 0.5f;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(void) {
    // Matrix dimensions (powers of 2, fits in 8 GB VRAM)
    const int M = 4096;   // rows of A and C
    const int N = 4096;   // cols of B and C
    const int K = 4096;   // cols of A / rows of B
    const int WARMUP = 5;
    const int ITERS  = 20;

    printf("cuBLAS SGEMM  M=%d  N=%d  K=%d\n", M, N, K);

    // ── host matrices ────────────────────────────────────────────────────────
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float *)malloc(bytesA);
    float *hB = (float *)malloc(bytesB);
    float *hC = (float *)malloc(bytesC);
    fill_random(hA, (size_t)M * K);
    fill_random(hB, (size_t)K * N);
    memset(hC, 0, bytesC);

    // ── device matrices ──────────────────────────────────────────────────────
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, bytesC, cudaMemcpyHostToDevice));

    // ── cuBLAS handle ────────────────────────────────────────────────────────
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;

    // cuBLAS uses column-major layout.
    // To compute C = A * B in row-major, we compute:
    //   C^T = B^T * A^T  →  swap A↔B and M↔N in the call.
    #define GEMM() \
        cublasSgemm(handle,           \
                    CUBLAS_OP_N,      \
                    CUBLAS_OP_N,      \
                    N, M, K,          \
                    &alpha,           \
                    dB, N,            \
                    dA, K,            \
                    &beta,            \
                    dC, N)

    // ── warm-up ──────────────────────────────────────────────────────────────
    for (int i = 0; i < WARMUP; ++i)
        CUBLAS_CHECK(GEMM());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── timed runs ───────────────────────────────────────────────────────────
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < ITERS; ++i)
        CUBLAS_CHECK(GEMM());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float ms_per_iter = ms / ITERS;

    // ── metrics ──────────────────────────────────────────────────────────────
    // FLOPs for one SGEMM: 2 * M * N * K  (multiply-add counts as 2)
    double flops     = 2.0 * M * N * K;
    double tflops    = flops / (ms_per_iter * 1e-3) / 1e12;

    // Memory traffic (read A + B, write C — no beta accumulation here)
    double bytes     = (double)(bytesA + bytesB + bytesC);
    double bw_GBs    = bytes / (ms_per_iter * 1e-3) / 1e9;

    printf("  avg latency : %.3f ms\n", ms_per_iter);
    printf("  TFLOPS      : %.2f  (RTX 4060 peak ~15 TFLOPS FP32)\n", tflops);
    printf("  memory BW   : %.1f GB/s  (theoretical ~256 GB/s)\n", bw_GBs);

    // ── cleanup ──────────────────────────────────────────────────────────────
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);

    return 0;
}
