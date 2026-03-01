// gemm_cuda.cu — Hand-written CUDA SGEMM kernels
// Three progressively optimized versions:
//   1. Naive:  1 thread = 1 output element, global memory only
//   2. Tiled:  32x32 shared memory tiles, reduces global traffic by 32x
//   3. Tiled + vectorized loads (float4) + register accumulation
//
// All compute: C = A * B   (row-major, M×K * K×N → M×N, FP32)

#include "gemm_cuda.cuh"

// ─── Kernel 1: Naive ────────────────────────────────────────────────────────
// Each thread reads an entire row of A and column of B from global memory.
// Arithmetic intensity: 2 FLOPs / (2 * 4B loads) = 0.25 FLOP/byte — memory bound.
// Global memory accesses: M*N*K reads of A + M*N*K reads of B = 2*M*N*K loads.

__global__ void sgemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ─── Kernel 2: Shared-memory tiling ─────────────────────────────────────────
// Loads TILE×TILE blocks of A and B into shared memory.
// Each element is reused TILE times → global traffic reduced by TILE×.
// Arithmetic intensity: ~2*TILE FLOPs/load ≈ 64 FLOP/byte (TILE=32).
// Bottleneck shifts toward compute, but still 1 output per thread (low ILP).

#define TILE 32

__global__ void sgemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ─── Kernel 3: Register-tiled (TM×TN per thread) ───────────────────────────
// Each thread computes a TM×TN sub-tile of C using registers.
// This increases arithmetic intensity per thread and ILP.
// A block computes (BM × BN) of C; each thread handles (TM × TN) elements.
// Threads per block: (BM/TM) × (BN/TN).
//
// Why this is faster than Kernel 2:
//   - More FLOPs per global load (higher register reuse)
//   - Better instruction-level parallelism (TM*TN independent FMAs)
//   - Fewer blocks needed → less launch overhead
//
// Still slower than cuBLAS because:
//   - No vectorized loads (float4)
//   - No double-buffering of shared memory tiles
//   - No software pipelining / async copies
//   - cuBLAS auto-tunes tile sizes per GPU architecture

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void sgemm_reg_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Block computes C[by*BM .. by*BM+BM-1, bx*BN .. bx*BN+BN-1]
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread position within the block's output tile
    // Block has (BM/TM) × (BN/TN) = 16×16 = 256 threads
    const int threadRow = threadIdx.x / (BN / TN);  // 0..15
    const int threadCol = threadIdx.x % (BN / TN);  // 0..15

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Register tile: each thread accumulates TM×TN results
    float regC[TM][TN] = {};

    // Pointers to this block's starting rows/cols in global A, B
    const float* baseA = A + by * BM * K;
    const float* baseB = B + bx * BN;

    // Number of threads in this block
    const int numThreads = (BM / TM) * (BN / TN);  // 256

    for (int bk = 0; bk < K; bk += BK) {
        // Cooperatively load A tile: BM×BK = 64×16 = 1024 floats, 256 threads → 4 each
        for (int i = threadIdx.x; i < BM * BK; i += numThreads) {
            int r = i / BK;
            int c = i % BK;
            int globalRow = by * BM + r;
            int globalCol = bk + c;
            As[r][c] = (globalRow < M && globalCol < K) ? baseA[r * K + bk + c] : 0.0f;
        }

        // Cooperatively load B tile: BK×BN = 16×64 = 1024 floats
        for (int i = threadIdx.x; i < BK * BN; i += numThreads) {
            int r = i / BN;
            int c = i % BN;
            int globalRow = bk + r;
            int globalCol = bx * BN + c;
            Bs[r][c] = (globalRow < K && globalCol < N) ? baseB[r * N + bk * N + c] : 0.0f;
        }

        __syncthreads();

        // Compute TM×TN partial results from this BK-wide slice
        for (int k = 0; k < BK; k++) {
            // Load TM values from A shared tile into registers
            float a[TM];
            for (int m = 0; m < TM; m++)
                a[m] = As[threadRow * TM + m][k];

            // Load TN values from B shared tile into registers
            float b[TN];
            for (int n = 0; n < TN; n++)
                b[n] = Bs[k][threadCol * TN + n];

            // Outer product: TM × TN FMAs
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    regC[m][n] += a[m] * b[n];
        }

        __syncthreads();
    }

    // Write TM×TN results to global memory
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int globalRow = by * BM + threadRow * TM + m;
            int globalCol = bx * BN + threadCol * TN + n;
            if (globalRow < M && globalCol < N)
                C[globalRow * N + globalCol] = regC[m][n];
        }
    }
}

// ─── Launch wrappers ────────────────────────────────────────────────────────

void launch_sgemm_naive(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    sgemm_naive<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_sgemm_tiled(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sgemm_tiled<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_sgemm_reg_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    dim3 block((BM / TM) * (BN / TN));  // 256 threads (1D)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_reg_tiled<<<grid, block>>>(A, B, C, M, N, K);
}
