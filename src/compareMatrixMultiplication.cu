#include <gputk.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <fp8_utils.cuh>
#include <fp8_cublaslt.cuh>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_SIZE 256

#define gpuTKCheck(stmt)                                                  \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                     \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));  \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// ─── FP32 kernel ─────────────────────────────────────────────────────────────

__global__ void matrixMultiplyFP32(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ─── FP16 kernels ─────────────────────────────────────────────────────────────

__global__ void convertFP32ToFP16(
    const float* src, half* dst, int srcRows, int srcCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < srcRows && col < srcCols)
        dst[row * srcCols + col] = __float2half(src[row * srcCols + col]);
}

__global__ void convertAndPadFP32ToFP16(
    const float* __restrict__ src, half* __restrict__ dst,
    int srcRows, int srcCols, int dstCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < srcRows && col < srcCols)
        dst[row * dstCols + col] = __float2half(src[row * srcCols + col]);
}

__global__ void matrixMultiplyFP16Naive(
    const half* A, const half* B, float* C,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        half sum = __float2half(0.0f);
        for (int k = 0; k < K; k++)
            sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
        C[row * N + col] = __half2float(sum);
    }
}

__global__ void matrixMultiplyWMMA(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN * WMMA_N, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N, c_frag, N, wmma::mem_row_major);
}

// ─── FP8 GEMM kernels ────────────────────────────────────────────────────────

template <typename FP8>
__global__ void matrixMultiplyFP8PerTensor(
    const FP8* __restrict__ A,
    const FP8* __restrict__ B,
    float* __restrict__ C,
    float invScaleA, float invScaleB,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += (float)A[row * K + k] * (float)B[k * N + col];
        C[row * N + col] = sum * invScaleA * invScaleB;
    }
}

template <typename FP8>
__global__ void matrixMultiplyFP8PerRowCol(
    const FP8*   __restrict__ A,
    const FP8*   __restrict__ B,
    float*       __restrict__ C,
    const float* __restrict__ invScaleA,
    const float* __restrict__ invScaleB,
    int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += (float)A[row * K + k] * (float)B[k * N + col];
        C[row * N + col] = sum * invScaleA[row] * invScaleB[col];
    }
}

// ─── FP8 pipeline helpers ────────────────────────────────────────────────────

template <typename FP8>
int runFP8PerTensor(const float* dA_fp32, const float* dB_fp32,
                    float* hostC, int M, int K, int N,
                    double* elapsed_ms)
{
    int sizeA = M * K, sizeB = K * N, sizeC = M * N;
    int threads = BLOCK_SIZE;

    FP8   *dA_fp8, *dB_fp8;
    float *dC, *dMaxA, *dMaxB;

    gpuTKCheck(cudaMalloc(&dA_fp8, sizeA * sizeof(FP8)));
    gpuTKCheck(cudaMalloc(&dB_fp8, sizeB * sizeof(FP8)));
    gpuTKCheck(cudaMalloc(&dC,     sizeC * sizeof(float)));
    gpuTKCheck(cudaMalloc(&dMaxA,  sizeof(float)));
    gpuTKCheck(cudaMalloc(&dMaxB,  sizeof(float)));
    gpuTKCheck(cudaMemset(dMaxA, 0, sizeof(float)));
    gpuTKCheck(cudaMemset(dMaxB, 0, sizeof(float)));

    findMaxAbsKernel<<<min(256, (sizeA + threads - 1) / threads), threads, threads * sizeof(float)>>>(dA_fp32, dMaxA, sizeA);
    findMaxAbsKernel<<<min(256, (sizeB + threads - 1) / threads), threads, threads * sizeof(float)>>>(dB_fp32, dMaxB, sizeB);
    gpuTKCheck(cudaDeviceSynchronize());

    float hMaxA, hMaxB;
    gpuTKCheck(cudaMemcpy(&hMaxA, dMaxA, sizeof(float), cudaMemcpyDeviceToHost));
    gpuTKCheck(cudaMemcpy(&hMaxB, dMaxB, sizeof(float), cudaMemcpyDeviceToHost));

    float scaleA = computeQuantScale<FP8>(hMaxA);
    float scaleB = computeQuantScale<FP8>(hMaxB);

    quantizeFP32toFP8<FP8><<<(sizeA + threads - 1) / threads, threads>>>(dA_fp32, dA_fp8, scaleA, sizeA);
    quantizeFP32toFP8<FP8><<<(sizeB + threads - 1) / threads, threads>>>(dB_fp32, dB_fp8, scaleB, sizeB);
    gpuTKCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    cudaEventRecord(start);
    matrixMultiplyFP8PerTensor<FP8><<<grid, block>>>(dA_fp8, dB_fp8, dC, 1.0f / scaleA, 1.0f / scaleB, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *elapsed_ms = (double)ms;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    gpuTKCheck(cudaMemcpy(hostC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    gpuTKCheck(cudaFree(dA_fp8));
    gpuTKCheck(cudaFree(dB_fp8));
    gpuTKCheck(cudaFree(dC));
    gpuTKCheck(cudaFree(dMaxA));
    gpuTKCheck(cudaFree(dMaxB));
    return 0;
}

template <typename FP8>
int runFP8PerRowCol(const float* dA_fp32, const float* dB_fp32,
                    float* hostC, int M, int K, int N,
                    double* elapsed_ms)
{
    int sizeA = M * K, sizeB = K * N, sizeC = M * N;
    int threads = BLOCK_SIZE;

    FP8   *dA_fp8, *dB_fp8;
    float *dC, *dScalesA, *dScalesB;

    gpuTKCheck(cudaMalloc(&dA_fp8,   sizeA * sizeof(FP8)));
    gpuTKCheck(cudaMalloc(&dB_fp8,   sizeB * sizeof(FP8)));
    gpuTKCheck(cudaMalloc(&dC,       sizeC * sizeof(float)));
    gpuTKCheck(cudaMalloc(&dScalesA, M * sizeof(float)));
    gpuTKCheck(cudaMalloc(&dScalesB, N * sizeof(float)));

    findMaxAbsPerRowKernel<<<M, threads, threads * sizeof(float)>>>(dA_fp32, dScalesA, M, K);
    findMaxAbsPerColKernel<<<N, threads, threads * sizeof(float)>>>(dB_fp32, dScalesB, K, N);
    gpuTKCheck(cudaDeviceSynchronize());

    maxAbsToScales<FP8><<<(M + threads - 1) / threads, threads>>>(dScalesA, M);
    maxAbsToScales<FP8><<<(N + threads - 1) / threads, threads>>>(dScalesB, N);
    gpuTKCheck(cudaDeviceSynchronize());

    {
        dim3 block(16, 16);
        dim3 gridA((K + 15) / 16, (M + 15) / 16);
        dim3 gridB((N + 15) / 16, (K + 15) / 16);
        quantizeFP32toFP8PerRow<FP8><<<gridA, block>>>(dA_fp32, dA_fp8, dScalesA, M, K);
        quantizeFP32toFP8PerCol<FP8><<<gridB, block>>>(dB_fp32, dB_fp8, dScalesB, K, N);
    }
    gpuTKCheck(cudaDeviceSynchronize());

    invertScales<<<(M + threads - 1) / threads, threads>>>(dScalesA, M);
    invertScales<<<(N + threads - 1) / threads, threads>>>(dScalesB, N);
    gpuTKCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    cudaEventRecord(start);
    matrixMultiplyFP8PerRowCol<FP8><<<grid, block>>>(dA_fp8, dB_fp8, dC, dScalesA, dScalesB, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    *elapsed_ms = (double)ms;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    gpuTKCheck(cudaMemcpy(hostC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    gpuTKCheck(cudaFree(dA_fp8));
    gpuTKCheck(cudaFree(dB_fp8));
    gpuTKCheck(cudaFree(dC));
    gpuTKCheck(cudaFree(dScalesA));
    gpuTKCheck(cudaFree(dScalesB));
    return 0;
}

// ─── Error metrics ───────────────────────────────────────────────────────────

void computeErrorMetrics(
    const float* ref, const float* result, int N,
    float* relL2, float* maxAbs)
{
    double l2_num = 0.0, l2_den = 0.0;
    *maxAbs = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(result[i] - ref[i]);
        if (diff > *maxAbs) *maxAbs = diff;
        l2_num += (double)diff * diff;
        l2_den += (double)ref[i] * ref[i];
    }
    *relL2 = (float)sqrt(l2_num / (l2_den + 1e-10));
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    gpuTKArg_t args;

    float *hostA, *hostB;
    float *deviceA_fp32, *deviceB_fp32;
    int numARows, numAColumns, numBRows, numBColumns;
    int numCRows, numCColumns;

    args = gpuTKArg_read(argc, argv);

    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows, &numBColumns);
    numCRows    = numARows;
    numCColumns = numBColumns;
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");

    gpuTKLog(TRACE, "A: ", numARows, " x ", numAColumns);
    gpuTKLog(TRACE, "B: ", numBRows, " x ", numBColumns);

    int M = numARows, K = numAColumns, N = numBColumns;
    int sizeC = numCRows * numCColumns;

    int M_pad = (M + 15) / 16 * 16;
    int K_pad = (K + 15) / 16 * 16;
    int N_pad = (N + 15) / 16 * 16;

    // Host output buffers
    float *hostC_fp32            = (float *)malloc(sizeC * sizeof(float));
    float *hostC_fp16_naive      = (float *)malloc(sizeC * sizeof(float));
    float *hostC_wmma            = (float *)malloc(sizeC * sizeof(float));
    float *hostC_e4m3_tensor     = (float *)malloc(sizeC * sizeof(float));
    float *hostC_e5m2_tensor     = (float *)malloc(sizeC * sizeof(float));
    float *hostC_e4m3_rowcol     = (float *)malloc(sizeC * sizeof(float));
    float *hostC_e5m2_rowcol     = (float *)malloc(sizeC * sizeof(float));
    float *hostC_e4m3_cublaslt   = (float *)malloc(sizeC * sizeof(float));
    float *hostC_e5m2_cublaslt   = (float *)malloc(sizeC * sizeof(float));

    // GPU buffers for FP32 inputs (shared across all kernels)
    gpuTKTime_start(GPU, "Allocating GPU memory");
    gpuTKCheck(cudaMalloc(&deviceA_fp32, numARows * numAColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceB_fp32, numBRows * numBColumns * sizeof(float)));

    float *deviceC_fp32, *deviceC_fp16_naive;
    half  *deviceA_half, *deviceB_half;
    half  *deviceA_half_pad, *deviceB_half_pad;
    float *deviceC_wmma_pad;

    gpuTKCheck(cudaMalloc(&deviceC_fp32,       sizeC * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceA_half,        numARows * numAColumns * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceB_half,        numBRows * numBColumns * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceC_fp16_naive,  sizeC * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceA_half_pad,    M_pad * K_pad * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceB_half_pad,    K_pad * N_pad * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceC_wmma_pad,    M_pad * N_pad * sizeof(float)));
    gpuTKCheck(cudaMemset(deviceA_half_pad, 0,  M_pad * K_pad * sizeof(half)));
    gpuTKCheck(cudaMemset(deviceB_half_pad, 0,  K_pad * N_pad * sizeof(half)));
    gpuTKCheck(cudaMemset(deviceC_wmma_pad, 0,  M_pad * N_pad * sizeof(float)));
    gpuTKTime_stop(GPU, "Allocating GPU memory");

    gpuTKTime_start(GPU, "Copying input memory to GPU");
    gpuTKCheck(cudaMemcpy(deviceA_fp32, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKCheck(cudaMemcpy(deviceB_fp32, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKTime_stop(GPU, "Copying input memory to GPU");

    // cuBLASLt handle (shared across FP8 tensor core runs)
    cublasLtHandle_t lt_handle;
    cublasLtCreate(&lt_handle);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    float ms = 0;
    double times[9] = {0};

    // --- FP32 GEMM ---
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        cudaEventRecord(ev_start);
        matrixMultiplyFP32<<<grid, block>>>(deviceA_fp32, deviceB_fp32, deviceC_fp32, M, K, N);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        times[0] = ms;
    }
    gpuTKCheck(cudaMemcpy(hostC_fp32, deviceC_fp32, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Naive FP16 GEMM ---
    {
        dim3 block(16, 16);
        dim3 gridA((K + 15) / 16, (M + 15) / 16);
        dim3 gridB((N + 15) / 16, (K + 15) / 16);
        convertFP32ToFP16<<<gridA, block>>>(deviceA_fp32, deviceA_half, M, K);
        convertFP32ToFP16<<<gridB, block>>>(deviceB_fp32, deviceB_half, K, N);
        cudaDeviceSynchronize();

        dim3 grid((N + 15) / 16, (M + 15) / 16);
        cudaEventRecord(ev_start);
        matrixMultiplyFP16Naive<<<grid, block>>>(deviceA_half, deviceB_half, deviceC_fp16_naive, M, K, N);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        times[1] = ms;
    }
    gpuTKCheck(cudaMemcpy(hostC_fp16_naive, deviceC_fp16_naive, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // --- WMMA Tensor Core GEMM ---
    {
        dim3 block(16, 16);
        dim3 gridA((K + 15) / 16, (M + 15) / 16);
        dim3 gridB((N + 15) / 16, (K + 15) / 16);
        convertAndPadFP32ToFP16<<<gridA, block>>>(deviceA_fp32, deviceA_half_pad, M, K, K_pad);
        convertAndPadFP32ToFP16<<<gridB, block>>>(deviceB_fp32, deviceB_half_pad, K, N, N_pad);
        cudaDeviceSynchronize();

        dim3 grid(N_pad / WMMA_N, M_pad / WMMA_M);
        dim3 warp_block(32);
        cudaEventRecord(ev_start);
        matrixMultiplyWMMA<<<grid, warp_block>>>(deviceA_half_pad, deviceB_half_pad, deviceC_wmma_pad, M_pad, K_pad, N_pad);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        times[2] = ms;
    }
    gpuTKCheck(cudaMemcpy2D(
        hostC_wmma,                      // dst
        numCColumns * sizeof(float),     // dst pitch
        deviceC_wmma_pad,                // src
        N_pad * sizeof(float),           // src pitch
        numCColumns * sizeof(float),     // width in bytes
        numCRows,                        // height
        cudaMemcpyDeviceToHost));

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    // --- FP8 naive variants (each allocates/frees its own device buffers) ---
    if (runFP8PerTensor<__nv_fp8_e4m3>(deviceA_fp32, deviceB_fp32, hostC_e4m3_tensor, M, K, N, &times[3]) != 0) return -1;
    if (runFP8PerTensor<__nv_fp8_e5m2>(deviceA_fp32, deviceB_fp32, hostC_e5m2_tensor, M, K, N, &times[4]) != 0) return -1;
    if (runFP8PerRowCol<__nv_fp8_e4m3>(deviceA_fp32, deviceB_fp32, hostC_e4m3_rowcol, M, K, N, &times[5]) != 0) return -1;
    if (runFP8PerRowCol<__nv_fp8_e5m2>(deviceA_fp32, deviceB_fp32, hostC_e5m2_rowcol, M, K, N, &times[6]) != 0) return -1;

    // --- FP8 cuBLASLt tensor core variants ---
    // E4M3 x E4M3: both operands E4M3 (highest precision FP8 combination)
    runCublasLtFP8(lt_handle, deviceA_fp32, deviceB_fp32, hostC_e4m3_cublaslt, M, K, N, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, &times[7]);
    // E4M3 x E5M2: mixed (A=E4M3, B=E5M2) — only mixed combo supported by cuBLASLt
    runCublasLtFP8(lt_handle, deviceA_fp32, deviceB_fp32, hostC_e5m2_cublaslt, M, K, N, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, &times[8]);

    // --- Error metrics ---
    float relL2[9], maxAbs[9];
    relL2[0] = 0.0f; maxAbs[0] = 0.0f;
    computeErrorMetrics(hostC_fp32, hostC_fp16_naive,    sizeC, &relL2[1], &maxAbs[1]);
    computeErrorMetrics(hostC_fp32, hostC_wmma,          sizeC, &relL2[2], &maxAbs[2]);
    computeErrorMetrics(hostC_fp32, hostC_e4m3_tensor,   sizeC, &relL2[3], &maxAbs[3]);
    computeErrorMetrics(hostC_fp32, hostC_e5m2_tensor,   sizeC, &relL2[4], &maxAbs[4]);
    computeErrorMetrics(hostC_fp32, hostC_e4m3_rowcol,   sizeC, &relL2[5], &maxAbs[5]);
    computeErrorMetrics(hostC_fp32, hostC_e5m2_rowcol,   sizeC, &relL2[6], &maxAbs[6]);
    if (times[7] >= 0)
        computeErrorMetrics(hostC_fp32, hostC_e4m3_cublaslt, sizeC, &relL2[7], &maxAbs[7]);
    else { relL2[7] = -1.0f; maxAbs[7] = -1.0f; }
    if (times[8] >= 0)
        computeErrorMetrics(hostC_fp32, hostC_e5m2_cublaslt, sizeC, &relL2[8], &maxAbs[8]);
    else { relL2[8] = -1.0f; maxAbs[8] = -1.0f; }

    // --- Print benchmark table ---
    printf("\n=== GEMM Benchmark: %dx%d x %dx%d ===\n", M, K, K, N);
    printf("%-30s  %10s  %10s  %10s\n", "Kernel", "Time (ms)", "Rel L2", "Max Abs");
    printf("%-30s  %10s  %10s  %10s\n", "------", "---------", "------", "-------");

    const char* labels[9] = {
        "FP32 naive",
        "FP16 naive",
        "FP16 WMMA TC",
        "FP8 E4M3 naive per-tensor",
        "FP8 E5M2 naive per-tensor",
        "FP8 E4M3 naive per-row/col",
        "FP8 E5M2 naive per-row/col",
        "FP8 E4M3xE4M3 cuBLASLt TC",
        "FP8 E4M3xE5M2 cuBLASLt TC"
    };
    for (int i = 0; i < 9; i++) {
        if (times[i] < 0)
            printf("%-30s  %10s  %10s  %10s\n", labels[i], "N/A", "N/A", "N/A");
        else
            printf("%-30s  %10.4f  %10.6f  %10.6f\n",
                   labels[i], times[i], relL2[i], maxAbs[i]);
    }
    printf("\n");

    cublasLtDestroy(lt_handle);

    // Free GPU memory
    gpuTKCheck(cudaFree(deviceA_fp32));
    gpuTKCheck(cudaFree(deviceB_fp32));
    gpuTKCheck(cudaFree(deviceC_fp32));
    gpuTKCheck(cudaFree(deviceA_half));
    gpuTKCheck(cudaFree(deviceB_half));
    gpuTKCheck(cudaFree(deviceC_fp16_naive));
    gpuTKCheck(cudaFree(deviceA_half_pad));
    gpuTKCheck(cudaFree(deviceB_half_pad));
    gpuTKCheck(cudaFree(deviceC_wmma_pad));

    free(hostA);
    free(hostB);
    free(hostC_fp32);
    free(hostC_fp16_naive);
    free(hostC_wmma);
    free(hostC_e4m3_tensor);
    free(hostC_e5m2_tensor);
    free(hostC_e4m3_rowcol);
    free(hostC_e5m2_rowcol);
    free(hostC_e4m3_cublaslt);
    free(hostC_e5m2_cublaslt);

    return 0;
}
