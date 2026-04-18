#include <gputk.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define gpuTKCheck(stmt)                                                  \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                     \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));  \
      return -1;                                                          \
    }                                                                     \
  } while (0)

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

// One warp (32 threads) per block, each computing one 16x16 output tile.
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

int main(int argc, char **argv) {
    gpuTKArg_t args;

    float *hostA, *hostB;
    float *hostC_fp32, *hostC_fp16_naive, *hostC_wmma;

    float *deviceA_fp32, *deviceB_fp32;
    float *deviceC_fp32, *deviceC_fp16_naive;
    half  *deviceA_half, *deviceB_half;
    half  *deviceA_half_pad, *deviceB_half_pad;
    float *deviceC_wmma_pad;

    int numARows, numAColumns, numBRows, numBColumns;
    int numCRows, numCColumns;

    args = gpuTKArg_read(argc, argv);

    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows, &numBColumns);
    numCRows    = numARows;
    numCColumns = numBColumns;
    hostC_fp32       = (float *)malloc(numCRows * numCColumns * sizeof(float));
    hostC_fp16_naive = (float *)malloc(numCRows * numCColumns * sizeof(float));
    hostC_wmma       = (float *)malloc(numCRows * numCColumns * sizeof(float));
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");

    gpuTKLog(TRACE, "A: ", numARows, " x ", numAColumns);
    gpuTKLog(TRACE, "B: ", numBRows, " x ", numBColumns);

    int M_pad = (numARows    + 15) / 16 * 16;
    int K_pad = (numAColumns + 15) / 16 * 16;
    int N_pad = (numBColumns + 15) / 16 * 16;

    gpuTKTime_start(GPU, "Allocating GPU memory");
    gpuTKCheck(cudaMalloc(&deviceA_fp32,      numARows * numAColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceB_fp32,      numBRows * numBColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceC_fp32,      numCRows * numCColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceA_half,      numARows * numAColumns * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceB_half,      numBRows * numBColumns * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceC_fp16_naive,numCRows * numCColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceA_half_pad,  M_pad * K_pad * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceB_half_pad,  K_pad * N_pad * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceC_wmma_pad,  M_pad * N_pad * sizeof(float)));
    gpuTKCheck(cudaMemset(deviceA_half_pad, 0, M_pad * K_pad * sizeof(half)));
    gpuTKCheck(cudaMemset(deviceB_half_pad, 0, K_pad * N_pad * sizeof(half)));
    gpuTKCheck(cudaMemset(deviceC_wmma_pad, 0, M_pad * N_pad * sizeof(float)));
    gpuTKTime_stop(GPU, "Allocating GPU memory");

    gpuTKTime_start(GPU, "Copying input memory to GPU");
    gpuTKCheck(cudaMemcpy(deviceA_fp32, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKCheck(cudaMemcpy(deviceB_fp32, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKTime_stop(GPU, "Copying input memory to GPU");

    // --- FP32 GEMM ---
    {
        dim3 block(16, 16);
        dim3 grid((numCColumns + 15) / 16, (numCRows + 15) / 16);
        gpuTKTime_start(Compute, "FP32 GEMM");
        matrixMultiplyFP32<<<grid, block>>>(deviceA_fp32, deviceB_fp32, deviceC_fp32,
                                            numARows, numAColumns, numBColumns);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, "FP32 GEMM");
    }
    gpuTKCheck(cudaMemcpy(hostC_fp32, deviceC_fp32, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Convert for naive FP16 (no padding needed) ---
    {
        dim3 block(16, 16);
        dim3 gridA((numAColumns + 15) / 16, (numARows + 15) / 16);
        dim3 gridB((numBColumns + 15) / 16, (numBRows + 15) / 16);
        gpuTKTime_start(Compute, "FP32 to FP16 conversion (naive)");
        convertFP32ToFP16<<<gridA, block>>>(deviceA_fp32, deviceA_half, numARows, numAColumns);
        convertFP32ToFP16<<<gridB, block>>>(deviceB_fp32, deviceB_half, numBRows, numBColumns);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, "FP32 to FP16 conversion (naive)");
    }

    // --- Naive FP16 GEMM ---
    {
        dim3 block(16, 16);
        dim3 grid((numCColumns + 15) / 16, (numCRows + 15) / 16);
        gpuTKTime_start(Compute, "Naive FP16 GEMM");
        matrixMultiplyFP16Naive<<<grid, block>>>(deviceA_half, deviceB_half, deviceC_fp16_naive,
                                                 numARows, numAColumns, numBColumns);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, "Naive FP16 GEMM");
    }
    gpuTKCheck(cudaMemcpy(hostC_fp16_naive, deviceC_fp16_naive, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Convert for WMMA (padded) ---
    {
        dim3 block(16, 16);
        dim3 gridA((numAColumns + 15) / 16, (numARows + 15) / 16);
        dim3 gridB((numBColumns + 15) / 16, (numBRows + 15) / 16);
        gpuTKTime_start(Compute, "FP32 to FP16 conversion (WMMA padded)");
        convertAndPadFP32ToFP16<<<gridA, block>>>(deviceA_fp32, deviceA_half_pad, numARows, numAColumns, K_pad);
        convertAndPadFP32ToFP16<<<gridB, block>>>(deviceB_fp32, deviceB_half_pad, numBRows, numBColumns, N_pad);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, "FP32 to FP16 conversion (WMMA padded)");
    }

    // --- WMMA Tensor Core GEMM ---
    {
        dim3 grid(N_pad / WMMA_N, M_pad / WMMA_M);
        dim3 block(32);
        gpuTKTime_start(Compute, "WMMA Tensor Core GEMM");
        matrixMultiplyWMMA<<<grid, block>>>(deviceA_half_pad, deviceB_half_pad, deviceC_wmma_pad,
                                            M_pad, K_pad, N_pad);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, "WMMA Tensor Core GEMM");
    }
    gpuTKCheck(cudaMemcpy2D(
        hostC_wmma,                      // dst
        numCColumns * sizeof(float),     // dst pitch
        deviceC_wmma_pad,                // src
        N_pad * sizeof(float),           // src pitch
        numCColumns * sizeof(float),     // width in bytes
        numCRows,                        // height
        cudaMemcpyDeviceToHost));

    // --- Error metrics ---
    float relL2, maxAbs;
    printf("\n=== Accuracy vs FP32 reference ===\n");

    computeErrorMetrics(hostC_fp32, hostC_fp16_naive, numCRows * numCColumns, &relL2, &maxAbs);
    printf("[Naive FP16]  Relative L2 error: %.6f  Max abs error: %.6f\n", relL2, maxAbs);

    computeErrorMetrics(hostC_fp32, hostC_wmma, numCRows * numCColumns, &relL2, &maxAbs);
    printf("[WMMA TC]     Relative L2 error: %.6f  Max abs error: %.6f\n", relL2, maxAbs);

    printf("===================================\n\n");

    gpuTKTime_start(GPU, "Freeing GPU memory");
    gpuTKCheck(cudaFree(deviceA_fp32));
    gpuTKCheck(cudaFree(deviceB_fp32));
    gpuTKCheck(cudaFree(deviceC_fp32));
    gpuTKCheck(cudaFree(deviceA_half));
    gpuTKCheck(cudaFree(deviceB_half));
    gpuTKCheck(cudaFree(deviceC_fp16_naive));
    gpuTKCheck(cudaFree(deviceA_half_pad));
    gpuTKCheck(cudaFree(deviceB_half_pad));
    gpuTKCheck(cudaFree(deviceC_wmma_pad));
    gpuTKTime_stop(GPU, "Freeing GPU memory");

    free(hostA);
    free(hostB);
    free(hostC_fp32);
    free(hostC_fp16_naive);
    free(hostC_wmma);

    return 0;
}
