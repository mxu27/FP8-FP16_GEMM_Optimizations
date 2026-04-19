#include <gputk.h>
#include <cuda_fp8.h>
#include <fp8_utils.cuh>
#include <stdlib.h>
#include <stdio.h>

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

// ─── GEMM kernels ────────────────────────────────────────────────────────────

// Per-tensor scaled FP8 GEMM.
// C[i][j] = invScaleA * invScaleB * sum_k( (float)A[i][k] * (float)B[k][j] )
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

// Per-row(A) / per-column(B) scaled FP8 GEMM.
// C[i][j] = invScaleA[i] * invScaleB[j] * sum_k( (float)A[i][k] * (float)B[k][j] )
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

// Per-tensor: one scale for all of A, one scale for all of B.
template <typename FP8>
int runPerTensor(const float* deviceA_fp32, const float* deviceB_fp32,
                 float* hostC, int M, int K, int N, const char* label)
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
    findMaxAbsKernel<<<min(256, (sizeA + threads - 1) / threads), threads, threads * sizeof(float)>>>(
        deviceA_fp32, dMaxA, sizeA);
    findMaxAbsKernel<<<min(256, (sizeB + threads - 1) / threads), threads, threads * sizeof(float)>>>(
        deviceB_fp32, dMaxB, sizeB);
    gpuTKCheck(cudaDeviceSynchronize());

    float hMaxA, hMaxB;
    gpuTKCheck(cudaMemcpy(&hMaxA, dMaxA, sizeof(float), cudaMemcpyDeviceToHost));
    gpuTKCheck(cudaMemcpy(&hMaxB, dMaxB, sizeof(float), cudaMemcpyDeviceToHost));

    float scaleA = computeQuantScale<FP8>(hMaxA);
    float scaleB = computeQuantScale<FP8>(hMaxB);

    quantizeFP32toFP8<FP8><<<(sizeA + threads - 1) / threads, threads>>>(deviceA_fp32, dA_fp8, scaleA, sizeA);
    quantizeFP32toFP8<FP8><<<(sizeB + threads - 1) / threads, threads>>>(deviceB_fp32, dB_fp8, scaleB, sizeB);
    gpuTKCheck(cudaDeviceSynchronize());

    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gpuTKTime_start(Compute, label);
        matrixMultiplyFP8PerTensor<FP8><<<grid, block>>>(
            dA_fp8, dB_fp8, dC, 1.0f / scaleA, 1.0f / scaleB, M, K, N);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, label);
    }

    gpuTKCheck(cudaMemcpy(hostC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    gpuTKCheck(cudaFree(dA_fp8));
    gpuTKCheck(cudaFree(dB_fp8));
    gpuTKCheck(cudaFree(dC));
    gpuTKCheck(cudaFree(dMaxA));
    gpuTKCheck(cudaFree(dMaxB));
    return 0;
}

// Per-row(A) / per-column(B): finer-grained scaling.
template <typename FP8>
int runPerRowCol(const float* deviceA_fp32, const float* deviceB_fp32,
                 float* hostC, int M, int K, int N, const char* label)
{
    int sizeA = M * K, sizeB = K * N, sizeC = M * N;
    int threads = BLOCK_SIZE;

    FP8   *dA_fp8, *dB_fp8;
    float *dC;
    float *dScalesA, *dScalesB;    // scales arrays: M rows, N cols

    gpuTKCheck(cudaMalloc(&dA_fp8,    sizeA * sizeof(FP8)));
    gpuTKCheck(cudaMalloc(&dB_fp8,    sizeB * sizeof(FP8)));
    gpuTKCheck(cudaMalloc(&dC,        sizeC * sizeof(float)));
    gpuTKCheck(cudaMalloc(&dScalesA,  M * sizeof(float)));
    gpuTKCheck(cudaMalloc(&dScalesB,  N * sizeof(float)));

    // Find max-abs per row of A, per column of B
    findMaxAbsPerRowKernel<<<M, threads, threads * sizeof(float)>>>(deviceA_fp32, dScalesA, M, K);
    findMaxAbsPerColKernel<<<N, threads, threads * sizeof(float)>>>(deviceB_fp32, dScalesB, K, N);
    gpuTKCheck(cudaDeviceSynchronize());

    // Convert max-abs to quantization scales in-place
    maxAbsToScales<FP8><<<(M + threads - 1) / threads, threads>>>(dScalesA, M);
    maxAbsToScales<FP8><<<(N + threads - 1) / threads, threads>>>(dScalesB, N);
    gpuTKCheck(cudaDeviceSynchronize());

    // Quantize
    {
        dim3 block(16, 16);
        dim3 gridA((K + 15) / 16, (M + 15) / 16);
        dim3 gridB((N + 15) / 16, (K + 15) / 16);
        quantizeFP32toFP8PerRow<FP8><<<gridA, block>>>(deviceA_fp32, dA_fp8, dScalesA, M, K);
        quantizeFP32toFP8PerCol<FP8><<<gridB, block>>>(deviceB_fp32, dB_fp8, dScalesB, K, N);
    }
    gpuTKCheck(cudaDeviceSynchronize());

    // Invert scales for dequantization during GEMM: scale -> 1/scale
    invertScales<<<(M + threads - 1) / threads, threads>>>(dScalesA, M);
    invertScales<<<(N + threads - 1) / threads, threads>>>(dScalesB, N);
    gpuTKCheck(cudaDeviceSynchronize());

    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gpuTKTime_start(Compute, label);
        matrixMultiplyFP8PerRowCol<FP8><<<grid, block>>>(
            dA_fp8, dB_fp8, dC, dScalesA, dScalesB, M, K, N);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, label);
    }

    gpuTKCheck(cudaMemcpy(hostC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    gpuTKCheck(cudaFree(dA_fp8));
    gpuTKCheck(cudaFree(dB_fp8));
    gpuTKCheck(cudaFree(dC));
    gpuTKCheck(cudaFree(dScalesA));
    gpuTKCheck(cudaFree(dScalesB));
    return 0;
}

// ─── Error reporting ─────────────────────────────────────────────────────────

void printError(const float* ref, const float* result, int N, const char* label)
{
    double l2_num = 0.0, l2_den = 0.0;
    float maxAbs = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(result[i] - ref[i]);
        if (diff > maxAbs) maxAbs = diff;
        l2_num += (double)diff * diff;
        l2_den += (double)ref[i] * ref[i];
    }
    float relL2 = (float)sqrt(l2_num / (l2_den + 1e-10));
    printf("  [%-28s]  Relative L2: %.6f  Max abs: %.6f\n", label, relL2, maxAbs);
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

    int sizeC = numCRows * numCColumns;

    // Allocate one output buffer per variant
    float *hostC_e4m3_tensor  = (float *)malloc(sizeC * sizeof(float));
    float *hostC_e5m2_tensor  = (float *)malloc(sizeC * sizeof(float));
    float *hostC_e4m3_rowcol  = (float *)malloc(sizeC * sizeof(float));
    float *hostC_e5m2_rowcol  = (float *)malloc(sizeC * sizeof(float));

    gpuTKTime_start(GPU, "Copying input memory to GPU");
    gpuTKCheck(cudaMalloc(&deviceA_fp32, numARows * numAColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceB_fp32, numBRows * numBColumns * sizeof(float)));
    gpuTKCheck(cudaMemcpy(deviceA_fp32, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKCheck(cudaMemcpy(deviceB_fp32, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKTime_stop(GPU, "Copying input memory to GPU");

    int M = numARows, K = numAColumns, N = numBColumns;

    // Run all four variants
    if (runPerTensor<__nv_fp8_e4m3>(deviceA_fp32, deviceB_fp32, hostC_e4m3_tensor, M, K, N,
                                     "E4M3 per-tensor GEMM") != 0) return -1;
    if (runPerTensor<__nv_fp8_e5m2>(deviceA_fp32, deviceB_fp32, hostC_e5m2_tensor, M, K, N,
                                     "E5M2 per-tensor GEMM") != 0) return -1;
    if (runPerRowCol<__nv_fp8_e4m3>(deviceA_fp32, deviceB_fp32, hostC_e4m3_rowcol, M, K, N,
                                     "E4M3 per-row/col GEMM") != 0) return -1;
    if (runPerRowCol<__nv_fp8_e5m2>(deviceA_fp32, deviceB_fp32, hostC_e5m2_rowcol, M, K, N,
                                     "E5M2 per-row/col GEMM") != 0) return -1;

    // Use gpuTKSolution to validate E4M3 per-row/col (finest granularity, best accuracy)
    gpuTKSolution(args, hostC_e4m3_rowcol, numCRows, numCColumns);

    // Compute FP32 reference for error comparison
    // (reuse E4M3 per-row/col as "best" — but true reference needs FP32 GEMM)
    // We compare all variants against the expected output from the data files
    printf("\n=== FP8 GEMM Accuracy vs FP32 Reference ===\n");

    // Load FP32 reference if available
    float *hostRef = NULL;
    int refRows, refCols;
    const char* expectedFile = gpuTKArg_getExpectedOutputFile(args);
    if (expectedFile != NULL) {
        hostRef = (float *)gpuTKImport(expectedFile, &refRows, &refCols);
        if (hostRef != NULL) {
            printError(hostRef, hostC_e4m3_tensor, sizeC, "E4M3 per-tensor");
            printError(hostRef, hostC_e5m2_tensor, sizeC, "E5M2 per-tensor");
            printError(hostRef, hostC_e4m3_rowcol, sizeC, "E4M3 per-row/col");
            printError(hostRef, hostC_e5m2_rowcol, sizeC, "E5M2 per-row/col");
            free(hostRef);
        }
    }
    printf("============================================\n\n");

    gpuTKTime_start(GPU, "Freeing GPU memory");
    gpuTKCheck(cudaFree(deviceA_fp32));
    gpuTKCheck(cudaFree(deviceB_fp32));
    gpuTKTime_stop(GPU, "Freeing GPU memory");

    free(hostA);
    free(hostB);
    free(hostC_e4m3_tensor);
    free(hostC_e5m2_tensor);
    free(hostC_e4m3_rowcol);
    free(hostC_e5m2_rowcol);

    return 0;
}
