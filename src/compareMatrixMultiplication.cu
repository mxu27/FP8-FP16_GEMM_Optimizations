#include <gputk.h>
#include <cuda_fp16.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Runs both FP32 and naive FP16 GEMM kernels on the same inputs and prints
// error metrics (relative L2 error, max absolute error) comparing the two.

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

// FP16 kernel copied over.
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
    float *hostC_fp32, *hostC_fp16;

    float *deviceA_fp32, *deviceB_fp32;
    float *deviceC_fp32, *deviceC_fp16;
    half  *deviceA_half, *deviceB_half;

    int numARows, numAColumns, numBRows, numBColumns;
    int numCRows, numCColumns;

    args = gpuTKArg_read(argc, argv);

    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows, &numBColumns);
    numCRows    = numARows;
    numCColumns = numBColumns;
    hostC_fp32 = (float *)malloc(numCRows * numCColumns * sizeof(float));
    hostC_fp16 = (float *)malloc(numCRows * numCColumns * sizeof(float));
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");

    gpuTKLog(TRACE, "A: ", numARows, " x ", numAColumns);
    gpuTKLog(TRACE, "B: ", numBRows, " x ", numBColumns);

    gpuTKTime_start(GPU, "Allocating GPU memory");
    gpuTKCheck(cudaMalloc(&deviceA_fp32, numARows * numAColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceB_fp32, numBRows * numBColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceC_fp32, numCRows * numCColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceA_half, numARows * numAColumns * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceB_half, numBRows * numBColumns * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceC_fp16, numCRows * numCColumns * sizeof(float)));
    gpuTKTime_stop(GPU, "Allocating GPU memory");

    gpuTKTime_start(GPU, "Copying input memory to GPU");
    gpuTKCheck(cudaMemcpy(deviceA_fp32, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKCheck(cudaMemcpy(deviceB_fp32, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKTime_stop(GPU, "Copying input memory to GPU");

    // Floating point 32 gemm
    dim3 blockFP32(16, 16);
    dim3 gridFP32((numCColumns + 15) / 16, (numCRows + 15) / 16);
    gpuTKTime_start(Compute, "FP32 GEMM");
    matrixMultiplyFP32<<<gridFP32, blockFP32>>>(deviceA_fp32, deviceB_fp32, deviceC_fp32,
                                        numARows, numAColumns, numBColumns);
    cudaDeviceSynchronize();
    gpuTKTime_stop(Compute, "FP32 GEMM");
    gpuTKCheck(cudaMemcpy(hostC_fp32, deviceC_fp32, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));

    // Input conversion to fp16
    dim3 blockConv(16, 16);
    dim3 gridConvA((numAColumns + 15) / 16, (numARows + 15) / 16);
    dim3 gridConvB((numBColumns + 15) / 16, (numBRows + 15) / 16);
    gpuTKTime_start(Compute, "FP32 to FP16 conversion");
    convertFP32ToFP16<<<gridConvA, blockConv>>>(deviceA_fp32, deviceA_half, numARows, numAColumns);
    convertFP32ToFP16<<<gridConvB, blockConv>>>(deviceB_fp32, deviceB_half, numBRows, numBColumns);
    cudaDeviceSynchronize();
    gpuTKTime_stop(Compute, "FP32 to FP16 conversion");

    // FP16 GEMM
    dim3 blockFP16(16, 16);
    dim3 gridFP16((numCColumns + 15) / 16, (numCRows + 15) / 16);
    gpuTKTime_start(Compute, "Naive FP16 GEMM");
    matrixMultiplyFP16Naive<<<gridFP16, blockFP16>>>(deviceA_half, deviceB_half, deviceC_fp16,
                                              numARows, numAColumns, numBColumns);
    cudaDeviceSynchronize();
    gpuTKTime_stop(Compute, "Naive FP16 GEMM");
    gpuTKCheck(cudaMemcpy(hostC_fp16, deviceC_fp16, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));

    // Measuring the error between the two
    float relL2, maxAbs;
    computeErrorMetrics(hostC_fp32, hostC_fp16, numCRows * numCColumns, &relL2, &maxAbs);
    printf("\n=== FP16 Naive vs FP32 ===\n");
    printf("Relative L2 error: %.6f\n", relL2);
    printf("Max absolute error: %.6f\n", maxAbs);
    printf("==========================\n\n");

    gpuTKTime_start(GPU, "Freeing GPU memory");
    gpuTKCheck(cudaFree(deviceA_fp32));
    gpuTKCheck(cudaFree(deviceB_fp32));
    gpuTKCheck(cudaFree(deviceC_fp32));
    gpuTKCheck(cudaFree(deviceA_half));
    gpuTKCheck(cudaFree(deviceB_half));
    gpuTKCheck(cudaFree(deviceC_fp16));
    gpuTKTime_stop(GPU, "Freeing GPU memory");

    free(hostA);
    free(hostB);
    free(hostC_fp32);
    free(hostC_fp16);

    return 0;
}
