#include <gputk.h>
#include <cuda_fp16.h>
#include <stdlib.h>
#include <stdio.h>

#define gpuTKCheck(stmt)                                                  \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                     \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));  \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Converts array from FP32 values to FP16 values, writing into a buffer with stride dstCols.
// Out-of-bounds positions in dst are left as zero (caller must cudaMemset).
__global__ void convertAndPadFP32ToFP16(
    const float* src, half* dst,
    int srcRows, int srcCols, int dstCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < srcRows && col < srcCols)
        dst[row * dstCols + col] = __float2half(src[row * srcCols + col]);
}

// Simple FP16 GEMM.
// This kernel takes in FP16 inputs, does FP16 multiply and accumulation, and outputs FP32.
__global__ void matrixMultiplyFP16Naive(
    const half* A,
    const half* B,
    float* C,
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

int main(int argc, char **argv) {
    gpuTKArg_t args;

    float *hostA, *hostB, *hostC;
    float *deviceA_fp32, *deviceB_fp32, *deviceC;
    half  *deviceA_half, *deviceB_half;

    int numARows, numAColumns, numBRows, numBColumns;
    int numCRows, numCColumns;

    args = gpuTKArg_read(argc, argv);

    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows, &numBColumns);
    numCRows    = numARows;
    numCColumns = numBColumns;
    hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");

    gpuTKLog(TRACE, "A: ", numARows, " x ", numAColumns);
    gpuTKLog(TRACE, "B: ", numBRows, " x ", numBColumns);

    gpuTKTime_start(GPU, "Allocating GPU memory");
    gpuTKCheck(cudaMalloc(&deviceA_fp32, numARows * numAColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceB_fp32, numBRows * numBColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceA_half, numARows * numAColumns * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceB_half, numBRows * numBColumns * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceC,      numCRows * numCColumns * sizeof(float)));
    gpuTKTime_stop(GPU, "Allocating GPU memory");

    gpuTKTime_start(GPU, "Copying input memory to GPU");
    gpuTKCheck(cudaMemcpy(deviceA_fp32, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKCheck(cudaMemcpy(deviceB_fp32, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKTime_stop(GPU, "Copying input memory to GPU");

    // call the converstion from fp32 to fp16
    {
        dim3 block(16, 16);
        dim3 gridA((numAColumns + 15) / 16, (numARows + 15) / 16);
        dim3 gridB((numBColumns + 15) / 16, (numBRows + 15) / 16);
        gpuTKTime_start(Compute, "FP32 to FP16 conversion");
        convertAndPadFP32ToFP16<<<gridA, block>>>(deviceA_fp32, deviceA_half, numARows, numAColumns, numAColumns);
        convertAndPadFP32ToFP16<<<gridB, block>>>(deviceB_fp32, deviceB_half, numBRows, numBColumns, numBColumns);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, "FP32 to FP16 conversion");
    }

    // run our fp16 gemm
    {
        dim3 block(16, 16);
        dim3 grid((numCColumns + 15) / 16, (numCRows + 15) / 16);
        gpuTKTime_start(Compute, "Naive FP16 GEMM");
        matrixMultiplyFP16Naive<<<grid, block>>>(deviceA_half, deviceB_half, deviceC,
                                                  numARows, numAColumns, numBColumns);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, "Naive FP16 GEMM");
    }

    gpuTKTime_start(Copy, "Copying output memory to CPU");
    gpuTKCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));
    gpuTKTime_stop(Copy, "Copying output memory to CPU");

    gpuTKSolution(args, hostC, numCRows, numCColumns);

    gpuTKTime_start(GPU, "Freeing GPU memory");
    gpuTKCheck(cudaFree(deviceA_fp32));
    gpuTKCheck(cudaFree(deviceB_fp32));
    gpuTKCheck(cudaFree(deviceA_half));
    gpuTKCheck(cudaFree(deviceB_half));
    gpuTKCheck(cudaFree(deviceC));
    gpuTKTime_stop(GPU, "Freeing GPU memory");

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
