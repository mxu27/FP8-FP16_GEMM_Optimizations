// links and references:
// https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/
// https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html

//baseline implementation

#include <gputk.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdlib.h>

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

__global__ void convertAndPadFP32ToFP16(
    const float* __restrict__ src, half* __restrict__ dst,
    int srcRows, int srcCols, int dstCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < srcRows && col < srcCols)
        dst[row * dstCols + col] = __float2half(src[row * srcCols + col]);
}

// One warp per block, each computing one 16x16 output tile.
// Utilizing the tensor cores
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

int main(int argc, char **argv) {
    gpuTKArg_t args;

    float *hostA, *hostB, *hostC;
    float *deviceA_fp32, *deviceB_fp32;
    half  *deviceA_half, *deviceB_half;
    float *deviceC_pad;

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

    int M_pad = (numARows    + 15) / 16 * 16;
    int K_pad = (numAColumns + 15) / 16 * 16;
    int N_pad = (numBColumns + 15) / 16 * 16;

    gpuTKTime_start(GPU, "Allocating GPU memory");
    gpuTKCheck(cudaMalloc(&deviceA_fp32, numARows * numAColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceB_fp32, numBRows * numBColumns * sizeof(float)));
    gpuTKCheck(cudaMalloc(&deviceA_half, M_pad * K_pad * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceB_half, K_pad * N_pad * sizeof(half)));
    gpuTKCheck(cudaMalloc(&deviceC_pad,  M_pad * N_pad * sizeof(float)));
    gpuTKCheck(cudaMemset(deviceA_half, 0, M_pad * K_pad * sizeof(half)));
    gpuTKCheck(cudaMemset(deviceB_half, 0, K_pad * N_pad * sizeof(half)));
    gpuTKCheck(cudaMemset(deviceC_pad,  0, M_pad * N_pad * sizeof(float)));
    gpuTKTime_stop(GPU, "Allocating GPU memory");

    gpuTKTime_start(GPU, "Copying input memory to GPU");
    gpuTKCheck(cudaMemcpy(deviceA_fp32, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKCheck(cudaMemcpy(deviceB_fp32, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKTime_stop(GPU, "Copying input memory to GPU");

    {
        dim3 block(16, 16);
        dim3 gridA((numAColumns + 15) / 16, (numARows + 15) / 16);
        dim3 gridB((numBColumns + 15) / 16, (numBRows + 15) / 16);
        gpuTKTime_start(Compute, "FP32 to FP16 conversion");
        convertAndPadFP32ToFP16<<<gridA, block>>>(deviceA_fp32, deviceA_half, numARows, numAColumns, K_pad);
        convertAndPadFP32ToFP16<<<gridB, block>>>(deviceB_fp32, deviceB_half, numBRows, numBColumns, N_pad);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, "FP32 to FP16 conversion");
    }

    {
        dim3 grid(N_pad / WMMA_N, M_pad / WMMA_M);
        dim3 block(32);
        gpuTKTime_start(Compute, "WMMA Tensor Core GEMM");
        matrixMultiplyWMMA<<<grid, block>>>(deviceA_half, deviceB_half, deviceC_pad,
                                            M_pad, K_pad, N_pad);
        cudaDeviceSynchronize();
        gpuTKTime_stop(Compute, "WMMA Tensor Core GEMM");
    }

    gpuTKCheck(cudaMemcpy2D(
        hostC,                           
        numCColumns * sizeof(float),     
        deviceC_pad,                    
        N_pad * sizeof(float),           
        numCColumns * sizeof(float),     
        numCRows,                        
        cudaMemcpyDeviceToHost));

    gpuTKSolution(args, hostC, numCRows, numCColumns);

    gpuTKTime_start(GPU, "Freeing GPU memory");
    gpuTKCheck(cudaFree(deviceA_fp32));
    gpuTKCheck(cudaFree(deviceB_fp32));
    gpuTKCheck(cudaFree(deviceA_half));
    gpuTKCheck(cudaFree(deviceB_half));
    gpuTKCheck(cudaFree(deviceC_pad));
    gpuTKTime_stop(GPU, "Freeing GPU memory");

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
