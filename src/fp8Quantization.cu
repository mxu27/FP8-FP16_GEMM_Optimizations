#include <gputk.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// This implementation only includes FP8 E4M3 quantization and dequantization.

#define FP8_E4M3_MAX 448.0f // max representable value
#define BLOCK_SIZE   256

#define CUDA_CHECK(stmt)                                                   \
  do {                                                                     \
    cudaError_t err = (stmt);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "[CUDA] %s:%d — %s\n",                             \
              __FILE__, __LINE__, cudaGetErrorString(err));                \
      return -1;                                                           \
    }                                                                      \
  } while (0)



__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int  old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Grid-stride reduction: find max absolute value across a FP32 matrix.
__global__ void findMaxAbsKernel(const float* __restrict__ input,
                                  float*       __restrict__ globalMax, int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float localMax = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
        localMax = fmaxf(localMax, fabsf(input[i]));
    sdata[tid] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) atomicMaxFloat(globalMax, sdata[0]);
}

// Per-tensor quantize: FP32 -> FP8 E4M3.
// scale used = FP8_E4M3_MAX / maxAbs — compresses dynamic range into [-448, 448].
__global__ void quantizeFP32toFP8(const float*   __restrict__ src,
                                   __nv_fp8_e4m3* __restrict__ dst,
                                   float scale, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = src[idx] * scale;
        val       = fmaxf(-FP8_E4M3_MAX, fminf(FP8_E4M3_MAX, val));
        dst[idx]  = __nv_fp8_e4m3(val);
    }
}

// Per-tensor dequantize: FP8 E4M3 -> FP32.
// inv_scale = 1/quantScale = maxAbs / FP8_E4M3_MAX — maps FP8 range back to original.
__global__ void dequantizeFP8toFP32(__nv_fp8_e4m3* __restrict__ src,
                                     float*          __restrict__ dst,
                                     float inv_scale, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = (float)src[idx] * inv_scale;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    gpuTKArg_t args = gpuTKArg_read(argc, argv);

    int numARows, numAColumns, numBRows, numBColumns;
    gpuTKTime_start(Generic, "Importing data and creating memory on host");
    float* hostA = (float*)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows, &numAColumns);
    float* hostB = (float*)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows, &numBColumns);
    int M = numARows, K = numAColumns, N = numBColumns;

    float* hostA_roundtrip = (float*)malloc(M * K * sizeof(float));
    float* hostB_roundtrip = (float*)malloc(K * N * sizeof(float));
    gpuTKTime_stop(Generic, "Importing data and creating memory on host");

    gpuTKLog(TRACE, "A: ", M, " x ", K);
    gpuTKLog(TRACE, "B: ", K, " x ", N);

    float         *d_A_fp32, *d_B_fp32;
    float         *d_A_dq,   *d_B_dq;
    __nv_fp8_e4m3 *d_A_fp8,  *d_B_fp8;
    float         *d_maxA,   *d_maxB;

    CUDA_CHECK(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_fp8,  M * K * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_B_fp8,  K * N * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_A_dq,   M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_dq,   K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxA,   sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_maxB,   sizeof(float)));

    gpuTKTime_start(GPU, "Copying input memory to GPU");
    CUDA_CHECK(cudaMemcpy(d_A_fp32, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp32, hostB, K * N * sizeof(float), cudaMemcpyHostToDevice));
    gpuTKTime_stop(GPU, "Copying input memory to GPU");

    int threads = BLOCK_SIZE;

    // --- Quantize A and B: FP32 -> FP8 E4M3 ---
    gpuTKTime_start(Compute, "FP8 quantization");

    CUDA_CHECK(cudaMemset(d_maxA, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_maxB, 0, sizeof(float)));
    findMaxAbsKernel<<<min(256,(M*K+threads-1)/threads), threads, threads*sizeof(float)>>>(d_A_fp32, d_maxA, M*K);
    findMaxAbsKernel<<<min(256,(K*N+threads-1)/threads), threads, threads*sizeof(float)>>>(d_B_fp32, d_maxB, K*N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_maxA, h_maxB;
    CUDA_CHECK(cudaMemcpy(&h_maxA, d_maxA, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_maxB, d_maxB, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  maxAbsA = %.4f  maxAbsB = %.4f\n", h_maxA, h_maxB);

    float h_quantScaleA = FP8_E4M3_MAX / (h_maxA + 1e-12f);
    float h_quantScaleB = FP8_E4M3_MAX / (h_maxB + 1e-12f);
    printf("  quantScaleA = %.6f  quantScaleB = %.6f\n", h_quantScaleA, h_quantScaleB);

    quantizeFP32toFP8<<<(M*K+threads-1)/threads, threads>>>(d_A_fp32, d_A_fp8, h_quantScaleA, M*K);
    quantizeFP32toFP8<<<(K*N+threads-1)/threads, threads>>>(d_B_fp32, d_B_fp8, h_quantScaleB, K*N);
    CUDA_CHECK(cudaDeviceSynchronize());
    gpuTKTime_stop(Compute, "FP8 quantization");

    // --- Dequantize A and B: FP8 E4M3 -> FP32 ---
    gpuTKTime_start(Compute, "FP8 dequantization");
    float h_invScaleA = 1.0f / h_quantScaleA;
    float h_invScaleB = 1.0f / h_quantScaleB;
    dequantizeFP8toFP32<<<(M*K+threads-1)/threads, threads>>>(d_A_fp8, d_A_dq, h_invScaleA, M*K);
    dequantizeFP8toFP32<<<(K*N+threads-1)/threads, threads>>>(d_B_fp8, d_B_dq, h_invScaleB, K*N);
    CUDA_CHECK(cudaDeviceSynchronize());
    gpuTKTime_stop(Compute, "FP8 dequantization");

    CUDA_CHECK(cudaMemcpy(hostA_roundtrip, d_A_dq, M * K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostB_roundtrip, d_B_dq, K * N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Calculate the Round-trip error vs original FP32 ---
    double l2_num = 0.0, l2_den = 0.0;
    float  maxAbsErrA = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float diff = fabsf(hostA_roundtrip[i] - hostA[i]);
        maxAbsErrA = fmaxf(maxAbsErrA, diff);
        l2_num += (double)diff * diff;
        l2_den += (double)hostA[i] * hostA[i];
    }
    float relL2A = (float)sqrt(l2_num / (l2_den + 1e-10));

    double l2_numB = 0.0, l2_denB = 0.0;
    float  maxAbsErrB = 0.0f;
    for (int i = 0; i < K * N; i++) {
        float diff = fabsf(hostB_roundtrip[i] - hostB[i]);
        maxAbsErrB = fmaxf(maxAbsErrB, diff);
        l2_numB += (double)diff * diff;
        l2_denB += (double)hostB[i] * hostB[i];
    }
    float relL2B = (float)sqrt(l2_numB / (l2_denB + 1e-10));

    printf("\n=== FP8 Q/DQ Round-Trip Error ===\n");
    printf("[Matrix A]  Relative L2: %.6f  Max abs: %.6f\n", relL2A, maxAbsErrA);
    printf("[Matrix B]  Relative L2: %.6f  Max abs: %.6f\n", relL2B, maxAbsErrB);
    printf("=================================\n");

    gpuTKTime_start(GPU, "Freeing GPU memory");
    CUDA_CHECK(cudaFree(d_A_fp32));
    CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_A_fp8));
    CUDA_CHECK(cudaFree(d_B_fp8));
    CUDA_CHECK(cudaFree(d_A_dq));
    CUDA_CHECK(cudaFree(d_B_dq));
    CUDA_CHECK(cudaFree(d_maxA));
    CUDA_CHECK(cudaFree(d_maxB));
    gpuTKTime_stop(GPU, "Freeing GPU memory");

    free(hostA);
    free(hostB);
    free(hostA_roundtrip);
    free(hostB_roundtrip);

    return 0;
}
