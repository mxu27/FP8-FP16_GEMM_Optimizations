#pragma once

#include <cuda_fp8.h>
#include <math.h>

// Traits to map FP8 type -> max representable value
template <typename T> struct Fp8Traits;
template <> struct Fp8Traits<__nv_fp8_e4m3> { static constexpr float kMax = 448.0f; };
template <> struct Fp8Traits<__nv_fp8_e5m2> { static constexpr float kMax = 57344.0f; };

__device__ inline float atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int  old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// ─── Per-tensor quantization ─────────────────────────────────────────────────

// Grid-stride reduction: find max absolute value across a FP32 array.
__global__ inline void findMaxAbsKernel(const float* __restrict__ input,
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

// Per-tensor quantize: FP32 -> FP8.
template <typename FP8>
__global__ void quantizeFP32toFP8(const float* __restrict__ src,
                                   FP8*         __restrict__ dst,
                                   float scale, int N)
{
    constexpr float kMax = Fp8Traits<FP8>::kMax;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = src[idx] * scale;
        val       = fmaxf(-kMax, fminf(kMax, val));
        dst[idx]  = FP8(val);
    }
}

// Per-tensor dequantize: FP8 -> FP32.
template <typename FP8>
__global__ void dequantizeFP8toFP32(const FP8* __restrict__ src,
                                     float*     __restrict__ dst,
                                     float inv_scale, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = (float)src[idx] * inv_scale;
}

// Compute per-tensor quantization scale
template <typename FP8>
inline float computeQuantScale(float maxAbs) {
    return Fp8Traits<FP8>::kMax / (maxAbs + 1e-12f);
}

// ─── Per-row / per-column quantization ───────────────────────────────────────

// Find max absolute value per row of an (rows x cols) matrix.
// Output: maxPerRow[rows], one value per row.
__global__ inline void findMaxAbsPerRowKernel(const float* __restrict__ input,
                                               float*       __restrict__ maxPerRow,
                                               int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float localMax = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x)
        localMax = fmaxf(localMax, fabsf(input[row * cols + c]));
    sdata[tid] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) maxPerRow[row] = sdata[0];
}

// Find max absolute value per column of an (rows x cols) matrix.
// Output: maxPerCol[cols], one value per column.
__global__ inline void findMaxAbsPerColKernel(const float* __restrict__ input,
                                               float*       __restrict__ maxPerCol,
                                               int rows, int cols)
{
    int col = blockIdx.x;
    if (col >= cols) return;

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float localMax = 0.0f;
    for (int r = tid; r < rows; r += blockDim.x)
        localMax = fmaxf(localMax, fabsf(input[r * cols + col]));
    sdata[tid] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) maxPerCol[col] = sdata[0];
}

// Convert max-abs values to quantization scales in-place: scale[i] = kMax / (maxAbs[i] + eps)
template <typename FP8>
__global__ void maxAbsToScales(float* __restrict__ data, int N)
{
    constexpr float kMax = Fp8Traits<FP8>::kMax;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] = kMax / (data[idx] + 1e-12f);
}

// Invert scales in-place: data[i] = 1 / data[i]
__global__ inline void invertScales(float* __restrict__ data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] = 1.0f / data[idx];
}

// Per-row quantize: each row of an (rows x cols) matrix uses its own scale.
template <typename FP8>
__global__ void quantizeFP32toFP8PerRow(const float* __restrict__ src,
                                         FP8*         __restrict__ dst,
                                         const float* __restrict__ scalePerRow,
                                         int rows, int cols)
{
    constexpr float kMax = Fp8Traits<FP8>::kMax;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float val = src[row * cols + col] * scalePerRow[row];
        val       = fmaxf(-kMax, fminf(kMax, val));
        dst[row * cols + col] = FP8(val);
    }
}

// Per-column quantize: each column of an (rows x cols) matrix uses its own scale.
template <typename FP8>
__global__ void quantizeFP32toFP8PerCol(const float* __restrict__ src,
                                         FP8*         __restrict__ dst,
                                         const float* __restrict__ scalePerCol,
                                         int rows, int cols)
{
    constexpr float kMax = Fp8Traits<FP8>::kMax;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float val = src[row * cols + col] * scalePerCol[col];
        val       = fmaxf(-kMax, fminf(kMax, val));
        dst[row * cols + col] = FP8(val);
    }
}
