#pragma once

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <fp8_utils.cuh>
#include <stdio.h>

// cuBLASLt FP8 tensor core GEMM for E4M3 or E5M2.

static int runCublasLtFP8(
    cublasLtHandle_t lt_handle,
    const float* dA_fp32, const float* dB_fp32,
    float* hostC,
    int M, int K, int N,
    cudaDataType_t fp8_type,
    double* elapsed_ms)
{
    int sizeA = M * K, sizeB = K * N, sizeC = M * N;
    int threads = 256;

    // --- Quantize A (per-tensor) ---
    float *dMaxA, *dMaxB;
    cudaMalloc(&dMaxA, sizeof(float));
    cudaMalloc(&dMaxB, sizeof(float));
    cudaMemset(dMaxA, 0, sizeof(float));
    cudaMemset(dMaxB, 0, sizeof(float));

    findMaxAbsKernel<<<min(256, (sizeA + threads - 1) / threads), threads, threads * sizeof(float)>>>(dA_fp32, dMaxA, sizeA);
    findMaxAbsKernel<<<min(256, (sizeB + threads - 1) / threads), threads, threads * sizeof(float)>>>(dB_fp32, dMaxB, sizeB);
    cudaDeviceSynchronize();

    float hMaxA, hMaxB;
    cudaMemcpy(&hMaxA, dMaxA, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hMaxB, dMaxB, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dMaxA);
    cudaFree(dMaxB);

    void *dA_fp8 = nullptr, *dB_fp8 = nullptr;
    size_t fp8_size = (fp8_type == CUDA_R_8F_E4M3) ? sizeof(__nv_fp8_e4m3) : sizeof(__nv_fp8_e5m2);
    cudaMalloc(&dA_fp8, sizeA * fp8_size);
    cudaMalloc(&dB_fp8, sizeB * fp8_size);

    if (fp8_type == CUDA_R_8F_E4M3) {
        float scaleA = computeQuantScale<__nv_fp8_e4m3>(hMaxA);
        float scaleB = computeQuantScale<__nv_fp8_e4m3>(hMaxB);
        quantizeFP32toFP8<__nv_fp8_e4m3><<<(sizeA + threads - 1) / threads, threads>>>(
            dA_fp32, (__nv_fp8_e4m3*)dA_fp8, scaleA, sizeA);
        quantizeFP32toFP8<__nv_fp8_e4m3><<<(sizeB + threads - 1) / threads, threads>>>(
            dB_fp32, (__nv_fp8_e4m3*)dB_fp8, scaleB, sizeB);
    } else {
        float scaleA = computeQuantScale<__nv_fp8_e5m2>(hMaxA);
        float scaleB = computeQuantScale<__nv_fp8_e5m2>(hMaxB);
        quantizeFP32toFP8<__nv_fp8_e5m2><<<(sizeA + threads - 1) / threads, threads>>>(
            dA_fp32, (__nv_fp8_e5m2*)dA_fp8, scaleA, sizeA);
        quantizeFP32toFP8<__nv_fp8_e5m2><<<(sizeB + threads - 1) / threads, threads>>>(
            dB_fp32, (__nv_fp8_e5m2*)dB_fp8, scaleB, sizeB);
    }
    cudaDeviceSynchronize();

    cublasLtMatmulDesc_t   matmul_desc = nullptr;
    cublasLtMatrixLayout_t layout_A    = nullptr;
    cublasLtMatrixLayout_t layout_B    = nullptr;
    cublasLtMatrixLayout_t layout_C    = nullptr;

    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t      scale_type   = CUDA_R_32F;

    cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_type);

    cublasOperation_t op_no_t = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_no_t, sizeof(op_no_t));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_no_t, sizeof(op_no_t));

    cublasLtMatrixLayoutCreate(&layout_A, fp8_type, N, K, N); 
    cublasLtMatrixLayoutCreate(&layout_B, fp8_type, K, M, K);  
    cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_32F, N, M, N); 

    float *dC = nullptr;
    cudaMalloc(&dC, sizeC * sizeof(float));
    cudaMemset(dC, 0, sizeC * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmulHeuristicResult_t heuristic = {};
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t workspace_size = 32 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspace_size, sizeof(workspace_size));

    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    int returned_results = 0;
    cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle, matmul_desc,
        layout_A, layout_B, layout_C, layout_C,
        pref, 1, &heuristic, &returned_results);

    if (heur_status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
        fprintf(stderr, "cuBLASLt: no FP8 algorithm found (status=%d, results=%d). "
                        "Requires SM 89+ and CUDA >= 11.8.\n",
                heur_status, returned_results);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layout_A);
        cublasLtMatrixLayoutDestroy(layout_B);
        cublasLtMatrixLayoutDestroy(layout_C);
        cublasLtMatmulDescDestroy(matmul_desc);
        cudaFree(dA_fp8); cudaFree(dB_fp8); cudaFree(dC); cudaFree(workspace);
        *elapsed_ms = -1.0;
        return -1;
    }

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    cublasLtMatmul(
        lt_handle, matmul_desc,
        &alpha,
        dB_fp8, layout_A,   
        dA_fp8, layout_B,   
        &beta,
        dC, layout_C,
        dC, layout_C,
        &heuristic.algo,
        workspace, workspace_size,
        0);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, ev_start, ev_stop);
    *elapsed_ms = (double)ms;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    cudaMemcpy(hostC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layout_A);
    cublasLtMatrixLayoutDestroy(layout_B);
    cublasLtMatrixLayoutDestroy(layout_C);
    cublasLtMatmulDescDestroy(matmul_desc);
    cudaFree(dA_fp8);
    cudaFree(dB_fp8);
    cudaFree(dC);
    cudaFree(workspace);
    return 0;
}
