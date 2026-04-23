#pragma once

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <fp8_utils.cuh>
#include <stdio.h>

// cuBLASLt FP8 tensor core GEMM for E4M3 or E5M2.
//
// Uses per-tensor scaling with scale pointers passed to the matmul descriptor,
// which is required by cuBLASLt FP8 on SM 89+.

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

    // --- Find max abs and compute scales on host ---
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

    float scaleA, scaleB, invScaleA, invScaleB;
    if (fp8_type == CUDA_R_8F_E4M3) {
        scaleA = computeQuantScale<__nv_fp8_e4m3>(hMaxA);
        scaleB = computeQuantScale<__nv_fp8_e4m3>(hMaxB);
    } else {
        scaleA = computeQuantScale<__nv_fp8_e5m2>(hMaxA);
        scaleB = computeQuantScale<__nv_fp8_e5m2>(hMaxB);
    }
    invScaleA = 1.0f / scaleA;
    invScaleB = 1.0f / scaleB;

    // cuBLASLt needs scale values as device pointers
    float *dScaleA, *dScaleB, *dScaleC, *dInvScaleD;
    cudaMalloc(&dScaleA,    sizeof(float));
    cudaMalloc(&dScaleB,    sizeof(float));
    cudaMalloc(&dScaleC,    sizeof(float));   
    cudaMalloc(&dInvScaleD, sizeof(float));   

    cudaMemcpy(dScaleA,    &invScaleA, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dScaleB,    &invScaleB, sizeof(float), cudaMemcpyHostToDevice);
    float one = 1.0f;
    cudaMemcpy(dScaleC,    &one,       sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dInvScaleD, &one,       sizeof(float), cudaMemcpyHostToDevice);

    // --- Quantize A and B ---
    void *dA_fp8 = nullptr, *dB_fp8 = nullptr;
    size_t fp8_size = (fp8_type == CUDA_R_8F_E4M3) ? sizeof(__nv_fp8_e4m3) : sizeof(__nv_fp8_e5m2);
    cudaMalloc(&dA_fp8, sizeA * fp8_size);
    cudaMalloc(&dB_fp8, sizeB * fp8_size);

    if (fp8_type == CUDA_R_8F_E4M3) {
        quantizeFP32toFP8<__nv_fp8_e4m3><<<(sizeA + threads - 1) / threads, threads>>>(
            dA_fp32, (__nv_fp8_e4m3*)dA_fp8, scaleA, sizeA);
        quantizeFP32toFP8<__nv_fp8_e4m3><<<(sizeB + threads - 1) / threads, threads>>>(
            dB_fp32, (__nv_fp8_e4m3*)dB_fp8, scaleB, sizeB);
    } else {
        quantizeFP32toFP8<__nv_fp8_e5m2><<<(sizeA + threads - 1) / threads, threads>>>(
            dA_fp32, (__nv_fp8_e5m2*)dA_fp8, scaleA, sizeA);
        quantizeFP32toFP8<__nv_fp8_e5m2><<<(sizeB + threads - 1) / threads, threads>>>(
            dB_fp32, (__nv_fp8_e5m2*)dB_fp8, scaleB, sizeB);
    }
    cudaDeviceSynchronize();

    // --- cuBLASLt descriptor ---
    cublasLtMatmulDesc_t   matmul_desc = nullptr;
    cublasLtMatrixLayout_t layout_A    = nullptr;  
    cublasLtMatrixLayout_t layout_B    = nullptr; 
    cublasLtMatrixLayout_t layout_C    = nullptr;  

    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA, sizeof(dScaleA));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dScaleB, sizeof(dScaleB));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &dInvScaleD, sizeof(dInvScaleD));

    cublasLtMatrixLayoutCreate(&layout_A, fp8_type,    N, K, N);
    cublasLtMatrixLayoutCreate(&layout_B, fp8_type,    K, M, K);
    cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_32F,  N, M, N);

    float *dC = nullptr;
    cudaMalloc(&dC, sizeC * sizeof(float));
    cudaMemset(dC, 0, sizeC * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    // --- Algorithm heuristic ---
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t workspace_size = 32 * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspace_size, sizeof(workspace_size));

    void* workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    int returned_results = 0;
    cublasLtMatmulHeuristicResult_t heuristic = {};
    cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(
        lt_handle, matmul_desc,
        layout_A, layout_B, layout_C, layout_C,
        pref, 1, &heuristic, &returned_results);

    if (heur_status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
        fprintf(stderr, "cuBLASLt FP8 (%s): no algorithm found (status=%d, results=%d).\n",
                fp8_type == CUDA_R_8F_E4M3 ? "E4M3" : "E5M2",
                (int)heur_status, returned_results);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layout_A);
        cublasLtMatrixLayoutDestroy(layout_B);
        cublasLtMatrixLayoutDestroy(layout_C);
        cublasLtMatmulDescDestroy(matmul_desc);
        cudaFree(dA_fp8); cudaFree(dB_fp8); cudaFree(dC); cudaFree(workspace);
        cudaFree(dScaleA); cudaFree(dScaleB); cudaFree(dScaleC); cudaFree(dInvScaleD);
        *elapsed_ms = -1.0;
        return -1;
    }

    // --- Run and time just the GEMM kernel ---
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
    cudaFree(dScaleA);
    cudaFree(dScaleB);
    cudaFree(dScaleC);
    cudaFree(dInvScaleD);
    return 0;
}
