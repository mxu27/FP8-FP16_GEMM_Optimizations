#pragma once

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <fp8_utils.cuh>
#include <stdio.h>

__global__ static void applyDeqScale(float* C, float invScaleA, float invScaleB, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] *= invScaleA * invScaleB;
}

static void diagnoseFP8Support(cublasLtHandle_t lt_handle, int M, int K, int N,
                                cudaDataType_t typeA, cudaDataType_t typeB)
{
    fprintf(stderr, "\n--- cuBLASLt FP8 diagnostic (A=%s B=%s) ---\n",
            typeA == CUDA_R_8F_E4M3 ? "E4M3" : "E5M2",
            typeB == CUDA_R_8F_E4M3 ? "E4M3" : "E5M2");

    const char* op_names[] = {"N", "T"};
    cublasOperation_t ops[] = {CUBLAS_OP_N, CUBLAS_OP_T};

    for (int ia = 0; ia < 2; ia++) {
        for (int ib = 0; ib < 2; ib++) {
            cublasLtMatmulDesc_t   desc = nullptr;
            cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;

            cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ops[ia], sizeof(ops[ia]));
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &ops[ib], sizeof(ops[ib]));

            int lda = (ops[ia] == CUBLAS_OP_N) ? K : M;
            int ldb = (ops[ib] == CUBLAS_OP_N) ? N : K;

            cublasLtMatrixLayoutCreate(&la, typeA, (ops[ia] == CUBLAS_OP_N) ? M : K,
                                                   (ops[ia] == CUBLAS_OP_N) ? K : M, lda);
            cublasLtMatrixLayoutCreate(&lb, typeB, (ops[ib] == CUBLAS_OP_N) ? K : N,
                                                   (ops[ib] == CUBLAS_OP_N) ? N : K, ldb);
            cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, M, N, M);

            cublasLtMatmulPreference_t pref = nullptr;
            cublasLtMatmulPreferenceCreate(&pref);
            size_t ws = 32 * 1024 * 1024;
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

            cublasLtMatmulHeuristicResult_t results[4] = {};
            int nresults = 0;
            cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
                lt_handle, desc, la, lb, lc, lc, pref, 4, results, &nresults);

            fprintf(stderr, "  transa=%s transb=%s lda=%d ldb=%d => status=%d results=%d\n",
                    op_names[ia], op_names[ib], lda, ldb, (int)st, nresults);

            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(la);
            cublasLtMatrixLayoutDestroy(lb);
            cublasLtMatrixLayoutDestroy(lc);
            cublasLtMatmulDescDestroy(desc);
        }
    }
    fprintf(stderr, "--- end diagnostic ---\n\n");
}

static int runCublasLtFP8(
    cublasLtHandle_t lt_handle,
    const float* dA_fp32, const float* dB_fp32,
    float* hostC,
    int M, int K, int N,
    cudaDataType_t typeA,
    cudaDataType_t typeB,
    double* elapsed_ms)
{
    int sizeA = M * K, sizeB = K * N, sizeC = M * N;
    int threads = 256;

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

    float scaleA = (typeA == CUDA_R_8F_E4M3) ? computeQuantScale<__nv_fp8_e4m3>(hMaxA)
                                              : computeQuantScale<__nv_fp8_e5m2>(hMaxA);
    float scaleB = (typeB == CUDA_R_8F_E4M3) ? computeQuantScale<__nv_fp8_e4m3>(hMaxB)
                                              : computeQuantScale<__nv_fp8_e5m2>(hMaxB);

    void *dA_fp8 = nullptr, *dB_fp8 = nullptr;
    cudaMalloc(&dA_fp8, sizeA * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&dB_fp8, sizeB * sizeof(__nv_fp8_e4m3));

    if (typeA == CUDA_R_8F_E4M3)
        quantizeFP32toFP8<__nv_fp8_e4m3><<<(sizeA + threads - 1) / threads, threads>>>(
            dA_fp32, (__nv_fp8_e4m3*)dA_fp8, scaleA, sizeA);
    else
        quantizeFP32toFP8<__nv_fp8_e5m2><<<(sizeA + threads - 1) / threads, threads>>>(
            dA_fp32, (__nv_fp8_e5m2*)dA_fp8, scaleA, sizeA);

    if (typeB == CUDA_R_8F_E4M3)
        quantizeFP32toFP8<__nv_fp8_e4m3><<<(sizeB + threads - 1) / threads, threads>>>(
            dB_fp32, (__nv_fp8_e4m3*)dB_fp8, scaleB, sizeB);
    else
        quantizeFP32toFP8<__nv_fp8_e5m2><<<(sizeB + threads - 1) / threads, threads>>>(
            dB_fp32, (__nv_fp8_e5m2*)dB_fp8, scaleB, sizeB);

    cudaDeviceSynchronize();

    cublasLtMatmulDesc_t   matmul_desc = nullptr;
    cublasLtMatrixLayout_t layout_A    = nullptr;
    cublasLtMatrixLayout_t layout_B    = nullptr;
    cublasLtMatrixLayout_t layout_C    = nullptr;

    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Only supported combo on SM 89: transa=T, transb=N
    // A stored as col-major(K x M) with ld=K, transposed to (M x K) by op=T
    //   => row-major A(M x K) with ld=K, same memory layout
    // B stored as col-major(K x N) with ld=K, used as-is by op=N
    //   => row-major B^T... no: col-major(K x N) ld=K with op=N gives K x N matrix
    // Result C col-major(M x N) ld=M = row-major C(M x N) ld=M (contiguous)
    cublasOperation_t op_t = CUBLAS_OP_T;
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // layout_A: col-major(K x M) ld=K  ->  after op=T gives (M x K)
    // layout_B: col-major(K x N) ld=K  ->  after op=N gives (K x N)
    // layout_C: col-major(M x N) ld=M
    cublasLtMatrixLayoutCreate(&layout_A, typeA,      K, M, K);
    cublasLtMatrixLayoutCreate(&layout_B, typeB,      K, N, K);
    cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_32F, M, N, M);

    float *dC = nullptr;
    cudaMalloc(&dC, sizeC * sizeof(float));
    cudaMemset(dC, 0, sizeC * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

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
        diagnoseFP8Support(lt_handle, M, K, N, typeA, typeB);
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
        dA_fp8, layout_A,
        dB_fp8, layout_B,
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

    applyDeqScale<<<(sizeC + threads - 1) / threads, threads>>>(
        dC, 1.0f / scaleA, 1.0f / scaleB, sizeC);
    cudaDeviceSynchronize();

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
