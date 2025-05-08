#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int N = 8192;
    size_t size = N * N * sizeof(float);
    float *A, *B, *C;

    cudaMalloc(&A, size);
    cudaMalloc(&B, size);
    cudaMalloc(&C, size);

    // Fill A and B on host then copy to device...

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, A, N, B, N, &beta, C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("cuBLAS latency: %.2f ms\n", ms);

    cudaFree(A); cudaFree(B); cudaFree(C);
    cublasDestroy(handle);
    return 0;
}

