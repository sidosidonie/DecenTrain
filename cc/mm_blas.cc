#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    int N = 2048;
    float *A = (float*)malloc(sizeof(float) * N * N);
    float *B = (float*)malloc(sizeof(float) * N * N);
    float *C = (float*)malloc(sizeof(float) * N * N);

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    clock_t start = clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    clock_t end = clock();

    printf("CBLAS latency: %.2f ms\n", 1000.0 * (end - start) / CLOCKS_PER_SEC);
    free(A); free(B); free(C);
    return 0;
}

