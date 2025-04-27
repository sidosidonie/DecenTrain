#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__       \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

int main() {
    const int dataSize = 100 * 1024 * 1024;  // 100 MB
    const size_t size = dataSize * sizeof(char);

    char *h_src, *h_dst;
    char *d_src, *d_dst;

    // Allocate host memory
    //h_src = (char *)malloc(size);
    //h_dst = (char *)malloc(size);
    cudaHostAlloc((void**)&h_src, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_dst, size, cudaHostAllocDefault);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_src, size));
    CHECK_CUDA(cudaMalloc((void **)&d_dst, size));

    // Fill source data
    memset(h_src, 1, size);

    cudaEvent_t start, stop;
    float elapsedTime;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Host to Device
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Host to Device bandwidth: " << (size / (elapsedTime * 1e6)) << " GB/s\n";

    // Device to Host
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Device to Host bandwidth: " << (size / (elapsedTime * 1e6)) << " GB/s\n";

    // Device to Device
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "Device to Device bandwidth: " << (size / (elapsedTime * 1e6)) << " GB/s\n";

    // Cleanup
    cudaFreeHost(h_src);
    cudaFreeHost(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
