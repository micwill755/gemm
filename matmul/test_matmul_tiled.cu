// Test program for naive matmul kernel
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                   int M, int K, int N);

int main() {
    int M = 1024, K = 1024, N = 1024;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Warmup
    matmul_tiled_kernel(d_A, d_B, d_C, M, K, N);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul_tiled_kernel(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Verify (should be K for all elements)
    printf("Result[0] = %.2f (expected %.2f)\n", h_C[0], (float)K);
    printf("Time: %.3f ms\n", ms);
    printf("GFLOPS: %.2f\n", (2.0 * M * N * K) / (ms * 1e6));
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
