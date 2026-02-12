#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// External function declarations
extern "C" void matmul_tiled_cuda(const float* A, const float* B, float* C, int M, int K, int N);
extern "C" void matmul_tensor_core_cuda(const float* A, const float* B, float* C, int M, int K, int N);

void init_matrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

float compare_matrices(const float* A, const float* B, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(A[i] - B[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <M> <K> <N>\\n", argv[0]);
        return 1;
    }
    
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\\n", M, K, K, N, M, N);
    
    // Allocate host memory
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C_tiled = (float*)malloc(M * N * sizeof(float));
    float *h_C_tensor = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices
    srand(42);
    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C_tiled, *d_C_tensor;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C_tiled, M * N * sizeof(float));
    cudaMalloc(&d_C_tensor, M * N * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Benchmark tiled implementation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("\\n=== Tiled Implementation ===");
    cudaEventRecord(start);
    matmul_tiled_cuda(d_A, d_B, d_C_tiled, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiled_time;
    cudaEventElapsedTime(&tiled_time, start, stop);
    printf("\\nTiled time: %.3f ms\\n", tiled_time);
    
    // Benchmark Tensor Core implementation
    printf("\\n=== Tensor Core Implementation ===");
    cudaEventRecord(start);
    matmul_tensor_core_cuda(d_A, d_B, d_C_tensor, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tensor_time;
    cudaEventElapsedTime(&tensor_time, start, stop);
    printf("\\nTensor Core time: %.3f ms\\n", tensor_time);
    
    // Copy results back
    cudaMemcpy(h_C_tiled, d_C_tiled, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_tensor, d_C_tensor, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare results
    float max_diff = compare_matrices(h_C_tiled, h_C_tensor, M * N);
    printf("\\n=== Results ===");
    printf("\\nSpeedup: %.2fx\\n", tiled_time / tensor_time);
    printf("Max difference: %.6f\\n", max_diff);
    printf("Accuracy: %s\\n", max_diff < 1e-2 ? "PASS" : "FAIL");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_tiled); free(h_C_tensor);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_tiled); cudaFree(d_C_tensor);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}