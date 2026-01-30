#include <cuda_runtime.h>
#include <stdio.h>

// Naive matrix multiplication kernel: C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, 
                                     int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function to launch kernel
extern "C" void matmul_naive_cuda(const float* A, const float* B, float* C,
                                   int M, int K, int N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    matmul_naive_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
