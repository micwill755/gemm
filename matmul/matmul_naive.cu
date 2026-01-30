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
        // the problem with this kernel loop 
        /*
            Why it's slow:
            For K=1024, each thread reads from global memory 2048 times (1024 from A, 1024 from B).
            
            Thread accesses A[row * K + k]
                ↓
            Registers (fastest, per-thread)
                ↓ miss
            L1 Cache (~128 KB per SM)
                ↓ miss
            L2 Cache (~shared across GPU)
                ↓ miss
            Global Memory (slowest, ~GB in size) ← A, B, C live here
             
            The problem:

            Global memory latency: ~400-800 cycles

            No data reuse between threads
            Thread 0 reads A[0][0], A[0][1], ..., A[0][1023]
            Thread 1 reads A[0][0], A[0][1], ..., A[0][1023] ← same data!
            Threads in the same row all read the same A values
            Threads in the same column all read the same B values

            Solution: Shared Memory (Tiled/Blocked approach)

            Load tiles of A and B into shared memory (fast, on-chip)
            All threads in a block reuse the same tile
            Reduces global memory accesses by ~16x (for 16×16 tiles)
        */
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