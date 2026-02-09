#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

extern "C" void transpose_tiled_cuda(const float *input, float *output, int rows, int cols);

__global__ void matmul_tiled_kernel(const float* A, const float* B_T, float* C, 
                                     int M, int K, int N) {
    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A into shared memory 
        // within each warp memory is coalesced, but between warps there's a stride of K elements.
        // to find the next row start index

        // Memory coalescing is only evaluated within a warp (32 threads). So if we are able, we want to ensure each thread
        // access memmory consecutively. Coalescing is a warp-level optimization

        // Warp i - find start row - (blockIdx.y * TILE_SIZE + threadIdx.y) * K
        // then process consecutively
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load tile from B^T into shared memory
        // B^T is now coalesced! Row-wise access instead of column-wise
        /*
        Matrix B^T (Row-wise access - COALESCED!):
        ┌─────┬─────┬─────┬─────┐
        │  →  │  →  │  →  │  →  │ ← Thread (0,0-3) - Consecutive addresses!
        │  →  │  →  │  →  │  →  │ ← Thread (1,0-3) - Consecutive addresses!
        │  →  │  →  │  →  │  →  │ ← Thread (2,0-3) - Consecutive addresses!
        │ ... │ ... │ ... │ ... │
        └─────┴─────┴─────┴─────┘
        
        After transpose: 1 memory transaction serves all 32 threads in a warp!
        Perfect coalescing achieved ✅
        */
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B_T[col * K + bRow];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch kernel
extern "C" void matmul_coalesced_b_cuda(const float* A, const float* B, float* C,
                                   int M, int K, int N) {
    // Allocate memory for transposed B
    float* B_T;
    cudaMalloc(&B_T, K * N * sizeof(float));
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // 1. Transpose B matrix for coalesced access
    transpose_tiled_cuda(B, B_T, K, N);
    
    // 2. Run matrix multiplication with transposed B
    matmul_tiled_kernel<<<gridDim, blockDim>>>(A, B_T, C, M, K, N);
    cudaDeviceSynchronize();
    
    // 3. Clean up transposed matrix
    cudaFree(B_T);
}