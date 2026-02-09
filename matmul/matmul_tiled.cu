#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, 
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
        
        // Load tile from B into shared memory
        // B is not coalesced because we're accessing down a column , which requires striding by N elements.
        /*
        Matrix B (Column-wise access):
        ┌─────┬─────┬─────┬─────┐
        │  ↓  │     │     │     │ ← Thread (0,0)
        │  ↓  │     │     │     │ ← Thread (1,0) - Stride N
        │  ↓  │     │     │     │ ← Thread (2,0) - Stride N  
        │  ↓  │     │     │     │ ← Thread (3,0) - Stride N
        │ ... │     │     │     │
        └─────┴─────┴─────┴─────┘
        ↑
        Column access = Large strides

        Why This Hurts Performance
        Memroy addresses are 256 bytes apart (64 floats × 4 bytes), but memory transactions are only 128 bytes.
        Each thread's address falls in a different 128-byte block:

        Instead of 1 memory transaction:
        32 threads → 32 separate memory transactions (separate transactions because addresses are too far apart to 
        fit in the same 128-byte window - in 8 we could do 1 single memory transaction for all 32 elements)
        32x more memory traffic!
        */
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
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
extern "C" void matmul_tiled_cuda(const float* A, const float* B, float* C,
                                   int M, int K, int N) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    // grid setup
    /*
    example Grid Layout (for 64×64 matrix with TILE_SIZE=16)
    Grid: 4×4 blocks - 64÷16 = 4 blocks needed per dimension

        blockIdx.x →
       ┌─────┬─────┬─────┬─────┐
    b  │(0,0)│(1,0)│(2,0)│(3,0)│
    l  ├─────┼─────┼─────┼─────┤
    o  │(0,1)│(1,1)│(2,1)│(3,1)│
    c  ├─────┼─────┼─────┼─────┤
    k  │(0,2)│(1,2)│(2,2)│(3,2)│
    I  ├─────┼─────┼─────┼─────┤
    d  │(0,3)│(1,3)│(2,3)│(3,3)│
    x  └─────┴─────┴─────┴─────┘
    .
    y
    ↓

    16 blocks (4×4 grid) can run simultaneously (AT THE SAME TIME) on different SMs
    Each block has 256 threads (16×16) working together
    Total parallelism: 16 blocks × 256 threads = 4,096 threads running at once!

    Maximum threads per block: 1024 (on modern GPUs)
    Why 1024 Limit?
    Hardware constraints:
        Warp size: 32 threads (fixed)
        Max warps per block: 32
        32 warps × 32 threads = 1024 threads

    Practical Considerations
        Shared memory: Larger blocks need more shared memory
        Registers: More threads = fewer registers per thread
        Occupancy: Balance threads vs resources for optimal performance

    */
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_tiled_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}