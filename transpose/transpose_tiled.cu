#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 32

__global__ void transpose_shared_kernel(float *input, float *output, int rows, int cols) {
    // Shared memory tile - add +1 to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    // Thread indices
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load data into shared memory (coalesced read)
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Calculate transposed coordinates
    int x_trans = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y_trans = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Write transposed data (coalesced write)
    if (x_trans < rows && y_trans < cols) {
        output[y_trans * rows + x_trans] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void transpose_shared_cuda(float *input, float *output, int rows, int cols) {
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
    
    transpose_shared_kernel<<<gridSize, blockSize>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
