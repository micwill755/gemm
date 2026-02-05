#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 32

__global__ void softmax_tiled_kernel(float *input, int rows, int cols) {
    __shared__ float sdata[TILE_SIZE];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < rows) {
        // Phase 1: Find global max for this row using reduction
        float thread_max = -INFINITY;
        
        // Each thread processes multiple elements
        for (int i = tid; i < cols; i += TILE_SIZE) {
            thread_max = fmaxf(thread_max, input[row * cols + i]);
        }
        
        sdata[tid] = thread_max;
        __syncthreads();
        
        // 6X speed up using parallelism technique tree reduction to find row max
        // Parallel reduction: 5 steps, multiple threads
        // Sequential loop: 31 steps, 1 thread
        for (int s = TILE_SIZE/2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        
        float row_max = sdata[0];
        
        // Phase 2: Compute sum using same reduction pattern
        float thread_sum = 0.0f;
        for (int i = tid; i < cols; i += TILE_SIZE) {
            thread_sum += expf(input[row * cols + i] - row_max);
        }
        
        sdata[tid] = thread_sum;
        __syncthreads();
        
        // Reduction to find row sum
        for (int s = TILE_SIZE/2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        float row_sum = sdata[0];
        
        // Phase 3: Normalize - we must visit every element to update it, 
        // so tree reduction doesn't apply here. We need element-wise operations.
        for (int i = tid; i < cols; i += TILE_SIZE) {
            input[row * cols + i] = expf(input[row * cols + i] - row_max) / row_sum;
        }
    }
}

extern "C" void softmax_tiled_cuda(float *input, int rows, int cols) {
    dim3 blockSize(TILE_SIZE);
    dim3 gridSize(rows);
    
    softmax_tiled_kernel<<<gridSize, blockSize>>>(input, rows, cols);
    cudaDeviceSynchronize();
}