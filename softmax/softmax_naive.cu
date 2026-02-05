#include <cuda_runtime.h>
#include <stdio.h>

__global__ void softmax_naive_kernel(float *input, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // 1D indexing
    
    if (row < rows) {
        // Step 1: Find row max for numerical stability
        float max_val = -INFINITY;
        for (int j = 0; j < cols; j++) {
            max_val = fmaxf(input[row * cols + j], max_val);
        }

        // Step 2: Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += expf(input[row * cols + j] - max_val);
        }

        // Step 3: Normalize
        for (int j = 0; j < cols; j++) {
            input[row * cols + j] = expf(input[row * cols + j] - max_val) / sum;
        }
    }
}

// Host function to launch kernel
extern "C" void softmax_naive_cuda(float *input, int rows, int cols) {
    // we define block size before calculating grid size:
    // first set threads per block
    dim3 block(256);
    // then how many blocks we need - We scale the number of blocks based on the amount of work (rows), not the data size (cols). 
    // The columns are handled sequentially within each thread.
    dim3 grid((rows + block.x - 1) / block.x);
    softmax_naive_kernel<<<grid, block>>>(input, rows, cols);

    cudaDeviceSynchronize();
}