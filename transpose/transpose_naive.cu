#include <cuda_runtime.h>
#include <stdio.h>

__global__ void transpose_naive_kernel(long *input, long *output, int row, int col) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < row && j < col) {
        output[j * row + i] = input[i * col + j];
    }
}

// Host function to launch kernel
extern "C" void transpose_naive_cuda(long *input, long *output, int row, int col) {
    dim3 block(16, 16);
    dim3 grid((col + block.x - 1) / block.x, (row + block.y - 1) / block.y);
    transpose_naive_kernel<<<grid, block>>>(input, output, row, col);

    cudaDeviceSynchronize();
}
