#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fp32_to_fp16_kernel(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void fp16_to_fp32_kernel(const half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

extern "C" void convert_fp32_to_fp16_cuda(const float* input, half* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    fp32_to_fp16_kernel<<<gridSize, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}

extern "C" void convert_fp16_to_fp32_cuda(const half* input, float* output, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    fp16_to_fp32_kernel<<<gridSize, blockSize>>>(input, output, size);
    cudaDeviceSynchronize();
}