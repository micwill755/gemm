#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>

using namespace nvcuda;
using bfloat16 = __nv_bfloat16;

#define WMMA_M 16
#define WMMA_N 16  
#define WMMA_K 16

__global__ void matmul_tensor_core_kernel(const bfloat16* A, const bfloat16* B, float* C,
                                         int M, int K, int N) {
    // Calculate warp position in the output matrix
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;
    
    // Bounds check
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;
    
    // Declare fragments for Tensor Core operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Loop over K dimension in chunks of WMMA_K
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds check for fragments
        if (aRow + WMMA_M <= M && aCol + WMMA_K <= K && 
            bRow + WMMA_K <= K && bCol + WMMA_N <= N) {
            // Load matrix fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform matrix multiplication using Tensor Cores
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}

__global__ void scale_output_kernel(float* matrix, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] *= scale;
    }
}

__global__ void fp32_to_bf16_kernel(const float* input, bfloat16* output, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2bfloat16(input[idx] * scale);
    }
}

extern "C" void matmul_tensor_core_cuda(const float* A_fp32, const float* B_fp32, float* C,
                                        int M, int K, int N) {
    // Allocate BF16 matrices
    bfloat16 *A_bf16, *B_bf16;
    cudaMalloc(&A_bf16, M * K * sizeof(bfloat16));
    cudaMalloc(&B_bf16, K * N * sizeof(bfloat16));
    
    // Scale inputs to prevent overflow
    float input_scale = 0.1f;
    
    // Convert matrices to BF16 with scaling
    int blockSize = 256;
    int gridSize_A = (M * K + blockSize - 1) / blockSize;
    int gridSize_B = (K * N + blockSize - 1) / blockSize;
    
    fp32_to_bf16_kernel<<<gridSize_A, blockSize>>>(A_fp32, A_bf16, M * K, input_scale);
    fp32_to_bf16_kernel<<<gridSize_B, blockSize>>>(B_fp32, B_bf16, K * N, input_scale);
    
    // Launch Tensor Core kernel
    dim3 blockDim(32, 1);
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    
    matmul_tensor_core_kernel<<<gridDim, blockDim>>>(A_bf16, B_bf16, C, M, K, N);
    
    // Scale output back up
    float output_scale = 1.0f / (input_scale * input_scale);
    int gridSize_C = (M * N + blockSize - 1) / blockSize;
    
    scale_output_kernel<<<gridSize_C, blockSize>>>(C, output_scale, M * N);
    cudaDeviceSynchronize();
    
    cudaFree(A_bf16);
    cudaFree(B_bf16);
}