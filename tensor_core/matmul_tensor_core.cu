#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16  
#define WMMA_K 16

__global__ void matmul_tensor_core_kernel(const half* A, const half* B, float* C,
                                         int M, int K, int N) {
    // Calculate warp position in the output matrix
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;
    
    // Bounds check
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;
    
    // Declare fragments for Tensor Core operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
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

// External conversion functions
extern "C" void convert_fp32_to_fp16_cuda(const float* input, half* output, int size);

extern "C" void matmul_tensor_core_cuda(const float* A_fp32, const float* B_fp32, float* C,
                                        int M, int K, int N) {
    // Convert FP32 to FP16 for Tensor Core input
    half *A_fp16, *B_fp16;
    cudaMalloc(&A_fp16, M * K * sizeof(half));
    cudaMalloc(&B_fp16, K * N * sizeof(half));
    
    // Convert matrices to FP16
    convert_fp32_to_fp16_cuda(A_fp32, A_fp16, M * K);
    convert_fp32_to_fp16_cuda(B_fp32, B_fp16, K * N);
    
    // Launch Tensor Core kernel - one block per 16x16 tile
    dim3 blockDim(32, 1); // One warp per block
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    
    matmul_tensor_core_kernel<<<gridDim, blockDim>>>(A_fp16, B_fp16, C, M, K, N);
    cudaDeviceSynchronize();
    
    cudaFree(A_fp16);
    cudaFree(B_fp16);
}