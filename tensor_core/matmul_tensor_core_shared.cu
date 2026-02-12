#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16  
#define WMMA_K 16
#define BLOCK_SIZE_M 32  // 2 warps in M dimension
#define BLOCK_SIZE_N 32  // 2 warps in N dimension

__global__ void scale_output_kernel(float* matrix, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] *= scale;
    }
}

__global__ void fp32_to_fp16_kernel(const float* input, half* output, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx] * scale);
    }
}

__global__ void matmul_tensor_core_shared_kernel(const half* A, const half* B, float* C,
                                                int M, int K, int N) {
    // Shared memory for cooperative loading
    __shared__ half A_shared[BLOCK_SIZE_M * WMMA_K];
    __shared__ half B_shared[WMMA_K * BLOCK_SIZE_N];
    
    // Block and warp indices
    int blockM = blockIdx.y;
    int blockN = blockIdx.x;
    int warpId = threadIdx.y;
    int laneId = threadIdx.x;
    
    // Calculate warp position within block (2x2 warp arrangement)
    int warpM = warpId / 2;  // 0 or 1
    int warpN = warpId % 2;  // 0 or 1
    
    // Global warp position
    int globalWarpM = blockM * 2 + warpM;
    int globalWarpN = blockN * 2 + warpN;
    
    // Bounds check for this warp's output tile
    if (globalWarpM * WMMA_M >= M || globalWarpN * WMMA_N >= N) return;
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Loop over K dimension in chunks
    for (int k = 0; k < K; k += WMMA_K) {
        // Cooperative loading to shared memory
        // Each warp loads part of the shared tile
        
        // Load A block: [blockM*32 : blockM*32+32, k : k+16]
        int a_row = blockM * BLOCK_SIZE_M + warpId * 8 + laneId / 4;
        int a_col = k + (laneId % 4) * 4;
        
        if (a_row < M && a_col < K && a_col + 4 <= K) {
            // Load 4 consecutive elements per thread for coalescing
            for (int i = 0; i < 4; i++) {
                if (a_col + i < K) {
                    A_shared[(warpId * 8 + laneId / 4) * WMMA_K + (laneId % 4) * 4 + i] = 
                        A[a_row * K + a_col + i];
                }
            }
        }
        
        // Load B block: [k : k+16, blockN*32 : blockN*32+32]  
        int b_row = k + warpId * 4 + laneId / 8;
        int b_col = blockN * BLOCK_SIZE_N + (laneId % 8) * 4;
        
        if (b_row < K && b_col < N && b_col + 4 <= N) {
            // Load 4 consecutive elements per thread for coalescing
            for (int i = 0; i < 4; i++) {
                if (b_col + i < N) {
                    B_shared[(warpId * 4 + laneId / 8) * BLOCK_SIZE_N + (laneId % 8) * 4 + i] = 
                        B[b_row * N + b_col + i];
                }
            }
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Bounds check for this K iteration
        if (k + WMMA_K <= K) {
            // Load fragments from shared memory
            int a_shared_offset = warpM * WMMA_M * WMMA_K;
            int b_shared_offset = warpN * WMMA_N;
            
            wmma::load_matrix_sync(a_frag, A_shared + a_shared_offset, WMMA_K);
            wmma::load_matrix_sync(b_frag, B_shared + b_shared_offset, BLOCK_SIZE_N);
            
            // Perform matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        // Synchronize before next iteration
        __syncthreads();
    }
    
    // Store result
    int cRow = globalWarpM * WMMA_M;
    int cCol = globalWarpN * WMMA_N;
    if (cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}

extern "C" void matmul_tensor_core_shared_cuda(const float* A_fp32, const float* B_fp32, float* C,
                                              int M, int K, int N) {
    // Allocate FP16 matrices
    half *A_fp16, *B_fp16;
    cudaMalloc(&A_fp16, M * K * sizeof(half));
    cudaMalloc(&B_fp16, K * N * sizeof(half));
    
    // Use smaller scale to prevent FP16 overflow
    float input_scale = 0.01f;
    
    // Convert matrices to FP16 with scaling
    int blockSize = 256;
    int gridSize_A = (M * K + blockSize - 1) / blockSize;
    int gridSize_B = (K * N + blockSize - 1) / blockSize;
    
    fp32_to_fp16_kernel<<<gridSize_A, blockSize>>>(A_fp32, A_fp16, M * K, input_scale);
    fp32_to_fp16_kernel<<<gridSize_B, blockSize>>>(B_fp32, B_fp16, K * N, input_scale);
    
    // Launch Tensor Core kernel with 4 warps per block (2x2 arrangement)
    dim3 blockDim(32, 4);  // 32 threads per warp, 4 warps per block
    dim3 gridDim((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    
    matmul_tensor_core_shared_kernel<<<gridDim, blockDim>>>(A_fp16, B_fp16, C, M, K, N);
    
    // Scale output back up
    float output_scale = 1.0f / (input_scale * input_scale);
    int gridSize_C = (M * N + blockSize - 1) / blockSize;
    
    scale_output_kernel<<<gridSize_C, blockSize>>>(C, output_scale, M * N);
    cudaDeviceSynchronize();
    
    cudaFree(A_fp16);
    cudaFree(B_fp16);
}