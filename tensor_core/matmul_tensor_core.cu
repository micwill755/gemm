#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16  
#define WMMA_K 16

// External kernel declarations
__global__ void scale_output_kernel(float* matrix, float scale, int size);
__global__ void fp32_to_fp16_kernel(const float* input, half* output, int size, float scale);

__global__ void matmul_tensor_core_kernel(const half* A, const half* B, float* C,
                                         int M, int K, int N) {
    // Calculate warp position in the output matrix
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;
    
    // Bounds check
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;
    
    // declare warp-level storage containers for Tensor Core operations
    // these fragments are stored in thread registers - fastest memory on the gpu
    /*

    Storage hierarchy:

    Each thread's registers: Holds ~8 elements of each fragment
    Warp collectively: All 32 threads' registers together form the complete 16×16 fragment
    Hardware managed: The compiler/hardware automatically distributes elements across threads
    
    */
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // Initialize accumulator to zero - zeros out the accumulator fragment across all 32 threads in the warp
    // i.e. Sets all 256 elements (16×16) of the accumulator to 0.0
    /*
    
    Why it's needed:
    Matrix multiplication is C = A × B, but when K is large, you compute it as:

    C += A[0:16] × B[0:16]    // First K chunk
    C += A[16:32] × B[16:32]  // Second K chunk  
    C += A[32:48] × B[32:48]  // Third K chunk...

    Copy

    Insert at cursor
    Without zeroing: Accumulator contains garbage values
    With zeroing: Clean slate for proper accumulation

    This is like initializing sum = 0 before a loop - you need a known starting point for the accumulation to work correctly across all the K-dimension chunks.
    */
    
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
            // Load matrix fragments - Yes! wmma::load_matrix_sync loads data from global memory into the fragment registers, preparing it for Tensor Core computation.
            /*
            
            Data flow:

            1. Global memory → Fragment registers (via load_matrix_sync)
            2. Fragment registers → Tensor Core units (via mma_sync)
            3. Tensor Core result → Fragment registers (accumulator)
            4. Fragment registers → Global memory (via store_matrix_sync)
            */
            
            /*
                memory transfers are the bottleneck, but Tensor Cores still win because of massive compute density.

                The trade-off:
                Memory cost: Load 512 elements (2×256) from global memory
                Compute benefit: Get 4,096 operations (16×16×16) from Tensor Cores

                Why it's still fast:

                Coalesced access: 32 threads load simultaneously → high bandwidth utilization
                Compute intensity: 4,096 ops per 512 elements = 8:1 compute-to-memory ratio
                Hardware acceleration: Tensor Core does 16×16×16 in ~1 cycle vs hundreds for regular cores
                
                Each warp loads its own 16×16 tiles independently, even if they share rows/columns.
                For this single-warp-per-block approach, shared memory won't help much since there's no data sharing between warps.
            */
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

extern "C" void matmul_tensor_core_cuda(const float* A_fp32, const float* B_fp32, float* C,
                                        int M, int K, int N) {
    // Allocate FP16 matrices
    half *A_fp16, *B_fp16;
    cudaMalloc(&A_fp16, M * K * sizeof(half));
    cudaMalloc(&B_fp16, K * N * sizeof(half));
    
    // Use smaller scale to prevent FP16 overflow
    float input_scale = 0.01f;  // Much smaller scale
    
    // Convert matrices to FP16 with scaling
    int blockSize = 256;
    int gridSize_A = (M * K + blockSize - 1) / blockSize;
    int gridSize_B = (K * N + blockSize - 1) / blockSize;
    
    fp32_to_fp16_kernel<<<gridSize_A, blockSize>>>(A_fp32, A_fp16, M * K, input_scale);
    fp32_to_fp16_kernel<<<gridSize_B, blockSize>>>(B_fp32, B_fp16, K * N, input_scale);
    
    // Launch Tensor Core kernel
    dim3 blockDim(32, 1);
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    
    matmul_tensor_core_kernel<<<gridDim, blockDim>>>(A_fp16, B_fp16, C, M, K, N);
    
    // Scale output back up
    float output_scale = 1.0f / (input_scale * input_scale);
    int gridSize_C = (M * N + blockSize - 1) / blockSize;
    
    scale_output_kernel<<<gridSize_C, blockSize>>>(C, output_scale, M * N);
    cudaDeviceSynchronize();
    
    cudaFree(A_fp16);
    cudaFree(B_fp16);
}