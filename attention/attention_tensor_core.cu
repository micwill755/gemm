#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <math.h>

// External Tensor Core matmul function
extern "C" void matmul_tensor_core_cuda(const float* A, const float* B, float* C, int M, int K, int N);

// External utility functions
extern "C" void transpose_tiled_cuda(const float* input, float* output, int rows, int cols);
extern "C" void softmax_tiled_cuda(float* matrix, int rows, int cols);

__global__ void scale_kernel(float* matrix, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] *= scale;
    }
}

extern "C" void self_attention_tensor_core_cuda(
    const float* input,     // [seq_len, d_model]
    const float* W_q,       // [d_model, d_model] 
    const float* W_k,       // [d_model, d_model]
    const float* W_v,       // [d_model, d_model]
    const float* W_o,       // [d_model, d_model]
    float* output,          // [seq_len, d_model]
    int seq_len,
    int d_model
) {
    // Allocate intermediate matrices
    float *Q, *K, *V, *K_T, *scores, *attention, *context;
    
    cudaMalloc(&Q, seq_len * d_model * sizeof(float));
    cudaMalloc(&K, seq_len * d_model * sizeof(float)); 
    cudaMalloc(&V, seq_len * d_model * sizeof(float));
    cudaMalloc(&K_T, d_model * seq_len * sizeof(float));
    cudaMalloc(&scores, seq_len * seq_len * sizeof(float));
    cudaMalloc(&attention, seq_len * seq_len * sizeof(float));
    cudaMalloc(&context, seq_len * d_model * sizeof(float));
    
    // Step 1: Compute Q, K, V using Tensor Cores
    // Q = input × W_q
    matmul_tensor_core_cuda(input, W_q, Q, seq_len, d_model, d_model);
    
    // K = input × W_k  
    matmul_tensor_core_cuda(input, W_k, K, seq_len, d_model, d_model);
    
    // V = input × W_v
    matmul_tensor_core_cuda(input, W_v, V, seq_len, d_model, d_model);
    
    // Step 2: Transpose K for efficient attention computation
    transpose_tiled_cuda(K, K_T, seq_len, d_model);
    
    // Step 3: Compute attention scores using Tensor Cores
    // scores = Q × K^T
    matmul_tensor_core_cuda(Q, K_T, scores, seq_len, d_model, seq_len);
    
    // Step 4: Scale by sqrt(d_model)
    float scale = 1.0f / sqrtf((float)d_model);
    int total_scores = seq_len * seq_len;
    int blockSize = 256;
    int gridSize = (total_scores + blockSize - 1) / blockSize;
    scale_kernel<<<gridSize, blockSize>>>(scores, scale, total_scores);
    cudaDeviceSynchronize();
    
    // Step 5: Apply softmax to get attention weights
    softmax_tiled_cuda(scores, seq_len, seq_len);
    cudaMemcpy(attention, scores, seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Step 6: Compute context using Tensor Cores
    // context = attention × V
    matmul_tensor_core_cuda(attention, V, context, seq_len, seq_len, d_model);
    
    // Step 7: Apply output projection using Tensor Cores
    // output = context × W_o
    matmul_tensor_core_cuda(context, W_o, output, seq_len, d_model, d_model);
    
    // Cleanup
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(K_T);
    cudaFree(scores);
    cudaFree(attention);
    cudaFree(context);
}

// Multi-head attention using Tensor Cores
extern "C" void multi_head_attention_tensor_core_cuda(
    const float* input,     // [seq_len, d_model]
    const float* W_q,       // [d_model, d_model]
    const float* W_k,       // [d_model, d_model] 
    const float* W_v,       // [d_model, d_model]
    const float* W_o,       // [d_model, d_model]
    float* output,          // [seq_len, d_model]
    int seq_len,
    int d_model,
    int num_heads
) {
    int head_dim = d_model / num_heads;
    
    // Allocate memory for all heads
    float *all_Q, *all_K, *all_V, *all_output;
    cudaMalloc(&all_Q, seq_len * d_model * sizeof(float));
    cudaMalloc(&all_K, seq_len * d_model * sizeof(float));
    cudaMalloc(&all_V, seq_len * d_model * sizeof(float));
    cudaMalloc(&all_output, seq_len * d_model * sizeof(float));
    
    // Compute Q, K, V for all heads at once using Tensor Cores
    matmul_tensor_core_cuda(input, W_q, all_Q, seq_len, d_model, d_model);
    matmul_tensor_core_cuda(input, W_k, all_K, seq_len, d_model, d_model);
    matmul_tensor_core_cuda(input, W_v, all_V, seq_len, d_model, d_model);
    
    // Process each head
    for (int h = 0; h < num_heads; h++) {
        int offset = h * head_dim;
        
        // Extract head-specific Q, K, V
        float *Q_h = all_Q + offset;
        float *K_h = all_K + offset;  
        float *V_h = all_V + offset;
        float *output_h = all_output + offset;
        
        // Allocate head-specific matrices
        float *K_T_h, *scores_h, *attention_h;
        cudaMalloc(&K_T_h, head_dim * seq_len * sizeof(float));
        cudaMalloc(&scores_h, seq_len * seq_len * sizeof(float));
        cudaMalloc(&attention_h, seq_len * seq_len * sizeof(float));
        
        // Transpose K for this head
        transpose_tiled_cuda(K_h, K_T_h, seq_len, head_dim);
        
        // Attention computation for this head using Tensor Cores
        matmul_tensor_core_cuda(Q_h, K_T_h, scores_h, seq_len, head_dim, seq_len);
        
        // Scale and softmax
        float scale = 1.0f / sqrtf((float)head_dim);
        int total_scores = seq_len * seq_len;
        int blockSize = 256;
        int gridSize = (total_scores + blockSize - 1) / blockSize;
        scale_kernel<<<gridSize, blockSize>>>(scores_h, scale, total_scores);
        
        softmax_tiled_cuda(scores_h, seq_len, seq_len);
        cudaMemcpy(attention_h, scores_h, seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Context computation using Tensor Cores
        matmul_tensor_core_cuda(attention_h, V_h, output_h, seq_len, seq_len, head_dim);
        
        cudaFree(K_T_h);
        cudaFree(scores_h);
        cudaFree(attention_h);
    }
    
    // Final output projection using Tensor Cores
    matmul_tensor_core_cuda(all_output, W_o, output, seq_len, d_model, d_model);
    
    // Cleanup
    cudaFree(all_Q);
    cudaFree(all_K);
    cudaFree(all_V);
    cudaFree(all_output);
}