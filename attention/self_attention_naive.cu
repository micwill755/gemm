#include <cuda_runtime.h>
#include <stdio.h>

// Forward declarations of your naive kernels
extern "C" void matmul_naive_cuda(const float* A, const float* B, float* C, int M, int K, int N);
extern "C" void transpose_naive_cuda(float *input, float *output, int row, int col);
extern "C" void softmax_naive_cuda(float *input, int rows, int cols);

extern "C" void self_attention_naive_cuda(
    float* x,           // Input [seq_len × d_model]
    float* W_q,         // Query weights [d_model × d_model]  
    float* W_k,         // Key weights [d_model × d_model]
    float* W_v,         // Value weights [d_model × d_model]
    float* W_o,         // Output weights [d_model × d_model]
    float* output,      // Output [seq_len × d_model]
    int seq_len,
    int d_model
) {
    // Allocate GPU memory for intermediate results
    float *d_Q, *d_K, *d_V, *d_K_T, *d_scores, *d_context;
    
    cudaMalloc(&d_Q, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_K, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_V, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_K_T, d_model * seq_len * sizeof(float));
    cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float));
    cudaMalloc(&d_context, seq_len * d_model * sizeof(float));
    
    // Step 1: Compute Q, K, V projections
    // Q = X × W_q
    matmul_naive_cuda(x, W_q, d_Q, seq_len, d_model, d_model);
    // K = X × W_k  
    matmul_naive_cuda(x, W_k, d_K, seq_len, d_model, d_model);
    // V = X × W_v
    matmul_naive_cuda(x, W_v, d_V, seq_len, d_model, d_model);
    
    // Step 2: Transpose K
    // K_T = K^T [d_model × seq_len]
    transpose_naive_cuda(d_K, d_K_T, seq_len, d_model);
    
    // Step 3: Compute attention scores
    // scores = Q × K_T [seq_len × seq_len]
    matmul_naive_cuda(d_Q, d_K_T, d_scores, seq_len, d_model, seq_len);
    
    // Step 4: Apply softmax to attention scores (in-place)
    softmax_naive_cuda(d_scores, seq_len, seq_len);
    
    // Step 5: Compute context
    // context = scores × V [seq_len × d_model]
    matmul_naive_cuda(d_scores, d_V, d_context, seq_len, seq_len, d_model);
    
    // Step 6: Final output projection
    // output = context × W_o [seq_len × d_model]
    matmul_naive_cuda(d_context, W_o, output, seq_len, d_model, d_model);
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_K_T);
    cudaFree(d_scores);
    cudaFree(d_context);
}