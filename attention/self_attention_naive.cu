#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

// Forward declarations of kernels
extern "C" void matmul_naive_cuda(const float* A, const float* B, float* C, int M, int K, int N);
extern "C" void transpose_naive_cuda(float *input, float *output, int row, int col);
extern "C" void softmax_naive_cuda(float *input, int rows, int cols);
extern "C" void softmax_tiled_cuda(float *input, int rows, int cols);

// Benchmark helper function
void benchmark_kernel(const char* name, cudaEvent_t start, cudaEvent_t stop, void (*kernel_func)()) {
    cudaEventRecord(start);
    kernel_func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("%s: %.3f ms\n", name, time);
}

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

extern "C" void self_attention_tiled_cuda(
    float* x, float* W_q, float* W_k, float* W_v, float* W_o, float* output,
    int seq_len, int d_model
) {
    float *d_Q, *d_K, *d_V, *d_K_T, *d_scores, *d_context;
    
    cudaMalloc(&d_Q, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_K, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_V, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_K_T, d_model * seq_len * sizeof(float));
    cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float));
    cudaMalloc(&d_context, seq_len * d_model * sizeof(float));
    
    matmul_naive_cuda(x, W_q, d_Q, seq_len, d_model, d_model);
    matmul_naive_cuda(x, W_k, d_K, seq_len, d_model, d_model);
    matmul_naive_cuda(x, W_v, d_V, seq_len, d_model, d_model);
    transpose_naive_cuda(d_K, d_K_T, seq_len, d_model);
    matmul_naive_cuda(d_Q, d_K_T, d_scores, seq_len, d_model, seq_len);
    
    // Use tiled softmax instead of naive
    softmax_tiled_cuda(d_scores, seq_len, seq_len);
    
    matmul_naive_cuda(d_scores, d_V, d_context, seq_len, seq_len, d_model);
    matmul_naive_cuda(d_context, W_o, output, seq_len, d_model, d_model);
    
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_K_T); cudaFree(d_scores); cudaFree(d_context);
}

int main() {
    int seq_len = 4, d_model = 4;
    int input_size = seq_len * d_model;
    int weight_size = d_model * d_model;
    
    // Allocate host memory
    float *h_x = (float*)malloc(input_size * sizeof(float));
    float *h_Wq = (float*)malloc(weight_size * sizeof(float));
    float *h_Wk = (float*)malloc(weight_size * sizeof(float));
    float *h_Wv = (float*)malloc(weight_size * sizeof(float));
    float *h_Wo = (float*)malloc(weight_size * sizeof(float));
    float *h_output = (float*)malloc(input_size * sizeof(float));
    
    // Initialize with random values
    srand(42);
    for(int i = 0; i < input_size; i++) h_x[i] = (float)rand()/RAND_MAX;
    for(int i = 0; i < weight_size; i++) {
        h_Wq[i] = (float)rand()/RAND_MAX - 0.5f;
        h_Wk[i] = (float)rand()/RAND_MAX - 0.5f;
        h_Wv[i] = (float)rand()/RAND_MAX - 0.5f;
        h_Wo[i] = (float)rand()/RAND_MAX - 0.5f;
    }
    
    // Allocate GPU memory and copy data
    float *d_x, *d_Wq, *d_Wk, *d_Wv, *d_Wo, *d_output;
    cudaMalloc(&d_x, input_size * sizeof(float));
    cudaMalloc(&d_Wq, weight_size * sizeof(float));
    cudaMalloc(&d_Wk, weight_size * sizeof(float));
    cudaMalloc(&d_Wv, weight_size * sizeof(float));
    cudaMalloc(&d_Wo, weight_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    
    cudaMemcpy(d_x, h_x, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wq, h_Wq, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wk, h_Wk, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wv, h_Wv, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wo, h_Wo, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // GPU timing using CUDA events (more accurate than CPU timers)
    // Events mark points in the GPU execution stream
    cudaEvent_t start, stop;
    cudaEventCreate(&start);  // Create start timestamp marker
    cudaEventCreate(&stop);   // Create stop timestamp marker
    
    printf("=== Detailed Kernel Benchmarks ===\n");
    
    // Allocate intermediate results for individual benchmarking
    float *d_Q, *d_K, *d_V, *d_K_T, *d_scores, *d_context;
    cudaMalloc(&d_Q, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_K, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_V, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_K_T, d_model * seq_len * sizeof(float));
    cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float));
    cudaMalloc(&d_context, seq_len * d_model * sizeof(float));
    
    // Benchmark individual kernels
    printf("\n--- Individual Kernel Times ---\n");
    
    // MatMul QKV projections
    cudaEventRecord(start);
    matmul_naive_cuda(d_x, d_Wq, d_Q, seq_len, d_model, d_model);
    matmul_naive_cuda(d_x, d_Wk, d_K, seq_len, d_model, d_model);
    matmul_naive_cuda(d_x, d_Wv, d_V, seq_len, d_model, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float matmul_qkv_time = 0;
    cudaEventElapsedTime(&matmul_qkv_time, start, stop);
    printf("MatMul QKV projections: %.3f ms\n", matmul_qkv_time);
    
    // Transpose
    cudaEventRecord(start);
    transpose_naive_cuda(d_K, d_K_T, seq_len, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float transpose_time = 0;
    cudaEventElapsedTime(&transpose_time, start, stop);
    printf("Transpose: %.3f ms\n", transpose_time);
    
    // Attention scores MatMul
    cudaEventRecord(start);
    matmul_naive_cuda(d_Q, d_K_T, d_scores, seq_len, d_model, seq_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float matmul_scores_time = 0;
    cudaEventElapsedTime(&matmul_scores_time, start, stop);
    printf("Attention scores MatMul: %.3f ms\n", matmul_scores_time);
    
    // Softmax naive
    cudaEventRecord(start);
    softmax_naive_cuda(d_scores, seq_len, seq_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float softmax_naive_time = 0;
    cudaEventElapsedTime(&softmax_naive_time, start, stop);
    printf("Softmax (naive): %.3f ms\n", softmax_naive_time);
    
    // Reset scores for tiled softmax test
    matmul_naive_cuda(d_Q, d_K_T, d_scores, seq_len, d_model, seq_len);
    
    // Softmax tiled
    cudaEventRecord(start);
    softmax_tiled_cuda(d_scores, seq_len, seq_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float softmax_tiled_time = 0;
    cudaEventElapsedTime(&softmax_tiled_time, start, stop);
    printf("Softmax (tiled): %.3f ms\n", softmax_tiled_time);
    printf("Softmax speedup: %.2fx\n", softmax_naive_time / softmax_tiled_time);
    
    // Context MatMul
    cudaEventRecord(start);
    matmul_naive_cuda(d_scores, d_V, d_context, seq_len, seq_len, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float matmul_context_time = 0;
    cudaEventElapsedTime(&matmul_context_time, start, stop);
    printf("Context MatMul: %.3f ms\n", matmul_context_time);
    
    // Output projection MatMul
    cudaEventRecord(start);
    matmul_naive_cuda(d_context, d_Wo, d_output, seq_len, d_model, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float matmul_output_time = 0;
    cudaEventElapsedTime(&matmul_output_time, start, stop);
    printf("Output projection MatMul: %.3f ms\n", matmul_output_time);
    
    printf("\n--- Full Attention Benchmarks ---\n");
    
    // Benchmark naive version
    cudaEventRecord(start);
    self_attention_naive_cuda(d_x, d_Wq, d_Wk, d_Wv, d_Wo, d_output, seq_len, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naive_time = 0;
    cudaEventElapsedTime(&naive_time, start, stop);
    printf("Full attention (naive): %.3f ms\n", naive_time);
    
    // Benchmark tiled version
    cudaEventRecord(start);
    self_attention_tiled_cuda(d_x, d_Wq, d_Wk, d_Wv, d_Wo, d_output, seq_len, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiled_time = 0;
    cudaEventElapsedTime(&tiled_time, start, stop);
    printf("Full attention (tiled): %.3f ms\n", tiled_time);
    
    printf("Full attention speedup: %.2fx\n", naive_time / tiled_time);
    
    // Cleanup intermediate allocations
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_K_T); cudaFree(d_scores); cudaFree(d_context);

    // Copy result back
    cudaMemcpy(h_output, d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Self-attention output:\n");
    for(int i = 0; i < seq_len; i++) {
        for(int j = 0; j < d_model; j++) {
            printf("%.3f ", h_output[i * d_model + j]);
        }
        printf("\n");
    }
    
    // Clean up event objects
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Cleanup
    free(h_x); free(h_Wq); free(h_Wk); free(h_Wv); free(h_Wo); free(h_output);
    cudaFree(d_x); cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv); cudaFree(d_Wo); cudaFree(d_output);
    
    return 0;
}