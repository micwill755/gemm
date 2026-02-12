#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// External function declarations
extern "C" void self_attention_tiled_cuda(
    const float* input, const float* W_q, const float* W_k, const float* W_v, const float* W_o,
    float* output, int seq_len, int d_model);

extern "C" void self_attention_tensor_core_cuda(
    const float* input, const float* W_q, const float* W_k, const float* W_v, const float* W_o,
    float* output, int seq_len, int d_model);

void init_matrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;  // Small values for stability
    }
}

void init_identity_weights(float* W, int d_model) {
    // Initialize as identity + small noise for stable attention
    for (int i = 0; i < d_model * d_model; i++) {
        W[i] = 0.0f;
    }
    for (int i = 0; i < d_model; i++) {
        W[i * d_model + i] = 1.0f + ((float)rand() / RAND_MAX) * 0.01f;
    }
}

float compare_matrices(const float* A, const float* B, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(A[i] - B[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <seq_len> <d_model>\\n", argv[0]);
        return 1;
    }
    
    int seq_len = atoi(argv[1]);
    int d_model = atoi(argv[2]);
    
    printf("Self-Attention dimensions: [%d, %d]\\n", seq_len, d_model);
    printf("Memory usage: ~%.1f MB\\n", 
           (seq_len * d_model * 6 + seq_len * seq_len + d_model * d_model * 4) * sizeof(float) / 1e6);
    
    // Allocate host memory
    float *h_input = (float*)malloc(seq_len * d_model * sizeof(float));
    float *h_W_q = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_W_k = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_W_v = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_W_o = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_output_tiled = (float*)malloc(seq_len * d_model * sizeof(float));
    float *h_output_tensor = (float*)malloc(seq_len * d_model * sizeof(float));
    
    // Initialize matrices
    srand(42);
    init_matrix(h_input, seq_len * d_model);
    init_identity_weights(h_W_q, d_model);
    init_identity_weights(h_W_k, d_model);
    init_identity_weights(h_W_v, d_model);
    init_identity_weights(h_W_o, d_model);
    
    // Allocate device memory
    float *d_input, *d_W_q, *d_W_k, *d_W_v, *d_W_o, *d_output_tiled, *d_output_tensor;
    cudaMalloc(&d_input, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_W_q, d_model * d_model * sizeof(float));
    cudaMalloc(&d_W_k, d_model * d_model * sizeof(float));
    cudaMalloc(&d_W_v, d_model * d_model * sizeof(float));
    cudaMalloc(&d_W_o, d_model * d_model * sizeof(float));
    cudaMalloc(&d_output_tiled, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_output_tensor, seq_len * d_model * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_input, h_input, seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_q, h_W_q, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_k, h_W_k, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_v, h_W_v, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_o, h_W_o, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    // Benchmark tiled attention
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("\\n=== Tiled Self-Attention ===\\n");
    cudaEventRecord(start);
    self_attention_tiled_cuda(d_input, d_W_q, d_W_k, d_W_v, d_W_o, d_output_tiled, seq_len, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiled_time;
    cudaEventElapsedTime(&tiled_time, start, stop);
    printf("Tiled time: %.3f ms\\n", tiled_time);
    
    // Benchmark Tensor Core attention
    printf("\\n=== Tensor Core Self-Attention ===\\n");
    cudaEventRecord(start);
    self_attention_tensor_core_cuda(d_input, d_W_q, d_W_k, d_W_v, d_W_o, d_output_tensor, seq_len, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tensor_time;
    cudaEventElapsedTime(&tensor_time, start, stop);
    printf("Tensor Core time: %.3f ms\\n", tensor_time);
    
    // Copy results back
    cudaMemcpy(h_output_tiled, d_output_tiled, seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_tensor, d_output_tensor, seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare results
    float max_diff = compare_matrices(h_output_tiled, h_output_tensor, seq_len * d_model);
    
    printf("\\n=== Results ===\\n");
    printf("Speedup: %.2fx\\n", tiled_time / tensor_time);
    printf("Max difference: %.6f\\n", max_diff);
    printf("Accuracy: %s\\n", max_diff < 1e-2 ? "PASS" : "FAIL");
    
    // Performance metrics
    long long flops = (long long)seq_len * seq_len * d_model * 2 + // Q×K^T
                      (long long)seq_len * seq_len * d_model * 2 + // attention×V  
                      (long long)seq_len * d_model * d_model * 2 * 4; // 4 linear layers
    
    printf("\\nPerformance:\\n");
    printf("Tiled GFLOPS: %.1f\\n", flops / (tiled_time * 1e6));
    printf("Tensor GFLOPS: %.1f\\n", flops / (tensor_time * 1e6));
    
    // Cleanup
    free(h_input); free(h_W_q); free(h_W_k); free(h_W_v); free(h_W_o);
    free(h_output_tiled); free(h_output_tensor);
    cudaFree(d_input); cudaFree(d_W_q); cudaFree(d_W_k); cudaFree(d_W_v); cudaFree(d_W_o);
    cudaFree(d_output_tiled); cudaFree(d_output_tensor);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}