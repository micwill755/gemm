#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Forward declarations
extern "C" void self_attention_naive_cuda(float* x, float* W_q, float* W_k, float* W_v, float* W_o, float* output, int seq_len, int d_model);
extern "C" void self_attention_tiled_cuda(float* x, float* W_q, float* W_k, float* W_v, float* W_o, float* output, int seq_len, int d_model);

int main(int argc, char *argv[]) {
    int seq_len = 128, d_model = 512;
    
    if (argc == 3) {
        seq_len = atoi(argv[1]);
        d_model = atoi(argv[2]);
    } else if (argc != 1) {
        printf("Usage: %s [seq_len d_model]\n", argv[0]);
        return 1;
    }
    
    printf("Benchmarking attention implementations: seq_len=%d, d_model=%d\n", seq_len, d_model);
    
    int input_size = seq_len * d_model;
    int weight_size = d_model * d_model;
    
    // Allocate host memory
    float *h_x = (float*)malloc(input_size * sizeof(float));
    float *h_Wq = (float*)malloc(weight_size * sizeof(float));
    float *h_Wk = (float*)malloc(weight_size * sizeof(float));
    float *h_Wv = (float*)malloc(weight_size * sizeof(float));
    float *h_Wo = (float*)malloc(weight_size * sizeof(float));
    float *h_output_naive = (float*)malloc(input_size * sizeof(float));
    float *h_output_tiled = (float*)malloc(input_size * sizeof(float));
    
    if (!h_x || !h_Wq || !h_Wk || !h_Wv || !h_Wo || !h_output_naive || !h_output_tiled) {
        printf("Host malloc failed\n");
        return 1;
    }
    
    // Initialize with random values
    srand(42);
    for(int i = 0; i < input_size; i++) h_x[i] = (float)rand()/RAND_MAX;
    for(int i = 0; i < weight_size; i++) {
        h_Wq[i] = (float)rand()/RAND_MAX - 0.5f;
        h_Wk[i] = (float)rand()/RAND_MAX - 0.5f;
        h_Wv[i] = (float)rand()/RAND_MAX - 0.5f;
        h_Wo[i] = (float)rand()/RAND_MAX - 0.5f;
    }
    
    // Allocate GPU memory
    float *d_x, *d_Wq, *d_Wk, *d_Wv, *d_Wo, *d_output_naive, *d_output_tiled;
    cudaMalloc(&d_x, input_size * sizeof(float));
    cudaMalloc(&d_Wq, weight_size * sizeof(float));
    cudaMalloc(&d_Wk, weight_size * sizeof(float));
    cudaMalloc(&d_Wv, weight_size * sizeof(float));
    cudaMalloc(&d_Wo, weight_size * sizeof(float));
    cudaMalloc(&d_output_naive, input_size * sizeof(float));
    cudaMalloc(&d_output_tiled, input_size * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_x, h_x, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wq, h_Wq, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wk, h_Wk, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wv, h_Wv, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wo, h_Wo, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("\n=== Attention Implementation Benchmark ===\n");
    
    // Benchmark naive implementation
    cudaEventRecord(start);
    self_attention_naive_cuda(d_x, d_Wq, d_Wk, d_Wv, d_Wo, d_output_naive, seq_len, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naive_time = 0;
    cudaEventElapsedTime(&naive_time, start, stop);
    printf("Naive attention: %.3f ms\n", naive_time);
    
    // Benchmark tiled implementation
    cudaEventRecord(start);
    self_attention_tiled_cuda(d_x, d_Wq, d_Wk, d_Wv, d_Wo, d_output_tiled, seq_len, d_model);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiled_time = 0;
    cudaEventElapsedTime(&tiled_time, start, stop);
    printf("Tiled attention: %.3f ms\n", tiled_time);
    
    printf("Speedup: %.2fx\n", naive_time / tiled_time);
    
    // Copy results back and verify correctness
    cudaMemcpy(h_output_naive, d_output_naive, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_tiled, d_output_tiled, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check correctness (first few elements)
    float max_diff = 0.0f;
    for(int i = 0; i < input_size && i < 16; i++) {
        float diff = fabsf(h_output_naive[i] - h_output_tiled[i]);
        if(diff > max_diff) max_diff = diff;
    }
    printf("Max difference (first 16 elements): %.6f\n", max_diff);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_x); free(h_Wq); free(h_Wk); free(h_Wv); free(h_Wo);
    free(h_output_naive); free(h_output_tiled);
    cudaFree(d_x); cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv); cudaFree(d_Wo);
    cudaFree(d_output_naive); cudaFree(d_output_tiled);
    
    return 0;
}