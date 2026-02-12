#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Forward declarations
extern "C" void self_attention_naive_cuda(float* x, float* W_q, float* W_k, float* W_v, float* W_o, float* output, int seq_len, int d_model);
extern "C" void self_attention_tiled_cuda(float* x, float* W_q, float* W_k, float* W_v, float* W_o, float* output, int seq_len, int d_model);
extern "C" void self_attention_coalesced_b_cuda(float* x, float* W_q, float* W_k, float* W_v, float* W_o, float* output, int seq_len, int d_model);
extern "C" void self_attention_tensor_core_cuda(const float* input, const float* W_q, const float* W_k, const float* W_v, const float* W_o, float* output, int seq_len, int d_model);

// Benchmark data structure
typedef struct {
    float *h_x, *h_Wq, *h_Wk, *h_Wv, *h_Wo;
    float *h_output_naive, *h_output_tiled, *h_output_coalesced, *h_output_tensor;
    float *d_x, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_output_naive, *d_output_tiled, *d_output_coalesced, *d_output_tensor;
    cudaEvent_t start, stop;
} BenchmarkData;

void initialize_data(BenchmarkData* data, int seq_len, int d_model) {
    int input_size = seq_len * d_model;
    int weight_size = d_model * d_model;
    
    // Allocate host memory
    data->h_x = (float*)malloc(input_size * sizeof(float));
    data->h_Wq = (float*)malloc(weight_size * sizeof(float));
    data->h_Wk = (float*)malloc(weight_size * sizeof(float));
    data->h_Wv = (float*)malloc(weight_size * sizeof(float));
    data->h_Wo = (float*)malloc(weight_size * sizeof(float));
    data->h_output_naive = (float*)malloc(input_size * sizeof(float));
    data->h_output_tiled = (float*)malloc(input_size * sizeof(float));
    data->h_output_coalesced = (float*)malloc(input_size * sizeof(float));
    data->h_output_tensor = (float*)malloc(input_size * sizeof(float));
    
    // Initialize with random values
    srand(42);
    for(int i = 0; i < input_size; i++) data->h_x[i] = (float)rand()/RAND_MAX;
    for(int i = 0; i < weight_size; i++) {
        data->h_Wq[i] = (float)rand()/RAND_MAX - 0.5f;
        data->h_Wk[i] = (float)rand()/RAND_MAX - 0.5f;
        data->h_Wv[i] = (float)rand()/RAND_MAX - 0.5f;
        data->h_Wo[i] = (float)rand()/RAND_MAX - 0.5f;
    }
    
    // Allocate GPU memory
    cudaMalloc(&data->d_x, input_size * sizeof(float));
    cudaMalloc(&data->d_Wq, weight_size * sizeof(float));
    cudaMalloc(&data->d_Wk, weight_size * sizeof(float));
    cudaMalloc(&data->d_Wv, weight_size * sizeof(float));
    cudaMalloc(&data->d_Wo, weight_size * sizeof(float));
    cudaMalloc(&data->d_output_naive, input_size * sizeof(float));
    cudaMalloc(&data->d_output_tiled, input_size * sizeof(float));
    cudaMalloc(&data->d_output_coalesced, input_size * sizeof(float));
    cudaMalloc(&data->d_output_tensor, input_size * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(data->d_x, data->h_x, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_Wq, data->h_Wq, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_Wk, data->h_Wk, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_Wv, data->h_Wv, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_Wo, data->h_Wo, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create CUDA events
    cudaEventCreate(&data->start);
    cudaEventCreate(&data->stop);
}

float benchmark_implementation(BenchmarkData* data, const char* name, 
                              void (*func)(float*, float*, float*, float*, float*, float*, int, int),
                              float* output_ptr, int seq_len, int d_model) {
    cudaEventRecord(data->start);
    func(data->d_x, data->d_Wq, data->d_Wk, data->d_Wv, data->d_Wo, output_ptr, seq_len, d_model);
    cudaEventRecord(data->stop);
    cudaEventSynchronize(data->stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, data->start, data->stop);
    printf("%s attention: %.3f ms\n", name, elapsed_time);
    return elapsed_time;
}

float benchmark_tensor_core(BenchmarkData* data, int seq_len, int d_model) {
    cudaEventRecord(data->start);
    self_attention_tensor_core_cuda(data->d_x, data->d_Wq, data->d_Wk, data->d_Wv, data->d_Wo, data->d_output_tensor, seq_len, d_model);
    cudaEventRecord(data->stop);
    cudaEventSynchronize(data->stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, data->start, data->stop);
    printf("Tensor Core attention: %.3f ms\n", elapsed_time);
    return elapsed_time;
}

void print_speedups(float naive_time, float tiled_time, float coalesced_time, float tensor_time) {
    printf("\n=== Speedups vs Naive ===\n");
    printf("Tiled speedup: %.2fx\n", naive_time / tiled_time);
    printf("Coalesced B speedup: %.2fx\n", naive_time / coalesced_time);
    printf("Tensor Core speedup: %.2fx\n", naive_time / tensor_time);
}

void verify_correctness(BenchmarkData* data, int seq_len, int d_model) {
    int input_size = seq_len * d_model;
    
    // Copy results back
    cudaMemcpy(data->h_output_naive, data->d_output_naive, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data->h_output_tiled, data->d_output_tiled, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data->h_output_coalesced, data->d_output_coalesced, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data->h_output_tensor, data->d_output_tensor, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check correctness
    printf("\n=== Accuracy Check ===\n");
    float max_diff_tiled = 0.0f, max_diff_coalesced = 0.0f, max_diff_tensor = 0.0f;
    for(int i = 0; i < input_size; i++) {
        float diff_tiled = fabsf(data->h_output_naive[i] - data->h_output_tiled[i]);
        float diff_coalesced = fabsf(data->h_output_naive[i] - data->h_output_coalesced[i]);
        float diff_tensor = fabsf(data->h_output_naive[i] - data->h_output_tensor[i]);
        if(diff_tiled > max_diff_tiled) max_diff_tiled = diff_tiled;
        if(diff_coalesced > max_diff_coalesced) max_diff_coalesced = diff_coalesced;
        if(diff_tensor > max_diff_tensor) max_diff_tensor = diff_tensor;
    }
    printf("Tiled vs Naive max diff: %.6f\n", max_diff_tiled);
    printf("Coalesced vs Naive max diff: %.6f\n", max_diff_coalesced);
    printf("Tensor Core vs Naive max diff: %.6f\n", max_diff_tensor);
}

void cleanup_data(BenchmarkData* data) {
    // Cleanup host memory
    free(data->h_x); free(data->h_Wq); free(data->h_Wk); free(data->h_Wv); free(data->h_Wo);
    free(data->h_output_naive); free(data->h_output_tiled); free(data->h_output_coalesced); free(data->h_output_tensor);
    
    // Cleanup device memory
    cudaFree(data->d_x); cudaFree(data->d_Wq); cudaFree(data->d_Wk); cudaFree(data->d_Wv); cudaFree(data->d_Wo);
    cudaFree(data->d_output_naive); cudaFree(data->d_output_tiled); cudaFree(data->d_output_coalesced); cudaFree(data->d_output_tensor);
    
    // Cleanup events
    cudaEventDestroy(data->start);
    cudaEventDestroy(data->stop);
}

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
    
    BenchmarkData data;
    initialize_data(&data, seq_len, d_model);
    
    printf("\n=== Attention Implementation Benchmark ===\n");
    
    // Benchmark all implementations
    float naive_time = benchmark_implementation(&data, "Naive", self_attention_naive_cuda, data.d_output_naive, seq_len, d_model);
    float tiled_time = benchmark_implementation(&data, "Tiled", self_attention_tiled_cuda, data.d_output_tiled, seq_len, d_model);
    float coalesced_time = benchmark_implementation(&data, "Coalesced B", self_attention_coalesced_b_cuda, data.d_output_coalesced, seq_len, d_model);
    float tensor_time = benchmark_tensor_core(&data, seq_len, d_model);
    
    print_speedups(naive_time, tiled_time, coalesced_time, tensor_time);
    verify_correctness(&data, seq_len, d_model);
    cleanup_data(&data);
    
    return 0;
}