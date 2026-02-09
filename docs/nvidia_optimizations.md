# NVIDIA CUDA Matrix Multiplication Optimizations

## Overview

This guide covers progressive optimization techniques for CUDA matrix multiplication, from naive implementations to production-level optimizations used in libraries like cuBLAS.

## 1. Naive Implementation Issues

### Memory Access Pattern
```cuda
for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
}
```

**Problems:**
- **Global memory latency**: ~400-800 cycles per access
- **No data reuse**: Each thread reads the same A/B values independently
- **Memory bandwidth bound**: GPU compute units idle waiting for data

## 2. Shared Memory Tiling

### Concept
Load tiles of A and B into fast shared memory, reuse across threads in a block.

### Benefits
- **16x reduction** in global memory accesses (for 16×16 tiles)
- **Data reuse**: All threads in block share the same tile
- **Bandwidth optimization**: Coalesced memory access patterns

### Implementation Pattern
```cuda
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// Load tile collaboratively
As[ty][tx] = A[row * K + tile_start + tx];
Bs[ty][tx] = B[(tile_start + ty) * N + col];
__syncthreads();

// Compute using shared memory
for (int k = 0; k < TILE_SIZE; k++) {
    sum += As[ty][k] * Bs[k][tx];
}
```

## 3. Memory Coalescing

### Global Memory Access Patterns
**Coalesced (Good)**: Consecutive threads access consecutive memory addresses
```
Thread 0: A[0], Thread 1: A[1], Thread 2: A[2], ...
```

**Uncoalesced (Bad)**: Random or strided access patterns
```
Thread 0: A[0], Thread 1: A[128], Thread 2: A[256], ...
```

### Optimization Techniques
- **Transpose B matrix** to improve access patterns
- **Padding** to avoid bank conflicts
- **Vectorized loads** (float2, float4) when possible

## 4. Occupancy Optimization

### Thread Block Configuration
```cuda
// Balance between parallelism and resource usage
dim3 blockDim(16, 16);  // 256 threads per block
dim3 gridDim((N + 15) / 16, (M + 15) / 16);
```

### Resource Considerations
- **Registers per thread**: Limit to maximize occupancy
- **Shared memory usage**: Balance tile size vs blocks per SM
- **Warp utilization**: Ensure full 32-thread warps

### Occupancy Calculator
```bash
# Use NVIDIA's occupancy calculator
nvcc --ptxas-options=-v kernel.cu
# Check register usage and shared memory
```

## 5. Tensor Core Optimization (Modern GPUs)

### Mixed Precision (FP16/BF16)
```cuda
// Use wmma (Warp Matrix Multiply Accumulate)
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, A, K);
wmma::load_matrix_sync(b_frag, B, N);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
```

### Benefits
- **Up to 10x speedup** on V100/A100/H100
- **Higher throughput**: 125 TFLOPS (FP16) vs 15 TFLOPS (FP32)
- **Energy efficient**: Lower precision = less power

## 6. Advanced Optimizations

### Double Buffering
```cuda
// Overlap computation with memory transfers
__shared__ float As[2][TILE_SIZE][TILE_SIZE];
__shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

int write_idx = 0, read_idx = 1;
// Load next tile while computing current tile
```

### Prefetching
```cuda
// Load data before it's needed
float reg_A = A[next_index];  // Prefetch into register
// Use reg_A in next iteration
```

### Loop Unrolling
```cuda
#pragma unroll
for (int k = 0; k < TILE_SIZE; k++) {
    sum += As[ty][k] * Bs[k][tx];
}
```

## 7. Memory Layout Optimizations

### Matrix Transposition
```cuda
// Transpose B for better cache locality
// B^T allows coalesced access in inner loop
```

### Padding for Bank Conflicts
```cuda
// Add padding to avoid shared memory bank conflicts
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
```

### Data Layout Transformation
- **NCHW → NHWC**: Better for convolutions
- **AoS → SoA**: Array of Structures → Structure of Arrays
- **Blocked layouts**: Improve cache locality

## 8. Kernel Fusion

### Combine Operations
```cuda
// Instead of separate kernels for C = α(A×B) + βC
__global__ void gemm_fused(float* A, float* B, float* C, 
                          float alpha, float beta) {
    // Compute A×B and apply α, β in single kernel
    float result = alpha * dot_product + beta * C[idx];
    C[idx] = result;
}
```

### Benefits
- **Reduced memory traffic**: Avoid intermediate results
- **Lower kernel launch overhead**
- **Better cache utilization**

## 9. Multi-GPU Scaling

### Data Parallelism
```cuda
// Split matrices across GPUs
// GPU 0: rows 0-511, GPU 1: rows 512-1023
cudaSetDevice(gpu_id);
matmul_kernel<<<grid, block>>>(A_local, B, C_local);
```

### Model Parallelism
```cuda
// Split computation across dimensions
// GPU 0: A×B[:, 0:N/2], GPU 1: A×B[:, N/2:N]
```

## 10. Performance Profiling

### NVIDIA Profiler Tools
```bash
# Nsight Compute - detailed kernel analysis
ncu --set full ./matmul

# Nsight Systems - system-wide profiling  
nsys profile ./matmul

# Key metrics to monitor:
# - Memory throughput (GB/s)
# - Compute utilization (%)
# - Occupancy (%)
# - Warp efficiency (%)
```

### Performance Metrics
```cuda
// Theoretical peak performance
float peak_gflops = cores * base_clock * 2;  // 2 ops per cycle (FMA)

// Memory bandwidth utilization
float bandwidth_util = (bytes_transferred / time) / peak_bandwidth;

// Arithmetic intensity
float ai = flops / bytes_transferred;
```

## 11. Production-Level Techniques

### Autotuning
```cuda
// Automatically find optimal tile sizes
for (int tile = 8; tile <= 32; tile += 8) {
    benchmark_kernel(tile);
}
// Select best performing configuration
```

### Template Specialization
```cuda
template<int TILE_SIZE, int BLOCK_SIZE>
__global__ void matmul_specialized() {
    // Compile-time constants enable better optimization
}
```

### Assembly-Level Optimization
```cuda
// Use PTX assembly for critical paths
asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
```

## 12. Library Integration

### cuBLAS Usage
```cuda
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// High-performance GEMM
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
           M, N, K, &alpha, A, M, B, K, &beta, C, M);
```

### cuBLASLt (Advanced)
```cuda
// More flexible, supports custom epilogues
cublasLtMatmul(ltHandle, matmulDesc, &alpha, A, Adesc, B, Bdesc,
               &beta, C, Cdesc, C, Cdesc, &algo, workspace, workspaceSize, stream);
```

## Performance Comparison

| Optimization Level | GFLOPS | Speedup | Key Technique |
|-------------------|--------|---------|---------------|
| Naive | 3,023 | 1.0x | Basic implementation |
| Shared Memory | 4,499 | 1.49x | Tiling + data reuse |
| Coalescing | 6,500 | 2.15x | Memory access optimization |
| Tensor Cores | 15,000 | 4.96x | Mixed precision |
| cuBLAS | 25,000+ | 8.27x+ | Production optimizations |

## Best Practices Summary

1. **Start with profiling** - Identify bottlenecks first
2. **Optimize memory access** - Coalescing and shared memory
3. **Maximize occupancy** - Balance resources per thread
4. **Use Tensor Cores** - For supported data types
5. **Consider cuBLAS** - For production workloads
6. **Profile iteratively** - Measure impact of each optimization

## Hardware-Specific Considerations

### Ampere (A100/RTX 30xx)
- **Tensor Cores**: BF16, TF32 support
- **Multi-Instance GPU**: Partition for isolation
- **Async copy**: Overlap compute and memory

### Hopper (H100)
- **Transformer Engine**: FP8 precision
- **Thread Block Clusters**: Cross-SM cooperation
- **DPX instructions**: Dynamic precision

### Ada Lovelace (RTX 40xx)
- **Ada Tensor Cores**: Improved sparsity support
- **AV1 encoding**: For visualization workloads