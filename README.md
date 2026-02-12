# Self-Attention CUDA Implementation Benchmark

This benchmark compares multiple implementations of self-attention mechanism using optimized CUDA kernels to reduce global memory access and leverage modern GPU features.

## Implementation Details

- **Naive**: Basic CUDA kernels without shared memory optimization
- **Tiled**: Shared memory tiling for matrix multiplication, transpose, and softmax operations
- **Coalesced B**: Memory coalescing optimizations for improved bandwidth utilization
- **Tensor Core**: Leverages Tensor Core units for mixed-precision matrix operations
- **Architecture**: Complete self-attention pipeline with Q, K, V projections, attention scores, and output projection

## Benchmark Results

### Production-Scale Performance (DeepSeek-V3 Dimensions)

| Seq Len | Model Dim | Naive (ms) | Tiled (ms) | Coalesced B (ms) | Tensor Core (ms) | Best Speedup |
|---------|-----------|------------|------------|------------------|------------------|---------------|
| 2048    | 4096      | 163.7      | 79.2       | 116.9           | 34.9            | **4.69x**     |
| 4096    | 5120      | 542.8      | 264.7      | 391.4           | 100.2           | **5.41x**     |
| 8192    | 6144      | 1766.8     | 886.0      | 1287.5          | 266.7           | **6.62x**     |

### Speedup Analysis

| Implementation | 2048x4096 | 4096x5120 | 8192x6144 | Avg Speedup |
|----------------|-----------|-----------|-----------|-------------|
| Tiled          | 2.07x     | 2.05x     | 1.99x     | **2.04x**   |
| Coalesced B    | 1.40x     | 1.39x     | 1.37x     | **1.39x**   |
| Tensor Core    | 4.69x     | 5.41x     | 6.62x     | **5.57x**   |

### Key Observations

1. **Tensor Core dominance** - Up to 6.62x speedup with scaling performance
2. **Perfect numerical accuracy** for Tiled and Coalesced B (0.000000 max difference)
3. **Tensor Core trade-off** - Significant speedup but reduced precision (mixed-precision artifacts)
4. **Consistent tiled performance** - Reliable 2x speedup across all scales

## Usage

```bash
# Compile
nvcc -o attention_benchmark attention_benchmark.cu self_attention_naive.cu self_attention_tiled.cu ../matmul/matmul_*.cu ../transpose/transpose_*.cu ../softmax/softmax_*.cu -lcudart -lm

# Run benchmarks
./attention_benchmark 2048 4096   # Standard transformer size
./attention_benchmark 4096 5120   # Large model size  
./attention_benchmark 8192 6144   # Extended context size
```

## Optimization Techniques

- **Shared Memory Tiling**: Reduces global memory bandwidth requirements
- **Memory Coalescing**: Optimized memory access patterns for improved throughput
- **Bank Conflict Avoidance**: Padding in shared memory tiles
- **Tensor Core Utilization**: Mixed-precision operations for maximum performance
- **Kernel Fusion**: Minimized intermediate memory allocations

## Hardware Requirements

- CUDA-capable GPU with sufficient memory
- Minimum 2GB GPU memory for small tests
- 8GB+ GPU memory recommended for large-scale benchmarks