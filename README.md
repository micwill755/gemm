# Self-Attention CUDA Implementation Benchmark

This benchmark compares multiple implementations of self-attention mechanism using optimized CUDA kernels to reduce global memory access and leverage modern GPU features.

## Implementation Details

- **Naive**: Basic CUDA kernels without shared memory optimization
- **Tiled**: Shared memory tiling for matrix multiplication, transpose, and softmax operations
- **Coalesced B**: Memory coalescing optimizations for improved bandwidth utilization
- **Tensor Core**: Leverages Tensor Core units for mixed-precision matrix operations
- **Tensor Core Shared**: Tensor Cores with cooperative shared memory loading
- **Architecture**: Complete self-attention pipeline with Q, K, V projections, attention scores, and output projection

## Benchmark Results

### Production-Scale Performance (DeepSeek-V3 Dimensions)

| Seq Len | Model Dim | Naive (ms) | Tiled (ms) | Coalesced B (ms) | Tensor Core (ms) | TC Shared (ms) | Best Speedup |
|---------|-----------|------------|------------|------------------|------------------|----------------|---------------|
| 2048    | 4096      | 166.5      | 88.0       | 117.5           | 34.8            | 34.8           | **4.78x**     |
| 4096    | 5120      | 534.4      | 264.6      | 391.1           | 100.8           | 113.9          | **5.30x**     |
| 8192    | 6144      | 1752.8     | 885.7      | 1287.0          | 270.8           | 270.8          | **6.47x**     |

### Speedup Analysis

| Implementation      | 2048x4096 | 4096x5120 | 8192x6144 | Avg Speedup |
|---------------------|-----------|-----------|-----------|-------------|
| Tiled               | 1.89x     | 2.02x     | 1.98x     | **1.96x**   |
| Coalesced B         | 1.42x     | 1.37x     | 1.36x     | **1.38x**   |
| Tensor Core         | 4.78x     | 5.30x     | 6.47x     | **5.52x**   |
| Tensor Core Shared  | 4.78x     | 4.69x     | 6.47x     | **5.31x**   |

### Key Observations

1. **Tensor Core dominance** - Up to 6.47x speedup, best overall performance
2. **TC Shared performance** - Matches basic TC at small/large scales, 11% slower at mid-scale
3. **Perfect numerical accuracy** for Tiled and Coalesced B (0.000000 max difference)
4. **Tensor Core accuracy** - Variable precision loss (0.0 to 1704.2 max diff) depending on scale
5. **Consistent tiled performance** - Reliable ~2x speedup with perfect accuracy

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