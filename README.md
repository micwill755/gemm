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
| 8192    | 6144      | 1761.8     | 886.0      | 1286.8          | 268.0           | 337.3          | **6.57x**     |

### Speedup Analysis

| Implementation      | 8192x6144 Speedup |
|---------------------|-------------------|
| Tiled               | **1.99x**         |
| Coalesced B         | **1.37x**         |
| Tensor Core         | **6.57x**         |
| Tensor Core Shared  | **5.22x**         |

### Key Observations

1. **Tensor Core dominance** - 6.57x speedup, best overall performance
2. **Shared memory overhead** - TC Shared (5.22x) slower than basic TC due to sync costs
3. **Perfect numerical accuracy** for Tiled and Coalesced B (0.000000 max difference)
4. **Tensor Core trade-off** - Significant speedup but reduced precision (max diff: 2163.6)
5. **Consistent tiled performance** - Reliable 2x speedup with perfect accuracy

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