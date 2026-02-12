# Tensor Core Mixed Precision: FP16 Input, FP32 Accumulate

## The Design
```
Tensor Core Operation: A[4×4, FP16] × B[4×4, FP16] → C[4×4, FP32]

Input matrices: Half precision (16-bit)
Output matrix: Single precision (32-bit)
```

## Why Mixed Precision?

### 1. Memory Bandwidth Optimization
```
Memory per 4×4 tile:
├── FP32 only: 16×4 + 16×4 = 128 bytes input
├── FP16 input: 16×2 + 16×2 = 64 bytes input  
└── Bandwidth savings: 50% reduction
```

### 2. Precision Requirements
```
FP16 characteristics:
├── Range: ±65,504
├── Precision: ~3-4 decimal digits
└── Sufficient for: weights, activations, gradients

FP32 characteristics:  
├── Range: ±3.4×10³⁸
├── Precision: ~7 decimal digits
└── Required for: accumulation, final results
```

### 3. The Accumulation Problem
```
Pure FP16 computation:
dot_product = Σ(aᵢ × bᵢ)  // i=1 to N

Each step: FP16 × FP16 → FP16 → FP16 + FP16 → FP16
Problem: Precision loss compounds with each addition
Result: Catastrophic error for large N
```

### 4. Mixed Precision Solution
```
Tensor Core computation:
Each multiply: FP16 × FP16 → FP32 (automatic promotion)
Each add: FP32 + FP32 → FP32 (maintained precision)
Result: No accumulation error
```

---

## Hardware Implementation

### Tensor Core Pipeline
```
Stage 1: Load FP16 Fragments
┌─────────────────────────────────┐
│ A_frag: [16 FP16 values]        │
│ B_frag: [16 FP16 values]        │  
└─────────────────────────────────┘

Stage 2: Parallel Multiplication  
┌─────────────────────────────────┐
│ 16 × (FP16 × FP16 → FP32)       │
│ Hardware promotion to FP32      │
└─────────────────────────────────┘

Stage 3: Tree Reduction
┌─────────────────────────────────┐
│ FP32 + FP32 → FP32 (no loss)    │
│ Accumulate into existing C_frag │
└─────────────────────────────────┘

Stage 4: Store Result
┌─────────────────────────────────┐
│ C_frag: [16 FP32 values]        │
└─────────────────────────────────┘
```

---

## Performance Benefits

### Memory Efficiency
```
Attention computation example:
Q: [2048×4096] K: [2048×4096] → Scores: [2048×2048]

FP32 storage:
├── Q matrix: 2048×4096×4 = 32MB
├── K matrix: 2048×4096×4 = 32MB  
├── Scores: 2048×2048×4 = 16MB
└── Total: 80MB

Mixed precision storage:
├── Q matrix: 2048×4096×2 = 16MB (50% less)
├── K matrix: 2048×4096×2 = 16MB (50% less)
├── Scores: 2048×2048×4 = 16MB (same precision)
└── Total: 48MB (40% reduction)
```

### Cache Utilization
```
GPU L2 cache: 40MB (A100)

FP32 matrices: Fits 2048×2560 elements
FP16 matrices: Fits 2048×5120 elements (2x capacity!)

Result: Higher cache hit rates, lower memory latency
```

### Training Workflow
```
Forward Pass:
├── Weights: FP16 (memory efficient)
├── Activations: FP16 (bandwidth efficient)  
└── Computations: FP16→FP32 (accurate)

Backward Pass:
├── Gradients: FP32 (precise accumulation)
├── Weight updates: FP32 (numerical stability)
└── Loss scaling: Prevent FP16 underflow
```

---

## Code Example

### WMMA API Usage
```cuda
#include <mma.h>
using namespace nvcuda;

// Declare fragments
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// Load FP16 inputs
wmma::load_matrix_sync(a_frag, A_fp16, lda);
wmma::load_matrix_sync(b_frag, B_fp16, ldb);

// Mixed precision computation: FP16 × FP16 → FP32
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store FP32 result  
wmma::store_matrix_sync(C_fp32, c_frag, ldc, wmma::mem_row_major);
```

### Conversion Utilities
```cuda
// Convert for Tensor Core input
__global__ void fp32_to_fp16_kernel(const float* input, half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

// Convert from Tensor Core output (if needed)
__global__ void fp16_to_fp32_kernel(const half* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}
```

---

## Numerical Stability

### Error Analysis
```
Pure FP16 dot product (N=1024):
├── Single multiply error: ~10⁻⁴
├── Accumulated error: ~10⁻¹ (catastrophic!)
└── Result: Unusable for training

Mixed precision dot product:
├── Single multiply error: ~10⁻⁴  
├── FP32 accumulation error: ~10⁻⁷
└── Result: Numerically stable
```

### Loss Scaling
```cuda
// Prevent FP16 gradient underflow
float loss_scale = 1024.0f;

// Scale loss before backward pass
scaled_loss = loss * loss_scale;

// Unscale gradients after accumulation  
gradients = gradients / loss_scale;
```

---

## Key Advantages

1. **Speed**: 50% memory bandwidth reduction
2. **Accuracy**: FP32 accumulation prevents error buildup  
3. **Capacity**: 2x more data fits in cache/memory
4. **Compatibility**: Seamless integration with existing FP32 workflows
5. **Hardware optimized**: Designed specifically for AI workloads

**Bottom line**: Mixed precision gives you the speed of FP16 with the accuracy of FP32 - the optimal balance for modern AI training and inference.