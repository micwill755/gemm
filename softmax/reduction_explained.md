# GPU Reduction Operations

## What is Reduction?

**Reduction** combines many values into a single result using an associative operation (sum, max, min, etc.).

```
Many Values → Single Value
[1, 5, 3, 8, 2, 7, 4, 6] → 36 (sum)
[1, 5, 3, 8, 2, 7, 4, 6] → 8  (max)
```

## Sequential vs Parallel

**❌ Sequential (CPU):**
```c
float sum = 0;
for (int i = 0; i < n; i++) {
    sum += array[i];  // One operation at a time
}
// Time: O(n)
```

**✅ Parallel (GPU):**
```
Step 0: [1, 5, 3, 8, 2, 7, 4, 6]  ← 8 values, 4 threads work

Step 1: [1+5, 3+8, 2+7, 4+6]      ← 4 values, 2 threads work
        [ 6 ,  11,   9,  10]

Step 2: [6+11, 9+10]               ← 2 values, 1 thread works  
        [ 17,   19]

Step 3: [17+19]                    ← 1 value (final result)
        [ 36 ]
```
**Time: O(log n)**

## Tree Reduction Pattern

```
Level 0:  T0   T1   T2   T3   T4   T5   T6   T7
          [1]  [5]  [3]  [8]  [2]  [7]  [4]  [6]
           |    |    |    |    |    |    |    |
Level 1:   \   /     \   /     \   /     \   /
            [6]       [11]      [9]       [10]
             |         |        |         |
Level 2:      \       /          \       /
                [17]              [19]
                 |                 |
Level 3:          \               /
                      [36]
```

## GPU Implementation

```cuda
__global__ void sum_reduction(float *input, float *output, int n) {
    __shared__ float sdata[256];  // Shared memory for block
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

## Common Operations

| Operation | Code | Use Case |
|-----------|------|----------|
| **Sum** | `sdata[tid] += sdata[tid + s]` | Total, average |
| **Max** | `sdata[tid] = fmaxf(sdata[tid], sdata[tid + s])` | Maximum value |
| **Min** | `sdata[tid] = fminf(sdata[tid], sdata[tid + s])` | Minimum value |

## Key Benefits

- **Logarithmic complexity:** O(log n) vs O(n)
- **High parallelism:** All threads work simultaneously  
- **Memory efficient:** Uses fast shared memory
- **Scalable:** Works with any block size