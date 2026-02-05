# Parallel Tree Reduction

## What is Tree Reduction?

**Tree reduction** is a parallel algorithm that combines many values into one result using a binary tree pattern. Instead of processing sequentially, multiple threads work simultaneously at each level.

## Sequential vs Tree Reduction

**❌ Sequential Reduction:**
```
Thread 0: result = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7
Time: 7 operations, 1 thread working
```

**✅ Tree Reduction:**
```
Level 1: 4 threads work in parallel
Level 2: 2 threads work in parallel  
Level 3: 1 thread works
Time: 3 levels, multiple threads per level
```

## Visual Tree Structure

**8-Element Example:**
```
Initial:  [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]
           |    |    |    |    |    |    |    |
Level 1:   \   /     \   /     \   /     \   /
           [a0+a1]  [a2+a3]  [a4+a5]  [a6+a7]
             |        |        |        |
Level 2:     \       /         \       /
              \     /           \     /
            [a0+a1+a2+a3]   [a4+a5+a6+a7]
                  |               |
Level 3:          \               /
                   \             /
                [a0+a1+a2+a3+a4+a5+a6+a7]
```

## Step-by-Step Execution

**Initial State:**
```
sdata: [3] [7] [2] [9] [1] [5] [8] [4]
Index:  0   1   2   3   4   5   6   7
```

**Step 1: s = 4 (stride = 4)**
```
Thread 0: sdata[0] = max(sdata[0], sdata[4]) = max(3, 1) = 3
Thread 1: sdata[1] = max(sdata[1], sdata[5]) = max(7, 5) = 7  
Thread 2: sdata[2] = max(sdata[2], sdata[6]) = max(2, 8) = 8
Thread 3: sdata[3] = max(sdata[3], sdata[7]) = max(9, 4) = 9

Result: [3] [7] [8] [9] [1] [5] [8] [4]
```

**Step 2: s = 2 (stride = 2)**
```
Thread 0: sdata[0] = max(sdata[0], sdata[2]) = max(3, 8) = 8
Thread 1: sdata[1] = max(sdata[1], sdata[3]) = max(7, 9) = 9

Result: [8] [9] [8] [9] [1] [5] [8] [4]
```

**Step 3: s = 1 (stride = 1)**
```
Thread 0: sdata[0] = max(sdata[0], sdata[1]) = max(8, 9) = 9

Final Result: [9] [9] [8] [9] [1] [5] [8] [4]
              ↑
         Final answer
```

## GPU Implementation Pattern

```cuda
// Tree reduction loop
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] = operation(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
}
```

**Loop iterations for TILE_SIZE = 32:**
- s = 16: threads 0-15 work
- s = 8:  threads 0-7 work  
- s = 4:  threads 0-3 work
- s = 2:  threads 0-1 work
- s = 1:  thread 0 works

## Thread Activity Diagram

```
Threads:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
Step 1:   ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  (16 active)
Step 2:   ●  ●  ●  ●  ●  ●  ●  ●  ○  ○  ○  ○  ○  ○  ○  ○  (8 active)
Step 3:   ●  ●  ●  ●  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  (4 active)
Step 4:   ●  ●  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  (2 active)
Step 5:   ●  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  ○  (1 active)

● = Active thread    ○ = Idle thread
```

## Complexity Analysis

| Method | Time Complexity | Parallelism |
|--------|----------------|-------------|
| **Sequential** | O(n) | 1 thread |
| **Tree Reduction** | O(log n) | n/2 → n/4 → ... → 1 threads |

**For 32 elements:**
- Sequential: 31 operations
- Tree: 5 steps (log₂(32) = 5)

## Common Operations

```cuda
// Sum reduction
sdata[tid] += sdata[tid + s];

// Max reduction  
sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);

// Min reduction
sdata[tid] = fminf(sdata[tid], sdata[tid + s]);

// Product reduction
sdata[tid] *= sdata[tid + s];
```

## Key Benefits

1. **Logarithmic time complexity** - O(log n) vs O(n)
2. **High parallelism** - Multiple threads work simultaneously
3. **Memory efficient** - Uses shared memory for fast access
4. **Scalable** - Works for any power-of-2 block size
5. **Hardware optimized** - Matches GPU warp execution model

## Applications

- **Softmax normalization** - Finding row max and sum
- **Matrix operations** - Norms, dot products
- **Statistics** - Mean, variance calculations
- **Image processing** - Histogram computation
- **Scientific computing** - Parallel aggregations