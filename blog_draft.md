Matrix multiplication computes the dot product of each row from the first matrix with each column from the second matrix.

More specifically, for matrices A (m×k) and B (k×n), the result C (m×n) is computed as:

C[i][j] = dot_product(row i of A, column j of B)

Standard notation - matmul = (m*k) * (k*n)

Here's the computation:

A (2×3):

[1, 2, 3]
[2, 3, 4]

B (3×2):

[4, 5]
[6, 3]
[2, 2]

C (2×2) = A × B:

C[0,0] = row 0 of A · col 0 of B = (1×4) + (2×6) + (3×2) = 4 + 12 + 6 = 22
C[0,1] = row 0 of A · col 1 of B = (1×5) + (2×3) + (3×2) = 5 + 6 + 6 = 17
C[1,0] = row 1 of A · col 0 of B = (2×4) + (3×6) + (4×2) = 8 + 18 + 8 = 34
C[1,1] = row 1 of A · col 1 of B = (2×5) + (3×3) + (4×2) = 10 + 9 + 8 = 27

Result:

C = [[22, 17],
     [34, 27]]

Running a matmul function in C:

// C = A @ B, where A is MxK, B is KxN, C is MxN
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[i*K + k] * B[k*N + j];  
        }
        C[i*N + j] = sum;
    }
}

If we run a matmul where m = 4, p = 3, n = 4, our result matrix c will be (16, 16) how do we translate this to a naive matmul kernel for a GPU using CUDA. Lets first review the GPU hierarchy

GPU Execution Hierarchy (top-down):

Grid - the entire kernel launch
- You define: gridDim (how many blocks in x, y, z dimensions)

Blocks - subdivisions of the grid
- You define: blockDim (how many threads per block in x, y, z)
- Blocks are assigned to SMs by the GPU scheduler

Threads - individual execution units within a block
- All threads in a block run on the same SM
- Threads are grouped into warps (32 threads each)

Warps - hardware scheduling unit
- GPU automatically groups 32 consecutive threads into a warp
- Warps execute in SIMT (Single Instruction, Multiple Threads)

SM (Streaming Multiprocessor) - physical hardware
- You don't control this - the GPU assigns blocks to available SMs
- Each SM can run multiple blocks concurrently (if resources allow)

You control grid/block dimensions. GPU controls SM assignment and warp scheduling.
     
# Understanding GPU Matrix Multiplication: Naive to Optimized

## What is GEMM?

GEMM stands for **General Matrix Multiply** and it's the operation:

```
C = α(A × B) + βC
```

Where:
- A, B, C are matrices
- α (alpha) and β (beta) are scalar multipliers
- × is matrix multiplication

For standard matrix multiplication `A @ B`, you'd use α=1, β=0:
```
C = 1×(A × B) + 0×C  →  C = A × B
```

### Why Use Different α and β Values?

Different α and β values are useful for several practical scenarios:

**1. β=1 (Accumulation)**

Use case: Gradient accumulation in training, multi-head attention combining heads

```python
# Instead of: C = A1@B1 + A2@B2 + A3@B3
C = zeros()
gemm(A1, B1, C, α=1, β=0)  # C = A1@B1
gemm(A2, B2, C, α=1, β=1)  # C += A2@B2
gemm(A3, B3, C, α=1, β=1)  # C += A3@B3
```

**2. α≠1 (Scaling)**

Use cases:
- Attention: `scores = (Q @ K.T) / sqrt(d_k)` → use α=1/sqrt(d_k)
- Learning rate scaling: `update = lr * (grad @ weights)`
- Residual connections: `out = x + 0.1 * (A @ B)` → use α=0.1, β=1

```python
# Instead of: C = 0.5 * (A @ B)  # two operations
gemm(A, B, C, α=0.5, β=0)  # one fused operation
```

**Why it matters:** Doing it in one GEMM call is much faster because:
- Avoids extra memory reads/writes
- Better cache utilization
- Single kernel launch (GPU)
- Hardware can optimize the fused operation

### Optimized BLAS Libraries

GEMM is heavily optimized in libraries like:
- **cuBLAS** (CUDA)
- **cuBLASLt** (CUDA with more flexibility)
- **MKL** (Intel)
- **OpenBLAS**

---

## GPU Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                          GPU CHIP                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Streaming Multiprocessor (SM)             │ │
│  │                                                        │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │              Thread Block                        │ │ │
│  │  │                                                  │ │ │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │ │ │
│  │  │  │ Thread 0│  │ Thread 1│  │ Thread 2│  ...    │ │ │
│  │  │  │         │  │         │  │         │         │ │ │
│  │  │  │ row=0   │  │ row=0   │  │ row=0   │         │ │ │
│  │  │  │ col=0   │  │ col=1   │  │ col=2   │         │ │ │
│  │  │  │ sum=0.0 │  │ sum=0.0 │  │ sum=0.0 │         │ │ │
│  │  │  │ k=0     │  │ k=0     │  │ k=0     │         │ │ │
│  │  │  └─────────┘  └─────────┘  └─────────┘         │ │ │
│  │  │       ↑            ↑            ↑               │ │ │
│  │  │  REGISTERS    REGISTERS    REGISTERS            │ │ │
│  │  │  (Private)    (Private)    (Private)            │ │ │
│  │  │  ~64KB per SM, divided among threads            │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  │                        ↕                              │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │         Shared Memory (__shared__)               │ │ │
│  │  │  Accessible by ALL threads in this block         │ │ │
│  │  │  ~48-96 KB per SM                                │ │ │
│  │  │  Programmer-managed cache                        │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  │                        ↕                              │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │         L1 Cache (automatic)                     │ │ │
│  │  │         ~128 KB per SM                           │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                          ↕                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         L2 Cache (shared across all SMs)              │ │
│  │         ~4-6 MB                                       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────────┐
│              Global Memory (DRAM)                            │
│              A, B, C arrays live here                        │
│              ~8-80 GB, slow (~400-800 cycles)                │
└─────────────────────────────────────────────────────────────┘
```

### Key Memory Types

**Registers:**
- Private to each thread - NOT shared
- Fastest memory (~1 cycle latency)
- Variables like `sum`, `row`, `col`, `k` live here
- Limited: ~64KB per SM divided among all threads
- If you use too many registers per thread, fewer threads can run

**Shared Memory:**
- Shared by all threads in a block
- Fast (~5-20 cycles)
- Declared with `__shared__`
- Must be explicitly managed by programmer

**Global Memory:**
- Shared by entire GPU
- Slow (~400-800 cycles)
- Where A, B, C live

---

## Naive CUDA Implementation

### The Kernel

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
    
if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

### Grid/Block Configuration

For a 16×16 result matrix C (M=16, N=16):

- `blockDim = (16, 16)` → each block has 16×16 = **256 threads**
- `gridDim = ((16+15)/16, (16+15)/16) = (1, 1)` → **1 block total**

**Thread mapping:**
```
Thread (0,0) → C[0][0]
Thread (0,1) → C[0][1]
...
Thread (15,15) → C[15][15]
```

**Warp organization** (warps execute 32 threads in lockstep):

> **Lockstep** means all 32 threads in a warp execute the same instruction at the same time on different data.

- Warp 0: threads (0,0) through (1,15) - first 2 rows
- Warp 1: threads (2,0) through (3,15) - next 2 rows
- ...
- Warp 7: threads (14,0) through (15,15) - last 2 rows

### Why Naive is Slow

The problem is in the kernel loop:

```cuda
for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];  // ← Global memory access EVERY iteration
}
```

**For K=1024:**
- Each thread reads from global memory **2048 times** (1024 from A, 1024 from B)
- Global memory latency: ~400-800 cycles
- **No data reuse between threads**
  - Thread 0 reads A[0][0], A[0][1], ..., A[0][1023]
  - Thread 1 reads A[0][0], A[0][1], ..., A[0][1023] ← **same data!**
  - Threads in the same row all read the same A values
  - Threads in the same column all read the same B values

---

## Optimized: Shared Memory Tiling

### The Solution

**Load tiles of A and B into shared memory (fast, on-chip)**
- All threads in a block reuse the same tile
- Reduces global memory accesses by ~16x (for 16×16 tiles)

### Performance Results

```bash
$ ./matmul_tiled 
Result[0] = 1024.00 (expected 1024.00)
Time: 0.477 ms
GFLOPS: 4498.82

$ ./matmul_naive  
Result[0] = 1024.00 (expected 1024.00)
Time: 0.710 ms
GFLOPS: 3022.92
```

**1.49x speedup!**

### Understanding GFLOPS

GFLOPS measures **throughput, not work done**.

Both versions do the same amount of work: `2 × M × N × K = 2.15 billion floating-point operations`

```
GFLOPS = Operations / Time

Naive:  2.15 billion ops / 0.710 ms = 3023 GFLOPS
Tiled:  2.15 billion ops / 0.477 ms = 4499 GFLOPS
```

**Higher GFLOPS = Better performance** (more operations per second)

### Why Tiled is Faster

- Spends less time waiting for memory
- More time actually computing (multiply-add operations)
- Better utilization of GPU compute units

**The goal:** Maximize GFLOPS by minimizing memory bottlenecks. The tiled version keeps the GPU's arithmetic units busy instead of stalled waiting for data from global memory.

---

## Memory Layout: Row-Major vs Column-Major

**Row-major:** Consecutive elements of a row are contiguous in memory (C/C++ default)

**Column-major:** Consecutive elements of a column are contiguous (Fortran/BLAS default)

This affects how you call GEMM and whether you need to transpose matrices.

---

## Matrix Multiplication Fundamentals

Matrix multiplication computes the **dot product** of each row from the first matrix with each column from the second matrix.

For matrices A (m×n) and B (n×p), the result C (m×p) is:

```
C[i][j] = dot_product(row i of A, column j of B)
```