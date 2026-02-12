# Tensor Core Tiling: Q×K^T Example

## Problem
```
Q: [2048, 4096] × K^T: [4096, 2048] = Result: [2048, 2048]
Tensor Core constraint: 4×4×4 operations only
```

## Solution: Decompose into 4×4 tiles

### Step 1: Output Tiling
```
Result [2048×2048] = 512×512 grid of 4×4 tiles
Total output tiles: 262,144
```

### Step 2: K-dimension Chunking
For **one** 4×4 output tile:
```
C[4×4] = Σ(k=0 to 4095) Q[4×k] × K^T[k×4]

K dimension chunks: 4096 ÷ 4 = 1024 chunks
Operations per tile: 1024 Tensor Core ops
```

### Step 3: Visual Breakdown
```
Q matrix for one 4×4 output tile:
┌─────────────────────────────────────────┐
│ q00 q01 q02 q03 │ q04 q05 q06 q07 │ ... │ ← 4096 elements
│ q10 q11 q12 q13 │ q14 q15 q16 q17 │ ... │
│ q20 q21 q22 q23 │ q24 q25 q26 q27 │ ... │
│ q30 q31 q32 q33 │ q34 q35 q36 q37 │ ... │
└─────────────────────────────────────────┘
  Chunk 0 (4×4)     Chunk 1 (4×4)    ...
  
K^T matrix for same tile:
┌─────────────────────────────────────────┐
│ k00 k10 k20 k30 │
│ k01 k11 k21 k31 │
│ k02 k12 k22 k32 │
│ k03 k13 k23 k33 │
├─────────────────┤
│ k04 k14 k24 k34 │ ← Chunk 1
│ k05 k15 k25 k35 │
│ k06 k16 k26 k36 │
│ k07 k17 k27 k37 │
├─────────────────┤
│      ...        │
└─────────────────┘

Tensor Core Operations:
Chunk 0: Q[4×4] × K^T[4×4] → partial_result_0
Chunk 1: Q[4×4] × K^T[4×4] → partial_result_1
...
Chunk 1023: Q[4×4] × K^T[4×4] → partial_result_1023

Final: sum all partial results = 1024 Tensor Core ops
```

### Step 4: Matrix Tiling Overview
```
Complete Q×K^T decomposition:

Q [2048×4096] tiled as:
┌─────┬─────┬─────┬─────┐
│[4×4]│[4×4]│ ... │[4×4]│ ← 512 rows of tiles
├─────┼─────┼─────┼─────┤
│[4×4]│[4×4]│ ... │[4×4]│
├─────┼─────┼─────┼─────┤
│ ... │ ... │ ... │ ... │
├─────┼─────┼─────┼─────┤
│[4×4]│[4×4]│ ... │[4×4]│
└─────┴─────┴─────┴─────┘
      1024 columns of tiles

Result [2048×2048] tiled as:
┌─────┬─────┬─────┬─────┐
│[4×4]│[4×4]│ ... │[4×4]│ ← 512×512 = 262,144 tiles
├─────┼─────┼─────┼─────┤
│[4×4]│[4×4]│ ... │[4×4]│
├─────┼─────┼─────┼─────┤
│ ... │ ... │ ... │ ... │
└─────┴─────┴─────┴─────┘
```

### Step 5: Complete Calculation
```
Per 4×4 tile: 1024 Tensor Core ops
Total tiles: 512×512 = 262,144 tiles  
Total operations: 262,144 × 1024 = 268M Tensor Core ops
```

## Code Pattern
```cuda
for (int k_chunk = 0; k_chunk < 1024; k_chunk++) {
    Q_tile = Q[tile_row, k_chunk*4:(k_chunk+1)*4];
    K_tile = K^T[k_chunk*4:(k_chunk+1)*4, tile_col];
    
    result += Q_tile × K_tile;  // One 4×4×4 Tensor Core op
}
```

**Key insight**: Each 4×4 output tile requires walking through the entire K-dimension (4096) in chunks of 4, needing 4096÷4 = 1024 separate Tensor Core operations.