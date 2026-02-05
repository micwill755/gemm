# GPU Shared Memory Bank Conflicts

## What Are Banks?

GPU shared memory is organized into **32 banks** that can each serve **one request per clock cycle**.

```
Bank:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Addr:  0  4  8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76 80 84 88 92 96100104108112116120124
```

**Bank mapping:** `bank_id = (byte_address / 4) % 32`

## Bank Conflicts Explained

**✅ No Conflict (Good):**
```
Thread 0: accesses bank 0
Thread 1: accesses bank 1  
Thread 2: accesses bank 2
...
Thread 31: accesses bank 31
```
All threads access different banks → **1 clock cycle**

**❌ Bank Conflict (Bad):**
```
Thread 0: accesses bank 0
Thread 1: accesses bank 0  ← Same bank!
Thread 2: accesses bank 0  ← Same bank!
...
Thread 31: accesses bank 0 ← Same bank!
```
All threads access same bank → **32 clock cycles** (serialized)

## Matrix Transpose Problem

```cuda
__shared__ float tile[32][32];  // Without padding

// When warp accesses row 0:
tile[0][0]  → bank 0
tile[0][1]  → bank 0  ← Conflict!
tile[0][2]  → bank 0  ← Conflict!
...
tile[0][31] → bank 0  ← Conflict!
```

## Solution: Padding

```cuda
__shared__ float tile[32][32 + 1];  // With +1 padding

// Now row 0 access pattern:
tile[0][0]  → bank 0
tile[0][1]  → bank 1  ← No conflict!
tile[0][2]  → bank 2  ← No conflict!
...
tile[0][31] → bank 31 ← No conflict!
```

## Visual Memory Layout

**Without Padding:**
```
Row 0: [0][1][2][3]...[31]     ← All map to banks 0,1,2,3...31
Row 1: [0][1][2][3]...[31]     ← All map to banks 0,1,2,3...31 (SAME!)
```

**With +1 Padding:**
```
Row 0: [0][1][2][3]...[31][X]  ← Maps to banks 0,1,2,3...31,0
Row 1: [0][1][2][3]...[31][X]  ← Maps to banks 1,2,3,4...31,0,1 (SHIFTED!)
```

## Performance Impact

- **32-way conflict:** 32× slower access
- **No conflict:** Full bandwidth utilization
- **Padding cost:** ~3% memory overhead for 32×32 tiles
- **Performance gain:** Up to 32× faster shared memory access

## Code Example

```cuda
// Bad: Bank conflicts
__shared__ float bad_tile[TILE_SIZE][TILE_SIZE];

// Good: No bank conflicts  
__shared__ float good_tile[TILE_SIZE][TILE_SIZE + 1];
```

The `+1` padding eliminates conflicts with minimal memory overhead.