// Question 1 hello 
// CUDA Hello World 
//  y = a * x + b  
// x: 1 D array - float 
// a, b:  const (scalar )
// y : 1D array - float 

__global__ void affinity (float a, float b, float *x, int n, float* y) { // 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // y = a * x + b  
        y[idx] = a * x[idx] + b;
    }
}

// Question 2 y = [1,2,3,4, ... , 10,11,12]
//  n-th element summation 
// example, n = 3 

// [1+2+3, 2+3+4, 3+4+5,... 10+11+12, 11+12, 12]

__global__ void summation(int* input, int* output, int size, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        int sum = 0;
        int count = 0;
        
        for (int i = idx; i < size && count < n; i++) {

            // When a thread accesses input[i], it's loading data from global memory into the thread's registers.
            // Global Memory → L2 Cache → L1 Cache → Registers
            sum += input[i];
            count++;
        }
        
        output[idx] = sum;
    }
}

/*

Original version: Each thread reads n elements directly from global memory
cuda
// Thread 0: reads input[0], input[1], input[2] from global memory
// Thread 1: reads input[1], input[2], input[3] from global memory  
// Thread 2: reads input[2], input[3], input[4] from global memory
// → Lots of redundant global memory reads!

Shared memory version: 
1. Cooperative loading: All threads in a block load data into shared memory once
2. Window overlap: Load extra n-1 elements to handle overlapping windows
3. Local computation: Each thread computes its sum from fast shared memory

## Memory access pattern:

Block with 4 threads, n=3:

Global memory: [1][2][3][4][5][6][7]...

Shared memory: [1][2][3][4][5][6] (blockDim + n-1 elements)
                ↑           ↑
            block data   overlap data

Thread 0: sum(1,2,3) from shared[0:2]
Thread 1: sum(2,3,4) from shared[1:3]  
Thread 2: sum(3,4,5) from shared[2:4]
Thread 3: sum(4,5,6) from shared[3:5]


## Performance improvements:

- **Reduced global memory traffic**: From n reads per thread to ~1 read per thread
- **Coalesced access**: Threads read consecutive memory locations together
- **Cache efficiency**: Shared memory is ~100x faster than global memory
- **Bandwidth utilization**: Better use of memory bandwidth

Speed improvement: Typically 2-10x faster depending on n and block size.

*/

__global__ void nthElementSumShared(int* input, int* output, int size, int n) {
    extern __shared__ int shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory with padding for n-element windows
    int load_size = blockDim.x + n - 1;
    
    // Each thread loads one element
    if (idx < size) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0;
    }
    
    // Load additional elements for window overlap
    if (tid < n - 1 && blockIdx.x * blockDim.x + blockDim.x + tid < size) {
        shared_data[blockDim.x + tid] = input[blockIdx.x * blockDim.x + blockDim.x + tid];
    }
    
    __syncthreads();
    
    // Compute sum from shared memory
    if (idx < size) {
        int sum = 0;
        for (int i = 0; i < n && (tid + i) < load_size; i++) {
            sum += shared_data[tid + i];
        }
        output[idx] = sum;
    }
}

/*

but what if the shared memory is full because the matrix is too big?
Shared memory is limited (~48KB per block). 

Here are strategies:

## 1. Reduce block size:
cuda
// Instead of 1024 threads per block
dim3 block(256);  // Use smaller blocks when n is large

// Shared memory needed = (blockDim.x + n - 1) * sizeof(int)
// 256 threads + n-1 overlap ≈ 1KB for reasonable n

## 2. Tiled processing for very large n:
cuda
__global__ void nthElementSumTiled(int* input, int* output, int size, int n) {
    extern __shared__ int shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    int sum = 0;
    int tile_size = blockDim.x;
    
    // Process n elements in tiles
    for (int tile_start = 0; tile_start < n; tile_start += tile_size) {
        int elements_in_tile = min(tile_size, n - tile_start);
        
        // Load tile into shared memory
        if (tid < elements_in_tile && idx + tile_start < size) {
            shared_data[tid] = input[idx + tile_start];
        }
        __syncthreads();
        
        // Sum elements from this tile
        for (int i = 0; i < elements_in_tile; i++) {
            sum += shared_data[i];
        }
        __syncthreads();
    }
    
    output[idx] = sum;
}

## 3. Fallback to global memory:
cuda
// Check available shared memory at runtime
if ((blockDim.x + n - 1) * sizeof(int) > 48000) {
    // Use original global memory version
    nthElementSum<<<grid, block>>>(input, output, size, n);
} else {
    // Use shared memory version
    nthElementSumShared<<<grid, block, shared_mem>>>(input, output, size, n);
}

The tiled approach works for any n size while still getting some shared memory benefits!
*/

__global__ void nthElementSumTiled(int* input, int* output, int size, int n) {
    extern __shared__ int shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    int sum = 0;
    int tile_size = blockDim.x;
    
    // Process n elements in tiles
    for (int tile_start = 0; tile_start < n; tile_start += tile_size) {
        int elements_in_tile = min(tile_size, n - tile_start);
        
        // Load tile into shared memory
        if (tid < elements_in_tile && idx + tile_start < size) {
            shared_data[tid] = input[idx + tile_start];
        }
        __syncthreads();
        
        // Sum elements from this tile
        for (int i = 0; i < elements_in_tile; i++) {
            sum += shared_data[i];
        }
        __syncthreads();
    }
    
    output[idx] = sum;
}