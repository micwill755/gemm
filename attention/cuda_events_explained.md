# CUDA Events Timing Explained

## Why CUDA Events?

CPU timers like `clock()` don't work for GPU timing because:
- GPU kernels execute **asynchronously** 
- CPU timer measures when kernel was **launched**, not when it **finished**
- Need to measure actual GPU execution time

## How CUDA Events Work

CUDA events are **timestamps placed in the GPU execution stream**:

```
CPU Thread:                GPU Stream:
-----------                -----------

cudaEventCreate(&start)    
cudaEventCreate(&stop)     

cudaEventRecord(start) -----> [START_EVENT] 
                              
kernel_launch() -----------> [KERNEL_1]
kernel_launch() -----------> [KERNEL_2] 
kernel_launch() -----------> [KERNEL_3]

cudaEventRecord(stop) ------> [STOP_EVENT]

cudaEventSynchronize(stop) <-- Wait for stop event
                              
cudaEventElapsedTime() <------ Calculate: STOP - START
```

## Timeline Comparison

**❌ Wrong way (CPU timer):**
```
CPU: |--start--launch_kernel--end--|  ← Measures launch time only
GPU:           |--------kernel_execution--------|  ← Actual work
```

**✅ Right way (CUDA events):**
```
CPU: |--start--launch_kernel--sync--|
GPU:           |--[START]--kernel_execution--[STOP]--|  ← Measures actual GPU work
```

## Code Flow

```c
// 1. Create event objects
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 2. Record start timestamp in GPU stream
cudaEventRecord(start);

// 3. Launch kernels (asynchronous)
my_kernel<<<blocks, threads>>>();

// 4. Record stop timestamp in GPU stream  
cudaEventRecord(stop);

// 5. Wait for stop event (ensures kernels finished)
cudaEventSynchronize(stop);

// 6. Calculate elapsed time on GPU
float ms;
cudaEventElapsedTime(&ms, start, stop);

// 7. Cleanup
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

## Key Benefits

- **Accurate**: Measures actual GPU execution time
- **Non-blocking**: Doesn't stall CPU-GPU pipeline  
- **Stream-aware**: Works with CUDA streams
- **Microsecond precision**: High-resolution timing