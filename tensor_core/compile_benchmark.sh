#!/bin/bash

# Compile tensor core benchmark with shared memory comparison
nvcc -o tensor_core_benchmark \
    test_tensor_core.cu \
    matmul_tensor_core.cu \
    matmul_tensor_core_shared.cu \
    ../matmul/matmul_tiled.cu \
    -lcudart -lm

echo "Compiled tensor_core_benchmark"
echo "Usage: ./tensor_core_benchmark <M> <K> <N>"
echo "Example: ./tensor_core_benchmark 1024 1024 1024"