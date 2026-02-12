NVCC = nvcc
CUDA_FLAGS = -arch=sm_70 -lcudart -lm
TENSOR_FLAGS = -arch=sm_70 -lcudart -lm

# Tensor Core benchmark
tensor_test: test_tensor_core.cu matmul_tensor_core.cu fp_conversion.cu ../matmul/matmul_tiled.cu
\t$(NVCC) $(TENSOR_FLAGS) -o tensor_test test_tensor_core.cu matmul_tensor_core.cu fp_conversion.cu ../matmul/matmul_tiled.cu

# Individual components
matmul_tensor_core.o: matmul_tensor_core.cu
\t$(NVCC) $(TENSOR_FLAGS) -c matmul_tensor_core.cu

fp_conversion.o: fp_conversion.cu
\t$(NVCC) $(CUDA_FLAGS) -c fp_conversion.cu

test_tensor_core.o: test_tensor_core.cu
\t$(NVCC) $(CUDA_FLAGS) -c test_tensor_core.cu

clean:
\trm -f *.o tensor_test

.PHONY: clean