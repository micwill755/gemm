NVCC = nvcc
CUDA_FLAGS = -arch=sm_70 -lcudart -lm
TENSOR_FLAGS = -arch=sm_70 -lcudart -lm

# Tensor Core benchmark
tensor_test: test_tensor_core.cu matmul_tensor_core.cu fp_conversion.cu ../matmul/matmul_tiled.cu
\t$(NVCC) $(TENSOR_FLAGS) -o tensor_test test_tensor_core.cu matmul_tensor_core.cu fp_conversion.cu ../matmul/matmul_tiled.cu

# Attention Tensor Core benchmark
attention_test: test_attention_tensor_core.cu attention_tensor_core.cu matmul_tensor_core.cu fp_conversion.cu
\t$(NVCC) $(TENSOR_FLAGS) -o attention_test test_attention_tensor_core.cu attention_tensor_core.cu matmul_tensor_core.cu fp_conversion.cu

# Individual components
matmul_tensor_core.o: matmul_tensor_core.cu
\t$(NVCC) $(TENSOR_FLAGS) -c matmul_tensor_core.cu

fp_conversion.o: fp_conversion.cu
\t$(NVCC) $(CUDA_FLAGS) -c fp_conversion.cu

test_tensor_core.o: test_tensor_core.cu
\t$(NVCC) $(CUDA_FLAGS) -c test_tensor_core.cu

clean:
\trm -f *.o tensor_test attention_test

.PHONY: clean