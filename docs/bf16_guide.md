# BF16 (Brain Floating Point 16) - Complete Guide

## What is BF16?

BF16 (Brain Floating Point 16) is a 16-bit floating-point format specifically designed for deep learning workloads. Originally developed by Google for their TPUs, it's now widely supported across modern AI accelerators.

---

## Format Structure

### BF16 Bit Layout
```
┌─────┬──────────────┬─────────────────────┐
│  S  │   Exponent   │      Mantissa       │
│ 1bit│    8 bits    │       7 bits        │
└─────┴──────────────┴─────────────────────┘
```

### Comparison with Other Formats

| Format | Sign | Exponent | Mantissa | Total | Dynamic Range |
|--------|------|----------|----------|-------|---------------|
| **FP32** | 1 bit | 8 bits | 23 bits | 32 bits | ±3.4 × 10³⁸ |
| **BF16** | 1 bit | 8 bits | 7 bits | 16 bits | ±3.4 × 10³⁸ |
| **FP16** | 1 bit | 5 bits | 10 bits | 16 bits | ±6.5 × 10⁴ |

### Key Insight

BF16 is essentially **FP32 with the mantissa truncated** from 23 bits to 7 bits:
- Same exponent bits (8) = same dynamic range as FP32
- Fewer mantissa bits (7) = less precision per value
- Half the memory footprint of FP32

---

## Why BF16 Exists

### The Problem with FP16

**FP16 limitations:**
- Only 5 exponent bits → narrow dynamic range (±65,504)
- Prone to overflow/underflow during training
- Requires loss scaling and careful gradient management
- Not ideal for large model training

**Example of FP16 overflow:**
```python
import torch

# Large gradient value
grad = torch.tensor([70000.0], dtype=torch.float16)
print(grad)  # tensor([inf], dtype=torch.float16) ← Overflow!

# Same value in BF16
grad_bf16 = torch.tensor([70000.0], dtype=torch.bfloat16)
print(grad_bf16)  # tensor([70016.], dtype=torch.bfloat16) ← Works!
```

### The BF16 Solution

✅ **Same dynamic range as FP32** - No overflow issues  
✅ **50% memory reduction** - Fits larger models  
✅ **Simple conversion** - Truncate FP32 mantissa  
✅ **Hardware efficient** - Easy to implement in silicon  
✅ **Training friendly** - Better numerical stability than FP16

---

## BF16 and Tensor Cores

### The Relationship

**BF16 is NOT exclusively a Tensor Core thing, but they work great together:**

- **BF16 = Data format** (how numbers are represented)
- **Tensor Cores = Hardware accelerator** (specialized compute units)

### Hardware Support Timeline

#### Ampere Architecture (2020) - First-Class BF16
**GPUs:** A100, RTX 3090, RTX 3080, etc.

**Tensor Core Performance:**
- BF16: 312 TFLOPS
- TF32: 156 TFLOPS  
- FP32: 19.5 TFLOPS

**16× speedup for BF16 vs FP32!**

#### Hopper Architecture (2022) - Enhanced BF16
**GPUs:** H100

**Improvements:**
- Even faster BF16 Tensor Core operations
- FP8 support introduced
- Transformer Engine optimizations

#### Ada Lovelace (2022) - Consumer BF16
**GPUs:** RTX 4090, RTX 4080, etc.

- Consumer GPUs with BF16 Tensor Core support
- Great for local LLM training/fine-tuning

#### Older Architectures (Volta/Turing)
**GPUs:** V100, RTX 2080, etc.

❌ **No BF16 Tensor Core support**
- Tensor Cores exist but only support FP16/FP32
- BF16 operations fall back to CUDA cores (slow)

---

## Using BF16 in PyTorch

### Basic Usage

```python
import torch

# Create BF16 tensor
x = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')

# Convert existing tensor
x_fp32 = torch.randn(1024, 1024, device='cuda')
x_bf16 = x_fp32.to(torch.bfloat16)

# Matrix multiplication (uses Tensor Cores on Ampere+)
y = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
z = x @ y  # Fast on Ampere+ GPUs!
```

### Mixed Precision Training with BF16

```python
import torch
from torch.cuda.amp import autocast

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())

for data, target in dataloader:
    optimizer.zero_grad()
    
    # Automatic mixed precision with BF16
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)
    
    # Backward pass (gradients in FP32)
    loss.backward()
    optimizer.step()
```

### BF16 vs FP16 Training

```python
# FP16 training (requires GradScaler)
from torch.cuda.amp import GradScaler

scaler = GradScaler()

with autocast(device_type='cuda', dtype=torch.float16):
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# BF16 training (no scaler needed!)
with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

---

## Performance Characteristics

### Memory Bandwidth

```python
# Memory usage comparison
batch_size = 32
seq_len = 512
hidden_dim = 768

# FP32: 32 bits per element
fp32_memory = batch_size * seq_len * hidden_dim * 4  # 50.3 MB

# BF16: 16 bits per element
bf16_memory = batch_size * seq_len * hidden_dim * 2  # 25.2 MB

# 50% memory reduction!
```

### Compute Throughput

**On NVIDIA A100:**

| Operation | FP32 | BF16 | Speedup |
|-----------|------|------|---------|
| Matrix Multiply (Tensor Cores) | 19.5 TFLOPS | 312 TFLOPS | 16× |
| Convolution | ~20 TFLOPS | ~300 TFLOPS | 15× |
| Attention (Flash Attention) | Baseline | 2-4× faster | 2-4× |

### Real-World Training Speed

```python
# Benchmark example
import time
import torch

model = torch.nn.Linear(4096, 4096).cuda()
x = torch.randn(128, 4096).cuda()

# FP32 benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    y = model(x)
torch.cuda.synchronize()
fp32_time = time.time() - start

# BF16 benchmark
model_bf16 = model.to(torch.bfloat16)
x_bf16 = x.to(torch.bfloat16)

torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    y = model_bf16(x_bf16)
torch.cuda.synchronize()
bf16_time = time.time() - start

print(f"FP32: {fp32_time:.3f}s")
print(f"BF16: {bf16_time:.3f}s")
print(f"Speedup: {fp32_time/bf16_time:.2f}×")
```

---

## When to Use BF16

### ✅ Great For

**Training Large Models:**
- LLMs (GPT, LLaMA, etc.)
- Vision Transformers
- Diffusion models
- Any model where memory is a bottleneck

**Why:** Reduces memory by 50%, enables larger batch sizes, faster training

**Mixed Precision Training:**
- Stable gradients without loss scaling
- Simpler training code than FP16
- Better numerical stability

**Hardware with BF16 Support:**
- NVIDIA Ampere/Hopper GPUs (A100, H100, RTX 30xx/40xx)
- Google TPUs
- Intel Xeon (AVX-512 BF16 instructions)

### ❌ Not Ideal For

**Inference on Edge Devices:**
- FP16 or INT8 quantization usually better
- Smaller models don't need BF16's dynamic range

**High-Precision Requirements:**
- Scientific computing
- Financial calculations
- Use FP32 or FP64 instead

**Older Hardware:**
- Pre-Ampere NVIDIA GPUs
- Falls back to slow emulation

---

## BF16 in Production

### Large Language Model Training

```python
# Example: Training LLaMA-style model with BF16
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,  # Load in BF16
    device_map="auto"
)

training_args = TrainingArguments(
    output_dir="./output",
    bf16=True,  # Enable BF16 training
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### Checking Hardware Support

```python
import torch

# Check if BF16 is supported
if torch.cuda.is_available():
    # Check GPU compute capability
    capability = torch.cuda.get_device_capability()
    
    # Ampere (8.0+) and Hopper (9.0+) support BF16
    if capability[0] >= 8:
        print(f"✅ BF16 supported on {torch.cuda.get_device_name()}")
        print(f"   Compute capability: {capability[0]}.{capability[1]}")
    else:
        print(f"❌ BF16 not supported on {torch.cuda.get_device_name()}")
        print(f"   Compute capability: {capability[0]}.{capability[1]}")
else:
    print("❌ CUDA not available")

# Check if BF16 Tensor Cores are being used
if torch.cuda.is_bf16_supported():
    print("✅ BF16 Tensor Cores available")
else:
    print("❌ BF16 Tensor Cores not available")
```

---

## Common Pitfalls

### 1. Assuming BF16 = Automatic Speedup

```python
# ❌ This won't use Tensor Cores
x = torch.randn(10, 10, dtype=torch.bfloat16, device='cuda')
y = x + 1  # Element-wise ops don't use Tensor Cores

# ✅ This will use Tensor Cores
x = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
y = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
z = x @ y  # Matrix multiply uses Tensor Cores
```

**Key:** BF16 speedup comes from Tensor Core operations (matmul, conv), not all operations.

### 2. Mixed Dtype Operations

```python
# ❌ Type mismatch - will upcast to FP32
x_bf16 = torch.randn(100, 100, dtype=torch.bfloat16, device='cuda')
y_fp32 = torch.randn(100, 100, dtype=torch.float32, device='cuda')
z = x_bf16 @ y_fp32  # Computed in FP32!

# ✅ Keep dtypes consistent
y_bf16 = y_fp32.to(torch.bfloat16)
z = x_bf16 @ y_bf16  # Computed in BF16
```

### 3. CPU BF16 (Slow!)

```python
# ❌ BF16 on CPU is emulated (very slow)
x = torch.randn(1000, 1000, dtype=torch.bfloat16)  # CPU tensor
y = x @ x  # Slow!

# ✅ Use BF16 on GPU
x = torch.randn(1000, 1000, dtype=torch.bfloat16, device='cuda')
y = x @ x  # Fast with Tensor Cores
```

---

## Summary

### Key Takeaways

1. **BF16 = FP32 range + FP16 memory**
   - 8 exponent bits (same as FP32) = wide dynamic range
   - 7 mantissa bits = reduced precision (but fine for DL)
   - 16 bits total = 50% memory savings

2. **BF16 ≠ Tensor Cores, but they're best friends**
   - BF16 is a data format
   - Tensor Cores are hardware accelerators
   - Ampere+ GPUs have BF16 Tensor Cores (huge speedup)

3. **Use BF16 for training large models**
   - Simpler than FP16 (no loss scaling)
   - More stable gradients
   - Enables larger batch sizes

4. **Check hardware support**
   - Ampere/Hopper GPUs: ✅ Native BF16
   - Volta/Turing GPUs: ❌ No BF16 Tensor Cores
   - CPU: ⚠️ Emulated (slow)

### Quick Reference

```python
# Enable BF16 training (PyTorch)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)

# Check support
torch.cuda.is_bf16_supported()

# Convert tensors
x_bf16 = x_fp32.to(torch.bfloat16)
```

---

## Further Reading

- [NVIDIA Ampere Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- [Google's BF16 Blog Post](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
- [PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [Hugging Face Training in BF16](https://huggingface.co/docs/transformers/perf_train_gpu_one)
