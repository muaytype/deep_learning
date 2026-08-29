import torch
import time

# 1. Initialize a large matrix size
M, N = 10000, 10000

# 2. Allocate on CPU using FP32 (Consumes ~400 MB of standard RAM)
print("Allocating Float32 tensor on CPU...")
tensor_cpu = torch.randn(M, N, dtype=torch.float32)

# 3. Time the CPU execution for matrix multiplication
start = time.time()
result_cpu = tensor_cpu @ tensor_cpu
print(f"CPU Time (Float32): {time.time() - start:.4f} seconds")

# 4. Move to GPU and downcast to BF16 (Halves footprint to ~200 MB VRAM)
if torch.cuda.is_available():
    print("\nMoving tensor to GPU and casting to Bfloat16...")
    tensor_gpu = tensor_cpu.to('cuda', dtype=torch.bfloat16)
    
    # 5. Time the GPU execution
    start = time.time()
    result_gpu = tensor_gpu @ tensor_gpu
    print(f"GPU Time (Bfloat16): {time.time() - start:.4f} seconds")
else:
    print("\nNo local GPU detected. This will execute when we send it to Slurm!")