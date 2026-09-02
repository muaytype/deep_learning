import torch

M = 20000
N = 20000

print(f"Initializing matrices of shape ({M}, {N})...\n")

mat_f32 = torch.rand(M, N, dtype=torch.float32, device="cuda")
mem_f32 = mat_f32.element_size() * mat_f32.nelement() / (1024 ** 3)

print(f"Float32 Memory Allocated: {mem_f32:.4f} GB")

mat_f16 = torch.rand(M, N, dtype=torch.float16, device="cuda")
mem_f16 = mat_f16.element_size() * mat_f16.nelement() / (1024 ** 3)

print(f"Float16 Memory Allocated: {mem_f16:.4f} GB")
