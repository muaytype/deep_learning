import torch
import time

M, N, K = 5000, 5000, 5000

def time_multiplication(dtype, use_gpu=True):
    device = "cuda" if use_gpu else "cpu"
    
    A = torch.rand(M, N, dtype=dtype, device=device)
    B = torch.rand(N, K, dtype=dtype, device=device)
    
    if use_gpu:
        _ = A @ B
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    result = A @ B
    
    if use_gpu:
        torch.cuda.synchronize()
        
    elapsed = time.time() - start_time
    hardware = "GPU" if use_gpu else "CPU"
    print(f"{hardware} Execution ({dtype}): {elapsed:.4f} seconds")

print("--- Part III: Hardware & Precision Race ---")

time_multiplication(torch.float32, use_gpu=True)
time_multiplication(torch.float16, use_gpu=True)
time_multiplication(torch.bfloat16, use_gpu=True)

time_multiplication(torch.float32, use_gpu=False)
