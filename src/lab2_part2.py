import torch

M, N, K = 100, 100, 100

def test_precision(scale_factor=1.0):
    print(f"\n--- Testing with Scale Factor: {scale_factor} ---")
    
    A_f32 = torch.rand(M, N, dtype=torch.float32, device="cuda") * scale_factor
    B_f32 = torch.rand(N, K, dtype=torch.float32, device="cuda") * scale_factor
    
    result_f32 = A_f32 @ B_f32
    
    A_f16, B_f16 = A_f32.to(torch.float16), B_f32.to(torch.float16)
    result_f16 = A_f16 @ B_f16
    
    A_bf16, B_bf16 = A_f32.to(torch.bfloat16), B_f32.to(torch.bfloat16)
    result_bf16 = A_bf16 @ B_bf16
    
    diff_f16 = torch.abs(result_f32 - result_f16).mean().item()
    diff_bf16 = torch.abs(result_f32 - result_bf16).mean().item()
    
    print(f"Average Error (Float16 vs Float32):  {diff_f16:.4f}")
    print(f"Average Error (Bfloat16 vs Float32): {diff_bf16:.4f}")

test_precision(scale_factor=1.0)

test_precision(scale_factor=100.0)
