import torch
import triton
import triton.language as tl
import time
import numpy as np
from triton import next_power_of_2


@triton.jit
def fused_gate_up_kernel(
    x_ptr, gate_weight_ptr, up_weight_ptr, output_ptr,
    B, H, I,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_m = tl.multiple_of(offs_m, BLOCK_SIZE_M)
    mask_m = offs_m < B
    
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = tl.multiple_of(offs_n, BLOCK_SIZE_N)
    mask_n = offs_n < I
    
    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    gate_weight_ptrs = gate_weight_ptr + offs_n[:, None] * H
    up_weight_ptrs = up_weight_ptr + offs_n[:, None] * H
    
    for k in range(0, H, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < H
        
        x_ptrs = x_ptr + offs_m[:, None] * H + k_offsets[None, :]
        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        k_gate_weight_ptrs = gate_weight_ptrs + k_offsets[None, :]
        k_up_weight_ptrs = up_weight_ptrs + k_offsets[None, :]
        
        gate_weight_block = tl.load(
            k_gate_weight_ptrs, 
            mask=mask_n[:, None] & mask_k[None, :], 
            other=0.0
        )
        
        up_weight_block = tl.load(
            k_up_weight_ptrs, 
            mask=mask_n[:, None] & mask_k[None, :], 
            other=0.0
        )
        
        gate_acc += tl.dot(x_block, tl.trans(gate_weight_block))
        up_acc += tl.dot(x_block, tl.trans(up_weight_block))
    
    silu_gate = gate_acc * tl.sigmoid(gate_acc)
    fused_output = silu_gate * up_acc
    
    output_ptrs = output_ptr + offs_m[:, None] * I + offs_n[None, :]
    tl.store(output_ptrs, fused_output, mask=mask_m[:, None] & mask_n[None, :])

def fused_silu_gate_up(x, gate_weight, up_weight):
    batch_size, seq_len, hidden_size = x.shape
    intermediate_size, _ = gate_weight.shape
    
    x_2d = x.reshape(-1, hidden_size).contiguous()
    gate_weight = gate_weight.contiguous()
    up_weight = up_weight.contiguous()
    
    output_2d = torch.empty(
        (batch_size * seq_len, intermediate_size), 
        device=x.device, 
        dtype=x.dtype
    )
    

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 16
    BLOCK_SIZE_N = 64
    
    grid = (
        triton.cdiv(batch_size * seq_len, BLOCK_SIZE_M),
        triton.cdiv(intermediate_size, BLOCK_SIZE_N)
    )
    
    fused_gate_up_kernel[grid](
        x_2d, gate_weight, up_weight, output_2d,
        batch_size * seq_len, hidden_size, intermediate_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output_2d.reshape(batch_size, seq_len, intermediate_size)



class Qwen2MLPTriton(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x):
        fused_output = fused_silu_gate_up(
            x,
            self.gate_proj.weight,
            self.up_proj.weight
        )
        return self.down_proj(fused_output)
    
    
def custom_mlp_forward_optimized(self, x):
    fused_output = fused_silu_gate_up(
            x,
            self.gate_proj.weight,
            self.up_proj.weight
        )
    return self.down_proj(fused_output)