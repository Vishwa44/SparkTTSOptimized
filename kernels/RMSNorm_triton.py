import triton
import torch.nn as nn
import triton.language as tl
import torch
from triton import next_power_of_2
import math

@triton.jit
def _rmsnorm_forward_triton_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    eps,
    B,
    H,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    indices_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = tl.arange(0, BLOCK_SIZE_H)

    mask_b = indices_b < B
    mask_h = indices_h < H

    mask_bh = mask_b[:, None] & mask_h[None, :]

    x_ptrs = x_ptr + indices_b[:, None] * H + indices_h[None, :]
    x = tl.load(x_ptrs, mask=mask_bh).to(tl.float32)

    squared_sum = tl.sum(x * x, axis=1)
    inverse_rms = tl.rsqrt((squared_sum / H) + eps)

    x *= inverse_rms[:, None]

    if weight_ptr is not None:
        weight = tl.load(weight_ptr + indices_h, mask=mask_h)
        x = x.to(x_ptr.dtype.element_ty) * weight[None, :]

    output_ptrs = output_ptr + indices_b[:, None] * H + indices_h[None, :]
    tl.store(output_ptrs, x, mask=mask_bh)


def rmsnorm_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    output: torch.Tensor,
    eps: float,
    BLOCK_SIZE_B: int,
    BLOCK_SIZE_H: int,
) -> None:
    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size
    num_warps=8
    if BLOCK_SIZE_H < hidden_size:
        raise ValueError(f"hidden_size should be more than the BLOCK_SIZE_H")

    with torch.device(x.device):
        _rmsnorm_forward_triton_kernel[(math.ceil(num_elements/BLOCK_SIZE_B),)](
            x_ptr=x,
            weight_ptr=weight,
            output_ptr=output,
            eps=eps,
            B=num_elements,
            H=hidden_size,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            num_warps=num_warps
        )


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    
    x_2d = x.reshape(-1, hidden_size)
    output = torch.empty_like(x_2d)
    
    
    BLOCK_SIZE_B = 1
    BLOCK_SIZE_H = next_power_of_2(hidden_size)

    
    rmsnorm_forward_triton(
        x=x_2d,
        weight=weight,
        output=output,
        eps=eps,
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    
    output = output.reshape(orig_shape)
    
    return output


class TritonRMSNorm(nn.Module):
    """
    RMSNorm implementation using Triton kernels for better performance.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        return rmsnorm(
            hidden_states, 
            self.weight, 
            self.eps
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
    

def custom_rms_forward_optimized(self, hidden_states):
    return rmsnorm(
        hidden_states, 
        self.weight, 
        eps=self.variance_epsilon
    )