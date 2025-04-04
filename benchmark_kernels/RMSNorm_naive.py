import torch
import torch.nn as nn
from kernels.RMSNorm_triton import custom_rms_forward_optimized


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
if __name__ == "__main__":
    Qwen2RMSNorm.forward = custom_rms_forward_optimized
    norm = Qwen2RMSNorm(hidden_size=896).cuda()
    ip = torch.rand(1, 1000, 896).cuda()
    op = norm(ip)
    print(op.shape)
