import torch
import torch.nn as nn
from kernels.fused_mlp import custom_mlp_forward_optimized


class Qwen2MLPTorch(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x):
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        gate_silu = gate_out * torch.sigmoid(gate_out)
        fusion_output = gate_silu * up_out
        return self.down_proj(fusion_output)


if __name__ == "__main__":
    Qwen2MLPTorch.forward = custom_mlp_forward_optimized
    norm = Qwen2MLPTorch(hidden_size=896, intermediate_size=4864).cuda()
    ip = torch.rand(1, 1000, 896).cuda()
    op = norm(ip)
    print(op.shape)
