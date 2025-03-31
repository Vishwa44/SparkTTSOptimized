import torch
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triton
import triton.language as tl
import time
import numpy as np
from triton import next_power_of_2
from kernels.fused_mlp import Qwen2MLPTriton

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


def benchmark_mlp_implementations(batch_size, seq_len, hidden_size, intermediate_size, 
                              dtype=torch.float16, num_runs=100, device="cuda"):
    """
    Benchmark PyTorch and Triton implementations of MLP and verify accuracy.
    
    Args:
        batch_size: Batch size for the input tensor
        seq_len: Sequence length for the input tensor
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size (typically 4x hidden_size)
        dtype: Data type to use for the tensors
        num_runs: Number of runs for timing
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        Dictionary containing timing results and accuracy metrics
    """
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    
    gate_weight = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
    up_weight = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device)
    
    pytorch_mlp = Qwen2MLPTorch(hidden_size, intermediate_size).to(device)
    pytorch_mlp.gate_proj.weight.data = gate_weight.clone()
    pytorch_mlp.up_proj.weight.data = up_weight.clone()
    pytorch_mlp.down_proj.weight.data = down_weight.clone()
    
    triton_mlp = Qwen2MLPTriton(hidden_size, intermediate_size).to(device)
    triton_mlp.gate_proj.weight.data = gate_weight.clone()
    triton_mlp.up_proj.weight.data = up_weight.clone()
    triton_mlp.down_proj.weight.data = down_weight.clone()
    
    for _ in range(10):
        pytorch_out = pytorch_mlp(x)
        triton_out = triton_mlp(x)
    
    torch.cuda.synchronize()
    pytorch_out = pytorch_mlp(x)
    triton_out = triton_mlp(x)
    
    abs_diff = (pytorch_out - triton_out).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    relative_diff = (abs_diff / (pytorch_out.abs() + 1e-7)).mean().item()
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        pytorch_out = pytorch_mlp(x)
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs * 1000  # ms
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        triton_out = triton_mlp(x)
        torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_runs * 1000  # ms
    
    results = {
        "pytorch_time_ms": pytorch_time,
        "triton_time_ms": triton_time,
        "speedup": pytorch_time / triton_time,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "mean_relative_diff": relative_diff,
        "shapes": {
            "batch_size": batch_size,
            "seq_len": seq_len, 
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size
        },
        "dtype": str(dtype)
    }
    
    return results

def run_mlp_benchmark_suite():
    """
    Run a suite of benchmarks for different shapes and dtypes.
    """
    configs = [
        {"batch_size": 1, "seq_len": 512, "hidden_size": 896, "intermediate_size": 4864},
        {"batch_size": 4, "seq_len": 512, "hidden_size": 896, "intermediate_size": 4864},
    ]
    
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    results = []
    
    for config in configs:
        for dtype in dtypes:
            print(f"Benchmarking {config} with {dtype}...")
            try:
                result = benchmark_mlp_implementations(**config, dtype=dtype, num_runs=20)
                results.append(result)
                
                shape_info = f"[{config['batch_size']}×{config['seq_len']}×{config['hidden_size']}→{config['intermediate_size']}] {str(dtype)}"
                print(f"  {shape_info}: PyTorch: {result['pytorch_time_ms']:.2f}ms, Triton: {result['triton_time_ms']:.2f}ms, "
                    f"Speedup: {result['speedup']:.2f}x, Max diff: {result['max_abs_diff']:.2e}")
            except Exception as e:
                print(f"  Error benchmarking {config} with {dtype}: {e}")
    
    return results

def print_detailed_results(results):
    """
    Print detailed benchmark results in a table format
    """
    print("\n=== Detailed Benchmark Results ===")
    print("-" * 120)
    print(f"{'Configuration':<30} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'Max Abs Diff':<15} {'Rel Diff':<15}")
    print("-" * 120)
    
    for r in results:
        config = f"{r['shapes']['batch_size']}×{r['shapes']['seq_len']}×{r['shapes']['hidden_size']}→{r['shapes']['intermediate_size']} {r['dtype']}"
        print(f"{config:<30} {r['pytorch_time_ms']:<15.2f} {r['triton_time_ms']:<15.2f} {r['speedup']:<10.2f} {r['max_abs_diff']:<15.2e} {r['mean_relative_diff']:<15.2e}")
    
    print("-" * 120)

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Running Qwen MLP benchmark suite...")
        results = run_mlp_benchmark_suite()
        
        print_detailed_results(results)
        
        print("\nSummary:")
        avg_speedup = np.mean([r["speedup"] for r in results])
        max_speedup = np.max([r["speedup"] for r in results])
        min_speedup = np.min([r["speedup"] for r in results])
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        print(f"Minimum speedup: {min_speedup:.2f}x")
        print(f"Average max absolute difference: {np.mean([r['max_abs_diff'] for r in results]):.2e}")
    else:
        print("CUDA not available. Benchmarks require a GPU to run.")