import math
import time
import torch
import torch.nn as nn
import triton
import triton.language as tl
import numpy as np
from triton import next_power_of_2
from kernels.RMSNorm_triton import TritonRMSNorm

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


def benchmark_rmsnorm(batch_size, seq_len, hidden_size, dtype=torch.float16, num_runs=100, device="cuda"):
    """
    Benchmark different RMSNorm implementations and verify accuracy.
    
    Args:
        batch_size: Batch size for the input tensor
        seq_len: Sequence length for the input tensor
        hidden_size: Hidden dimension size
        dtype: Data type to use for the tensors
        num_runs: Number of runs for timing
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        Dictionary containing timing results and accuracy metrics
    """
    # Create input tensor with same values for fair comparison
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    
    # Set up implementations
    pytorch_module = Qwen2RMSNorm(hidden_size).to(device)
    pytorch_module.weight.data = weight.clone()
    
    triton_module = TritonRMSNorm(hidden_size).to(device)
    triton_module.weight.data = weight.clone()
    
    # Warmup runs
    for _ in range(10):
        pytorch_out = pytorch_module(x)
        triton_out = triton_module(x)
    
    # Check accuracy
    pytorch_out = pytorch_module(x)
    triton_out = triton_module(x)
    
    # Calculate metrics
    abs_diff = (pytorch_out - triton_out).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    relative_diff = (abs_diff / (pytorch_out.abs() + 1e-7)).mean().item()
    
    # Benchmark PyTorch implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        pytorch_out = pytorch_module(x)
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs * 1000  # ms
    
    # Benchmark Triton implementation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        triton_out = triton_module(x)
        torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_runs * 1000  # ms
    
    # Return results
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
            "hidden_size": hidden_size
        },
        "dtype": str(dtype)
    }
    
    return results


def run_benchmark_suite():
    """
    Run a suite of benchmarks for different shapes and dtypes.
    """
    configs = [
        # Small models
        {"batch_size": 1, "seq_len": 512, "hidden_size": 896},
        {"batch_size": 8, "seq_len": 512, "hidden_size": 896},
        # Medium models
        {"batch_size": 1, "seq_len": 1000, "hidden_size": 896},
        {"batch_size": 4, "seq_len": 1024, "hidden_size": 896},
        # # Large models
        {"batch_size": 1, "seq_len": 2048, "hidden_size": 896},
        {"batch_size": 2, "seq_len": 2048, "hidden_size": 896},
        # # Very large models
        # {"batch_size": 1, "seq_len": 4096, "hidden_size": 8192},
    ]
    
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    results = []
    
    for config in configs:
        for dtype in dtypes:
            print(f"Benchmarking {config} with {dtype}...")
            result = benchmark_rmsnorm(**config, dtype=dtype)
            results.append(result)
            
            # Print current result
            shape_info = f"[{config['batch_size']}×{config['seq_len']}×{config['hidden_size']}] {str(dtype)}"
            print(f"  {shape_info}: PyTorch: {result['pytorch_time_ms']:.2f}ms, Triton: {result['triton_time_ms']:.2f}ms, "
                  f"Speedup: {result['speedup']:.2f}x, Max diff: {result['max_abs_diff']:.2e}")
    
    return results


if __name__ == "__main__":
    # If this script is run directly, execute the benchmark suite
    if torch.cuda.is_available():
        results = run_benchmark_suite()
        
        # Print summary
        print("\nSummary:")
        avg_speedup = np.mean([r["speedup"] for r in results])
        max_speedup = np.max([r["speedup"] for r in results])
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        print(f"Average max absolute difference: {np.mean([r['max_abs_diff'] for r in results]):.2e}")
    else:
        print("CUDA not available. Benchmarks require a GPU to run.")