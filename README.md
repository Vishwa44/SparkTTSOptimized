# SparkTTSOptimized

This repository contains optimizations for Spark-TTS-0.5B to reduce the time to first audio. The goal is to improve the user experience by reducing latency in text-to-speech generation without compromising quality.

## Overview

I identified and optimized key bottlenecks in the Spark-TTS-0.5B model through profiling and custom kernel implementations. The optimizations focus on:

1. Implementing custom Triton kernels for the most compute-intensive operations
2. Creating an efficient generation pipeline with optimized sampling logic
3. Fusing operations where possible to reduce overhead

## Profiling Results

I profiled the naive HuggingFace implementation and identified the following main bottlenecks:

### Forward Pass Profiling

The profiling of the forward pass revealed two major performance bottlenecks:

1. **RMSNorm Operations**: Taking approximately 20% of the forward pass time
2. **MLP Blocks**: Taking approximately 18% of the forward pass time
3. **Attention Mechanism**: Taking approximately 60% of the forward pass time

For the attention mechanism, I used Flash Attention 2 (`flash_attention_2`), which already provides significant speedups over the standard attention implementation. This allowed us to focus the custom optimization efforts on RMSNorm and MLP blocks.

### Sequence Length Impact on Performance

We tested various sequence lengths to understand scaling behavior:

| Sequence Length | PyTorch Implementation | Optimized Triton Implementation | Speedup |
|-----------------|------------------------|--------------------------------|---------|
| 500 tokens      | 0.0497s               | 0.0421s                        | 1.18x   |
| 1000 tokens     | 0.1188s               | 0.0700s                        | 1.70x   |
| 1500 tokens     | 0.2377s               | 0.1064s                        | 2.23x   |
| 2000 tokens     | 0.3953s               | 0.2181s                        | 1.81x   |
| 2500 tokens     | 0.5909s               | 0.3428s                        | 1.72x   |

The speedup factor increases with sequence length up to a point, which is critical for TTS applications where longer text inputs are common.

### Kernel-Specific Benchmarks

#### RMSNorm Kernel

| Configuration (batch×seq×hidden) | PyTorch (ms) | Triton (ms) | Speedup | Max Abs Diff |
|----------------------------------|--------------|-------------|---------|--------------|
| 1×512×896 (float32)              | 0.185        | 0.076       | 2.43x   | 5.96e-07     |
| 8×512×896 (float32)              | 1.297        | 0.374       | 3.47x   | 5.96e-07     |
| 1×512×896 (float16)              | 0.098        | 0.042       | 2.33x   | 5.86e-04     |
| 8×512×896 (float16)              | 0.672        | 0.173       | 3.88x   | 5.86e-04     |
| 1×512×896 (bfloat16)             | 0.097        | 0.043       | 2.25x   | 9.77e-03     |
| 8×512×896 (bfloat16)             | 0.668        | 0.172       | 3.88x   | 9.77e-03     |

#### MLP Kernel

| Configuration (batch×seq×hidden→inter) | PyTorch (ms) | Triton (ms) | Speedup | Max Abs Diff |
|---------------------------------------|--------------|-------------|---------|--------------|
| 1×512×896→4864 (float32)              | 2.254        | 0.793       | 2.84x   | 8.58e-05     |
| 4×512×896→4864 (float32)              | 8.937        | 3.101       | 2.88x   | 8.59e-05     |
| 1×512×896→4864 (float16)              | 1.136        | 0.438       | 2.59x   | 1.95e-03     |
| 4×512×896→4864 (float16)              | 4.515        | 1.693       | 2.67x   | 1.96e-03     |
| 1×512×896→4864 (bfloat16)             | 1.145        | 0.442       | 2.59x   | 3.13e-02     |
| 4×512×896→4864 (bfloat16)             | 4.523        | 1.694       | 2.67x   | 3.13e-02     |

## Implementation Details

### 1. Attention Optimization with Flash Attention 2

We utilized `flash_attention_2` from HuggingFace's Transformers library to optimize the attention mechanism, which represents around 60% of the inference time. Flash Attention 2:

- Reduces memory bandwidth requirements by fusing operations
- Uses a tiling strategy to maximize GPU cache utilization
- Provides 2-4x speedup over standard attention implementations
- Maintains numerical stability and accuracy

Using `flash_attention_2` allows for faster processing of longer sequences, which is critical for TTS applications where inputs can be lengthy.

### 2. RMSNorm Optimization

I implemented a custom Triton kernel for RMSNorm that:
- Fuses the normalization and scaling operations
- Uses block-based parallelism to maximize GPU utilization
- Optimizes memory access patterns

The optimized RMSNorm kernel achieves 2.2-3.9x speedup depending on batch size and data type.

### 3. MLP Block Optimization

The MLP blocks in transformer models contain multiple operations that can be fused:
- SiLU activation (x * sigmoid(x))
- Element-wise multiplication
- Multiple matrix multiplications

The fused MLP kernel combines:
- Gate projection and up projection into a single kernel launch
- SiLU activation and multiplication in the same kernel
- Optimized matrix multiplication patterns

This optimization yields a 2.6-2.9x speedup for MLP operations.

### 4. Efficient Generation Pipeline

I implemented an optimized text generation pipeline with:
- Custom Triton-based sampling kernel supporting:
  - Temperature scaling
  - Top-k filtering
  - Top-p (nucleus) sampling
- Efficient softmax implementation

## Thought Process

1. **Focus on High-Impact Components**: We prioritized optimizing components that:
   - Consumed the most execution time
   - Could benefit from parallelism and memory access optimizations
   - Were called frequently during inference

2. **Kernel Fusion Strategy**: We identified operations that could be fused to reduce kernel launch overhead and memory transfers, particularly in:
   - Normalization layers
   - MLP blocks
   - Sampling logic

## Learnings

1. **Memory Access Patterns Matter**: Optimizing memory access patterns often provided larger speedups than computational optimizations, especially in memory-bound operations like normalization.

2. **Kernel Launch Overhead**: For operations with relatively small compute requirements, the kernel launch overhead can be significant. Fusing operations reduced this overhead.

3. **Data Type Impact**: Different data types (float32, float16, bfloat16) showed different optimization potential, with mixed-precision operations offering a good balance between performance and accuracy.

4. **Sequence Length Scaling**: Optimizations showed different speedup factors at different sequence lengths, indicating that optimization strategies should consider the expected workload characteristics.

5. **Profiling Granularity**: Fine-grained profiling was essential to identify specific operations within larger blocks that contributed disproportionately to execution time.


### Running the Optimized Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from kernels import custom_rms_forward_optimized, custom_mlp_forward_optimized
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP

# Apply the optimized kernels
Qwen2RMSNorm.forward = custom_rms_forward_optimized
Qwen2MLP.forward = custom_mlp_forward_optimized

# Load the model with optimized attention
model = AutoModelForCausalLM.from_pretrained("path/to/model", 
                                             torch_dtype="bfloat16", 
                                             _attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")
audio_tokenizer = BiCodecTokenizer("path/to/model", device="cuda:0")

# Generate text with optimized sampling
from efficient_generation import generate_text
output = generate_text(model, tokenizer, prompt, 
                      temperature=0.8, 
                      top_k=50, 
                      top_p=0.9)
```

### Benchmarking

```bash
# Compare optimized vs original implementation
python Compare_optimized.py --model_dir /path/to/model

# Benchmark specific kernels
python benchmark_kernels/benchmark_rmsnorm.py
python benchmark_kernels/benchmark_mlp.py
```

## Conclusion

The optimizations achieve a significant speedup (1.7-2.2x) for the Spark-TTS-0.5B model, reducing the time to first audio and improving the overall user experience. The custom Triton kernels for RMSNorm and MLP operations, combined with an efficient sampling implementation, address the main bottlenecks identified through profiling.

These improvements maintain the model's accuracy while significantly reducing latency, making the TTS system more responsive and suitable for interactive applications.