import torch
import triton
import triton.language as tl

@triton.jit
def fused_temperature_topk_topp_kernel(
    logits_ptr, out_ptr, 
    temp, top_k, top_p,
    n_vocab, BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for temperature scaling, top-k and top-p sampling.
    
    Args:
        logits_ptr: Pointer to logits tensor
        out_ptr: Pointer to output tensor
        temp: Temperature value
        top_k: k value for top-k sampling
        top_p: p value for top-p sampling
        n_vocab: Size of vocabulary
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Compute offsets
    offset_base = pid * n_vocab
    
    # Load logits
    mask = tl.arange(0, BLOCK_SIZE) < n_vocab
    logits = tl.load(logits_ptr + offset_base + tl.arange(0, BLOCK_SIZE), mask=mask, other=-float('inf'))
    
    # Apply temperature scaling
    logits_scaled = logits / temp
    
    # Compute softmax
    row_max = tl.max(logits_scaled, axis=0)
    row_minus_max = logits_scaled - row_max
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    probs = numerator / denominator
    
    # Sort probabilities in descending order for top-k and top-p
    sorted_probs, sorted_indices = tl.sort(probs, descending=True)
    
    # Apply top-k filtering if k > 0
    if top_k > 0:
        top_k = tl.minimum(top_k, n_vocab)
        # Create k-mask
        k_mask = tl.arange(0, BLOCK_SIZE) < top_k
        sorted_probs = tl.where(k_mask, sorted_probs, 0.0)
    
    # Apply top-p filtering
    # if top_p < 1.0:
    #     # Compute cumulative sum
    #     cumsum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    #     for i in range(BLOCK_SIZE):
    #         if i < n_vocab:
    #             cumsum[i] = cumsum[i-1] + sorted_probs[i] if i > 0 else sorted_probs[i]
        
    #     # Create p-mask based on cumulative sum
    #     p_mask = cumsum <= top_p
    #     sorted_probs = tl.where(p_mask, sorted_probs, 0.0)
    
    # Renormalize probabilities after filtering
    new_sum = tl.sum(sorted_probs, axis=0)
    if new_sum > 0:
        sorted_probs = sorted_probs / new_sum
    
    # Store result (probabilities and their original indices)
    tl.store(out_ptr + offset_base + tl.arange(0, BLOCK_SIZE), sorted_probs, mask=mask)
    # tl.store(out_ptr + offset_base + n_vocab + tl.arange(0, BLOCK_SIZE), sorted_indices.to(tl.float32), mask=mask)

# Wrapper function to call the kernel
def sample_next_token_triton(logits, temperature=1.0, top_k=0, top_p=1.0):
    batch_size, vocab_size = logits.shape
    # Output tensor will contain both probabilities and indices
    output = torch.empty((batch_size, vocab_size * 2), dtype=torch.float32, device=logits.device)
    
    # Configure grid (one block per sequence in batch)
    grid = (batch_size,)
    
    # Launch kernel
    fused_temperature_topk_topp_kernel[grid](
        logits, output, 
        temperature, top_k, top_p,
        vocab_size, triton.next_power_of_2(vocab_size)
    )
    
    # Extract probabilities and indices
    probs = output[:, :vocab_size]
    indices = output[:, vocab_size:2*vocab_size].long()
    
    # Sample from the filtered distribution
    selected_indices = torch.multinomial(probs, num_samples=1)
    
    # Gather the actual token ids
    next_tokens = torch.gather(indices, dim=1, index=selected_indices)
    
    return next_tokens

# Example usage
if __name__ == "__main__":
    # Create some fake logits
    batch_size, vocab_size = 2, 128
    logits = torch.randn((batch_size, vocab_size), device="cuda")
    
    # Sample tokens
    next_tokens = sample_next_token_triton(logits, temperature=0.8, top_k=50, top_p=0.95)
    print(f"Sampled tokens: {next_tokens}")