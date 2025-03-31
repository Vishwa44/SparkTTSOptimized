import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton
import triton.language as tl
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP
from typing import Tuple
from pathlib import Path
import re
import soundfile as sf

def process_prompt(
    text: str,
    audio_tokenizer,
    prompt_speech_path: Path,
    prompt_text: str = None,
) -> Tuple[str, torch.Tensor]:
    """
    Process input for voice cloning.

    Args:
        text (str): The text input to be converted to speech.
        prompt_speech_path (Path): Path to the audio file used as a prompt.
        prompt_text (str, optional): Transcript of the prompt audio.

    Return:
        Tuple[str, torch.Tensor]: Input prompt; global tokens
    """

    global_token_ids, semantic_token_ids = audio_tokenizer.tokenize(
        prompt_speech_path
    )
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    # Prepare the input tokens for the model
    if prompt_text is not None:
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            prompt_text,
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
        ]
    else:
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]

    inputs = "".join(inputs)

    return inputs, global_token_ids


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
    
    # # Apply top-k
    # if top_k > 0:
    #     # Sort logits in descending order
    #     sorted_logits, _ = tl.sort(logits_scaled, descending=True)
        
    #     # Get k-th largest value as threshold (if k < vocab size)
    #     k = tl.minimum(top_k, n_vocab)
    #     threshold = tl.load(sorted_logits + k - 1)  # k-1 for 0-based indexing
        
    #     # Apply mask for values below threshold
    #     logits_scaled = tl.where(logits_scaled >= threshold, logits_scaled, -float('inf'))
    
    # Apply softmax
    logits_exp = tl.exp(logits_scaled - tl.max(logits_scaled, axis=0))
    logits_sum = tl.sum(logits_exp, axis=0)
    probs = logits_exp / logits_sum
    
    # Apply top-p
    # Sort probabilities (expensive in Triton, in practice would use more optimized approach)
    sorted_probs = tl.sort(probs, descending=True)
    cumsum = tl.cumsum(sorted_probs, axis=0)
    
    # Create mask where cumsum > top_p
    topp_mask = cumsum <= top_p
        

    
    # Store result
    tl.store(out_ptr + offset_base + tl.arange(0, BLOCK_SIZE), sorted_probs, mask=topp_mask)

# Wrapper function to call the kernel
def sample_next_token_triton(logits, temperature=1.0, top_k=0, top_p=1.0):
    vocab_size = len(logits)
    output = torch.empty_like(logits)
    
    # Configure grid
    grid = (1,)
    
    # Launch kernel
    fused_temperature_topk_topp_kernel[grid](
        logits, output, 
        temperature, top_k, top_p,
        vocab_size, triton.next_power_of_2(vocab_size)
    )

    samples = torch.multinomial(output, num_samples=1)
    
    return samples

def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    # Apply temperature scaling
    if temperature > 0:
        logits = logits / temperature
    
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Get sorted indices of probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Apply top-k filtering if k > 0
    if top_k > 0:
        # Keep only top-k tokens
        sorted_probs = sorted_probs[:top_k]
        sorted_indices = sorted_indices[:top_k]
    
    # Apply top-p (nucleus) sampling
    if top_p < 1.0:
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep tokens with cumulative probability <= to threshold
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False  # Always keep the top token
        
        # Filter the distributions
        indices_to_keep = sorted_indices[~sorted_indices_to_remove]
        filtered_probs = sorted_probs[~sorted_indices_to_remove]
        
        # Renormalize the probabilities
        filtered_probs = filtered_probs / filtered_probs.sum()
    else:
        indices_to_keep = sorted_indices
        filtered_probs = sorted_probs
    
    # Sample from the filtered distribution
    sampled_idx = torch.multinomial(filtered_probs, num_samples=1, replacement=True)
    
    # Get the token ID
    next_token_id = indices_to_keep[sampled_idx]
    
    return next_token_id.item()

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_k=0,
    top_p=0.9,
    triton_sample=False
):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Move to the same device as the model
    input_ids = input_ids.to(model.device)
    
    # Initialize the generated token list with input tokens
    generated_ids = input_ids[0].tolist()
    print("starting generation")
    # Auto-regressive generation
    for _ in range(max_new_tokens):
        input_sequence = torch.tensor([generated_ids], device=model.device)
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = model(input_sequence)
            
        # Get the logits for the next token (last position in the sequence)
        next_token_logits = outputs.logits[0, -1, :]
        
        # Sample the next token using our custom sampling function
        if triton_sample:
            next_token_id = sample_next_token_triton(next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
                )
        else:
            next_token_id = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Add the sampled token to the generated sequence
        generated_ids.append(next_token_id)
        
        # Check if generation should stop (EOS token)
        if next_token_id == tokenizer.eos_token_id:
            print("Breaking")
            break
    
    print("Generation done")
    # generated_ids = [
    #     output_ids[len(input_ids) :]
    #     for input_ids, output_ids in zip(input_ids, generated_ids)
    # ]

    predicts = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return predicts

def main():
    print("Starting soon")
    model_dir = "/home/vishwa/small_projects/pretrained_model"
    device = torch.device("cuda:0")
    text = "Hi! whata re you doing, can you explain me what is going on?"
    prompt_speech_path = "example/prompt_audio.wav"
    save_path = "output/effienct_generation.wav"
    temperature = 0.8
    top_k = 50
    top_p = 0.95
    
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/LLM")
    model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/LLM", torch_dtype="float32", _attn_implementation="sdpa")
    audio_tokenizer = BiCodecTokenizer(model_dir, device=device)
    model.to(device)
    print("Loaded the model and the tokenizer")    
    
    prompt, global_token_ids = process_prompt(text, audio_tokenizer, prompt_speech_path)
    
    predicts = generate_text(
        model, tokenizer, prompt, temperature=temperature, top_k=top_k, top_p=top_p, triton_sample=True
    )
    print(predicts)
    
    # Extract semantic token IDs from the generated text
    pred_semantic_ids = (
        torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
        .long()
        .unsqueeze(0)
    )


    # Convert semantic tokens back to waveform
    wav = audio_tokenizer.detokenize(
        global_token_ids.to(device).squeeze(0),
        pred_semantic_ids.to(device),
    )
    sf.write(save_path, wav, samplerate=16000)

if __name__ == "__main__":
    main()