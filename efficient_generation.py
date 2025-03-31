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
import argparse

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


@triton.jit
def softmax_kernel(
    output_ptr, 
    logits_ptr, 
    n_vocab,
    temperature,
    BLOCK_SIZE: tl.constexpr
):
    block_offsets = tl.arange(0, BLOCK_SIZE)
    
    max_val = float("-inf")
    
    for i in range(0, n_vocab, BLOCK_SIZE):
        valid_items = tl.minimum(BLOCK_SIZE, n_vocab - i)
        mask = block_offsets < valid_items
        block_logits = tl.load(logits_ptr + i + block_offsets, mask=mask, other=float("-inf"))
        block_max = tl.max(block_logits, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    max_val = tl.broadcast_to(max_val, (BLOCK_SIZE,))
    
    sum_exp = 0.0
    for i in range(0, n_vocab, BLOCK_SIZE):
        valid_items = tl.minimum(BLOCK_SIZE, n_vocab - i)
        mask = block_offsets < valid_items
        block_logits = tl.load(logits_ptr + i + block_offsets, mask=mask, other=float("-inf"))
        scaled_logits = block_logits / temperature
        block_exp = tl.exp(scaled_logits - max_val)
        sum_exp += tl.sum(block_exp, axis=0)
        tl.store(output_ptr + i + block_offsets, block_exp, mask=mask)
    
    sum_exp = tl.broadcast_to(sum_exp, (BLOCK_SIZE,))
    
    for i in range(0, n_vocab, BLOCK_SIZE):
        valid_items = tl.minimum(BLOCK_SIZE, n_vocab - i)
        mask = block_offsets < valid_items
        block_probs = tl.load(output_ptr + i + block_offsets, mask=mask, other=0.0)
        block_probs = block_probs / sum_exp
        tl.store(output_ptr + i + block_offsets, block_probs, mask=mask)


def softmax_with_temperature(logits, temperature):
    if not logits.is_cuda:
        logits = logits.cuda()
        
    n_vocab = logits.shape[0]
    
    probs = torch.empty_like(logits)
    
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_vocab))
    
    softmax_kernel[(1,)](
        probs, 
        logits, 
        n_vocab,
        temperature,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return probs

def sample_next_token_triton(logits, temperature=1.0, top_k=0, top_p=1.0):
    
    output = softmax_with_temperature(logits, temperature)

    sorted_probs, sorted_indices = torch.sort(output, descending=True)
    
    if top_k > 0:
        sorted_probs = sorted_probs[:top_k]
        sorted_indices = sorted_indices[:top_k]
    
    if top_p < 1.0:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        indices_to_keep = sorted_indices[~sorted_indices_to_remove]
        filtered_probs = sorted_probs[~sorted_indices_to_remove]
        
        filtered_probs = filtered_probs / filtered_probs.sum()
    else:
        indices_to_keep = sorted_indices
        filtered_probs = sorted_probs

    sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
    
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
):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    input_ids = input_ids.to(model.device)
    
    generated_ids = input_ids[0].tolist()
    print("starting generation")
    for _ in range(max_new_tokens):
        input_sequence = torch.tensor([generated_ids], device=model.device)
        
        with torch.no_grad():
            outputs = model(input_sequence)
            
        next_token_logits = outputs.logits[0, -1, :]
        
        next_token_id = sample_next_token_triton(next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
            )

        generated_ids.append(next_token_id)
        
        if next_token_id == tokenizer.eos_token_id:
            print("Breaking")
            break
    
    print("Generation done")

    predicts = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return predicts

def main(args):
    print("Starting soon")

    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_dir}/LLM")
    model = AutoModelForCausalLM.from_pretrained(f"{args.model_dir}/LLM", torch_dtype="bfloat16", _attn_implementation="flash_attention_2")
    audio_tokenizer = BiCodecTokenizer(args.model_dir, device=device)
    model.to(device)
    print("Loaded the model and the tokenizer")    
    
    prompt, global_token_ids = process_prompt(args.text, audio_tokenizer, args.prompt_speech_path)
    
    predicts = generate_text(
        model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(predicts)
    
    pred_semantic_ids = (
        torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
        .long()
        .unsqueeze(0)
    )


    wav = audio_tokenizer.detokenize(
        global_token_ids.to(device).squeeze(0),
        pred_semantic_ids.to(device),
    )
    sf.write(args.save_path, wav, samplerate=16000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default="Hi! what are you doing? can you explain me what is going on?")
    parser.add_argument('--model_dir', type=str, default="/home/vishwa/small_projects/pretrained_model", help='path to the model')
    parser.add_argument('--prompt_speech_path', type=str, default="example/prompt_audio.wav")
    parser.add_argument('--save_path', type=str, default="output/effienct_generation.wav", help='path to the output audio')
    parser.add_argument('--temperature', type=float, default=0.8, help="temperature")
    parser.add_argument('--top_k', type=int, default=50, help='topK')
    parser.add_argument('--top_p', type=float, default=0.95, help='topP')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Max new token')
    args = parser.parse_args()
    main(args)