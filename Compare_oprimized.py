import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import time
import numpy as np
import argparse
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from typing import Tuple
from pathlib import Path
from sparktts.utils.token_parser import TASK_TOKEN_MAP
import argparse
from kernels.RMSNorm_triton import custom_rms_forward_optimized
from kernels.fused_mlp import custom_mlp_forward_optimized


def process_prompt(
    audio_tokenizer,
    text: str,
    prompt_speech_path: Path,
    prompt_text: str = None,
) -> Tuple[str, torch.Tensor]:
    """
    Process input for voice cloning.

    Args:
        audio_tokenizer: Tokenizer object
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


def main(args):
    device = torch.device("cuda:0")
    
    def warmup_triton():
        from kernels.RMSNorm_triton import rmsnorm
        from kernels.fused_mlp import fused_silu_gate_up

        print("Warming up Triton kernel to avoid compilation overhead in benchmarks...")
        sizes_rms = [(1, 896), (8, 896), (16, 896)]
        for size in sizes_rms:
            dummy_input = torch.randn(*size, device=device)
            dummy_weight = torch.randn(size[-1], device=device)

            _ = rmsnorm(dummy_input, dummy_weight)
        sizes_mlp = [(1, 50, 896, 4864), (8, 100, 896, 4864), (16, 48, 896, 4864)]
        for batch_size, seq_len, hidden_size,intermediate_size in sizes_mlp:
            x = torch.randn(batch_size, seq_len, hidden_size, device=device)
            gate_weight = torch.randn(intermediate_size, hidden_size, device=device)
            up_weight = torch.randn(intermediate_size, hidden_size, device=device)
            
            _ = fused_silu_gate_up(x, gate_weight, up_weight)
        print("Warmup complete!")
    
    # Run the warmup
    warmup_triton()
    
    # Benchmark original PyTorch implementation
    print("Benchmarking original PyTorch implementation...")
    model = AutoModelForCausalLM.from_pretrained(f"{args.model_dir}/LLM", torch_dtype="bfloat16", _attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_dir}/LLM")
    audio_tokenizer = BiCodecTokenizer(args.model_dir, device=device)
    # text = "Hi! How are you?"
    text = '''The wind whispered through the ancient trees, their gnarled branches reaching toward the sky like the fingers of forgotten giants. A narrow dirt path wound through the dense forest, illuminated only by slivers of moonlight breaking through the thick canopy above. Somewhere in the distance, an owl hooted—a lone sentinel of the night.

Emma pulled her coat tighter around her as she walked, her breath forming small clouds in the crisp autumn air. She wasn’t sure why she had taken this path, only that something deep inside urged her forward. The old stories spoke of a hidden place, one that only revealed itself to those who truly needed it. A place of answers.

Just as she was about to turn back, the trees parted before her, revealing a clearing bathed in silver light. In the center stood an old wooden door—standing alone, unattached to any wall or building. Emma's heart pounded as she reached out a trembling hand. The handle was cool beneath her fingers. With a deep breath, she turned it.

And the door opened.
The wind whispered through the ancient trees, their gnarled branches reaching toward the sky like the fingers of forgotten giants. A narrow dirt path wound through the dense forest, illuminated only by slivers of moonlight breaking through the thick canopy above. Somewhere in the distance, an owl hooted—a lone sentinel of the night.

Emma pulled her coat tighter around her as she walked, her breath forming small clouds in the crisp autumn air. She wasn’t sure why she had taken this path, only that something deep inside urged her forward. The old stories spoke of a hidden place, one that only revealed itself to those who truly needed it. A place of answers.

Just as she was about to turn back, the trees parted before her, revealing a clearing bathed in silver light. In the center stood an old wooden door—standing alone, unattached to any wall or building. Emma's heart pounded as she reached out a trembling hand. The handle was cool beneath her fingers. With a deep breath, she turned it.

And the door opened.
The wind whispered through the ancient trees, their gnarled branches reaching toward the sky like the fingers of forgotten giants. A narrow dirt path wound through the dense forest, illuminated only by slivers of moonlight breaking through the thick canopy above. Somewhere in the distance, an owl hooted—a lone sentinel of the night.

Emma pulled her coat tighter around her as she walked, her breath forming small clouds in the crisp autumn air. She wasn’t sure why she had taken this path, only that something deep inside urged her forward. The old stories spoke of a hidden place, one that only revealed itself to those who truly needed it. A place of answers.

Just as she was about to turn back, the trees parted before her, revealing a clearing bathed in silver light. In the center stood an old wooden door—standing alone, unattached to any wall or building. Emma's heart pounded as she reached out a trembling hand. The handle was cool beneath her fingers. With a deep breath, she turned it.

And the door opened.
The wind whispered through the ancient trees, their gnarled branches reaching toward the sky like the fingers of forgotten giants. A narrow dirt path wound through the dense forest, illuminated only by slivers of moonlight breaking through the thick canopy above. Somewhere in the distance, an owl hooted—a lone sentinel of the night.

Emma pulled her coat tighter around her as she walked, her breath forming small clouds in the crisp autumn air. She wasn’t sure why she had taken this path, only that something deep inside urged her forward. The old stories spoke of a hidden place, one that only revealed itself to those who truly needed it. A place of answers.

Just as she was about to turn back, the trees parted before her, revealing a clearing bathed in silver light. In the center stood an old wooden door—standing alone, unattached to any wall or building. Emma's heart pounded as she reached out a trembling hand. The handle was cool beneath her fingers. With a deep breath, she turned it.

And the door opened.'''
    prompt_speech_path = "example/prompt_audio.wav"

    model.to(device)
    
    # Save the original forward method
    original_forward_rms = Qwen2RMSNorm.forward
    original_forward_mlp = Qwen2MLP.forward
    
    og_time_list = []
    if args.profile_imp:
         with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=tensorboard_trace_handler('./profiler/tokenizer')):
            prompt, global_token_ids = process_prompt(audio_tokenizer, text, prompt_speech_path)
            model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    else:
        st = time.time()
        prompt, global_token_ids = process_prompt(audio_tokenizer, text, prompt_speech_path)
        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
        tokenizer_execution_time = time.time() - st
    if args.profile_imp:
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=3, active=5),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=tensorboard_trace_handler('./profiler/forwardpass_pytorch')
            ) as prof:
            with torch.no_grad():
                torch.cuda.synchronize()
                for i in range(15):
                    # ip = torch.randint(low=0, high=166000, size=(1, seq_len), device=device).cuda()
                    
                    torch.cuda.synchronize()
                    st = time.time()
                    _ = model(model_inputs['input_ids'])
                    # _ = model(ip)
                    torch.cuda.synchronize()
                    execution_time = time.time() - st
                    
                    if i >= 5:
                        og_time_list.append(execution_time)
                    prof.step()
    else:
        with torch.no_grad():
                torch.cuda.synchronize()
                for i in range(15):
                    # ip = torch.randint(low=0, high=166000, size=(1, seq_len), device=device).cuda()
                    
                    torch.cuda.synchronize()
                    st = time.time()
                    _ = model(model_inputs['input_ids'])
                    # _ = model(ip)
                    torch.cuda.synchronize()
                    execution_time = time.time() - st
                    
                    if i >= 5:
                        og_time_list.append(execution_time)
    
    # Benchmark our optimized Triton implementation
    print("Benchmarking optimized Triton implementation...")
    Qwen2RMSNorm.forward = custom_rms_forward_optimized
    Qwen2MLP.forward = custom_mlp_forward_optimized
    model = AutoModelForCausalLM.from_pretrained(f"{args.model_dir}/LLM", torch_dtype="bfloat16", _attn_implementation="flash_attention_2")
    model.to(device)
    
    triton_time_list = []
    if args.profile_imp:
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=3, active=5),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=tensorboard_trace_handler('./profiler/forwardpass_triton_optimized')
            ) as prof:
            with torch.no_grad():
                torch.cuda.synchronize()
                for i in range(15):
                    # ip = torch.randint(low=0, high=166000, size=(1, seq_len), device=device).cuda()
                    
                    torch.cuda.synchronize()
                    st = time.time()
                    _ = model(model_inputs['input_ids'])
                    # _ = model(ip)
                    torch.cuda.synchronize()
                    execution_time = time.time() - st
                    
                    if i >= 5:
                        triton_time_list.append(execution_time)
                    prof.step()
    else:
        with torch.no_grad():
                torch.cuda.synchronize()
                for i in range(15):
                    # ip = torch.randint(low=0, high=166000, size=(1, seq_len), device=device).cuda()
                    
                    torch.cuda.synchronize()
                    st = time.time()
                    _ = model(model_inputs['input_ids'])
                    # _ = model(ip)
                    torch.cuda.synchronize()
                    execution_time = time.time() - st
                    
                    if i >= 5:
                        triton_time_list.append(execution_time)

    Qwen2RMSNorm.forward = original_forward_rms
    Qwen2MLP.forward = original_forward_mlp
    
    # Print results with statistics
    def print_stats(name, times):
        avg = np.mean(times)
        std = np.std(times)
        min_t = np.min(times)
        max_t = np.max(times)
        print(f"{name} - avg: {avg:.4f}s, std: {std:.4f}s, min: {min_t:.4f}s, max: {max_t:.4f}s")
    if not args.profile_imp:
        print("\nPerformance Results:")
        print("Tokenizer time", tokenizer_execution_time)
        print_stats("PyTorch Implementation", og_time_list)
        print_stats("Optimized Triton Implementation", triton_time_list)
        
        # Calculate speedups
        pt_avg = np.mean(og_time_list)
        triton_avg = np.mean(triton_time_list)
        
        print(f"\nSpeed Comparison:")
        print(f"Optimized Triton vs PyTorch: {pt_avg/triton_avg:.2f}x speedup")
        return pt_avg/triton_avg, pt_avg, triton_avg
    else:
        print("Profiling done")
        return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="/home/vishwa/small_projects/pretrained_model", help='path to the model')
    parser.add_argument('--profile_imp', type=bool, default=False, help='profile?')
    args = parser.parse_args()
    main(args)

# [500, 1000, 1500, 2000, 2500]
# [1.1790394150302708, 1.698469844530697, 2.234700139957699, 1.8123769616315415, 1.7236127499987484]
# [0.04965386390686035, 0.11882259845733642, 0.23769795894622803, 0.39529168605804443, 0.5909201145172119]
# [0.04211382865905762, 0.06995861530303955, 0.10636682510375976, 0.2181067705154419, 0.34283809661865233]
