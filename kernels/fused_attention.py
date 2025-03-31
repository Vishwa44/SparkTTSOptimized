import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple, Dict, Callable, Any
from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
from transformers import AutoModelForCausalLM
import math
import time

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float32)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None,
):
    """
    Compute flash attention using a simplified Triton kernel.
    
    Args:
        q: query tensor [batch_size, num_heads, seq_len, head_dim]
        k: key tensor [batch_size, num_heads, seq_len, head_dim]
        v: value tensor [batch_size, num_heads, seq_len, head_dim]
        causal: whether to apply causal masking
        sm_scale: scaling factor (default: 1/sqrt(head_dim))
        
    Returns:
        output: attention output [batch_size, num_heads, seq_len, head_dim]
    """
    # Extract dimensions
    batch_size, num_heads, seq_len, head_dim = q.shape
    # Set default scale if not provided
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    
    # Create output tensor
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    stage = 3 if causal else 1
    # Set block sizes - these are now fixed rather than autotuned
    BLOCK_M = 64   # block size for seq_len (queries)
    BLOCK_N = 64   # block size for seq_len (keys/values)
    BLOCK_D = min(128, triton.next_power_of_2(head_dim))  # block size for head_dim
    
    # Fixed number of warps
    num_warps = 4
    
    # Prepare the grid for kernel launch
    grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    _attn_fwd[grid](
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        HEAD_DIM=head_dim,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        STAGE=stage)

    return o


def triton_flash_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scaling: float = 1.0,
    dropout: float = 0.0,
    sliding_window: Optional[int] = None,
    **kwargs,
):
    """
    Drop-in replacement for eager_attention_forward using Flash Attention via Triton.
    
    Args:
        module: The attention module (used to get num_key_value_groups)
        query: [batch_size, num_heads, seq_len, head_dim]
        key: [batch_size, num_kv_heads, seq_len, head_dim]
        value: [batch_size, num_kv_heads, seq_len, head_dim]
        attention_mask: Optional attention mask
        scaling: Scaling factor for attention scores
        dropout: Dropout probability (not implemented in this version)
        sliding_window: Optional sliding window size (not implemented in this version)
        **kwargs: Additional arguments
    
    Returns:
        output: [batch_size, seq_len, num_heads, head_dim]
        None: Attention weights (not computed)
    """
    # For now, we're ignoring attention_mask, dropout, and sliding_window
    # This is a simplified implementation of Flash Attention
    
    # Call the flash attention implementation
    output = flash_attention(
        q=query,
        k=key,
        v=value,
        sm_scale=scaling,
        causal=True,  # Most LLMs use causal attention
    )
    
    # Transpose output to match expected format [batch, seq_len, num_heads, head_dim]
    output = output.transpose(1, 2).contiguous()
    
    # Flash attention doesn't compute attention weights
    attn_weights = None
    
    return output, attn_weights


def patch_qwen2_with_flash_attention():
    """
    Patch the Qwen2 model to use Flash Attention
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS
        
        # Add Flash Attention to the dictionary of attention functions
        ALL_ATTENTION_FUNCTIONS["flash"] = triton_flash_attention
        
        print("Successfully patched Qwen2Attention with Flash Attention")
        return True
    except ImportError:
        print("Could not import Qwen2 model. Make sure transformers is installed.")
        return False


def use_flash_attention_for_model(model):
    """
    Set a specific model instance to use Flash Attention
    """
    # Ensure flash attention is registered
    if not patch_qwen2_with_flash_attention():
        return False
    
    # Set the model to use flash attention
    success = False
    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
            module.config._attn_implementation = "flash"
            success = True
    
    if success:
        print("Successfully set model to use Flash Attention")
    else:
        print("Could not find any compatible modules to set Flash Attention")
    
    return success

if __name__ == "__main__":
    model_dir = "/home/vishwa/small_projects/pretrained_model"
    device = torch.device("cuda:0")
    ip = torch.randint(low = 0, high=166000,size=(1,1000)).to(device)
    patch_qwen2_with_flash_attention()
    model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/LLM", torch_dtype="float32", _attn_implementation="sdpa")
    model.to(device)
    torch_time = []
    for i in range(10):
        st = time.time()
        generated_ids = model(ip)
        en = time.time()
        torch_time.append(en - st)
    print(sum(torch_time)/10)
    use_flash_attention_for_model(model)
    triton_time = []
    for i in range(10):
        st = time.time()
        generated_ids = model(ip)
        en = time.time()
        triton_time.append(en - st)
    print(sum(triton_time)/10)