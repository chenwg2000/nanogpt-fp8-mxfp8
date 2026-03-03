# Specification: Fused MXFP8 Flash Attention with RoPE + QK RMSNorm

## Motivation

In our NanoGPT MXFP8 training pipeline, the pre-attention operations (RoPE + QK RMSNorm + FP8 cast) are executed as 5 separate CUDA kernel launches, costing **0.29 ms/layer** (5.8 ms over 20 layers). The MXFP8 flash attention kernel itself takes only 0.20 ms fwd. Fusing these pre-attention ops into the flash attention kernel prologue would eliminate this overhead, closing the remaining ~9 ms/iter gap to TE's fused BF16 baseline.

### Current per-layer breakdown (Custom MXFP8 Block)

| Operator | Fwd (ms) | Bwd (ms) | Total (ms) |
|---|---|---|---|
| QKV proj (te.LayerNormLinear, FP8 GEMM) | 0.27 | 0.41 | 0.68 |
| **RoPE × 2 + QK RMSNorm × 2 + FP8 cast** | **0.33** | **—** | **0.33** |
| MXFP8 Attention kernel | 0.20 | 0.78 | 0.98 |
| Output proj (te.Linear, FP8 GEMM) | 0.16 | 0.32 | 0.48 |
| MLP (te.LayerNormMLP, FP8 GEMM) | 0.53 | 0.97 | 1.50 |
| **Layer total** | **1.48** | **2.49** | **3.97** |

Target: eliminate the 0.33 ms by fusing RoPE + QK RMSNorm + FP8 cast into the attention kernel.

### Current vs target performance

| Config | Iter time | MFU |
|---|---|---|
| TE baseline (MXFP8 linear + BF16 SDPA) | 113 ms | 61% |
| Current MXFP8 attention (separate pre-attn ops) | 119 ms | 58% |
| **Target: fused MXFP8 attention** | **~113 ms** | **~61%** |

## Current API (unfused)

```python
from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func

# Forward: expects pre-quantized FP8 inputs + UE8M0 scales
out, softmax_lse = flash_attn_mxfp8_func(
    q,                  # (B, T, H, D), float8_e4m3fn
    k,                  # (B, T, H, D), float8_e4m3fn
    v,                  # (B, T, H, D), float8_e4m3fn
    q_scale,            # (B, H, T, D//32), uint8 UE8M0
    k_scale,            # (B, H, T, D//32), uint8 UE8M0
    v_scale,            # (B, H, T, D//32), uint8 UE8M0
    softmax_scale=None, # defaults to 1/sqrt(D)
    causal=False,
)

# Backward: expects FP8 inputs + fwd outputs
dq, dk, dv = flash_attn_mxfp8_bwd_func(
    dout,               # (B, T, H, D), bfloat16
    q, k, v,            # float8_e4m3fn (from forward)
    out,                # (B, T, H, D), bfloat16 (from forward)
    softmax_lse,        # (B, H, T), float32 (from forward)
    q_scale, k_scale,   # uint8 UE8M0
    softmax_scale=None,
    causal=False,
)
```

### Current caller code (train.py)

```python
# Pre-attention: 5 separate kernel launches
q = apply_rotary_pos_emb(q, rotary_emb, tensor_format="bshd")   # kernel 1
k = apply_rotary_pos_emb(k, rotary_emb, tensor_format="bshd")   # kernel 2
q = rms_norm(q)                                                   # kernel 3
k = rms_norm(k)                                                   # kernel 4
q_fp8 = q.to(torch.float8_e4m3fn)                                # kernel 5 (cast)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)
identity_scale = torch.full((B, H, T, D//32), 127, dtype=torch.uint8, ...)

out, lse = flash_attn_mxfp8_func(q_fp8, k_fp8, v_fp8,
                                   identity_scale, identity_scale, identity_scale,
                                   causal=True)
```

## Proposed Fused API

### Forward

```python
def flash_attn_mxfp8_fused_func(
    q,                      # (B, T, H, D), bfloat16 — raw QKV proj output
    k,                      # (B, T, H_kv, D), bfloat16
    v,                      # (B, T, H_kv, D), bfloat16
    cos,                    # (T, 1, 1, D) or (T, D), float32 or bfloat16 — RoPE cos
    sin,                    # (T, 1, 1, D) or (T, D), float32 or bfloat16 — RoPE sin
    q_norm_weight,          # (D,), float32 — QK RMSNorm weight for Q
    k_norm_weight,          # (D,), float32 — QK RMSNorm weight for K
    softmax_scale=None,     # float, defaults to 1/sqrt(D)
    causal=False,           # bool
    norm_eps=1e-6,          # float — RMSNorm epsilon
) -> Tuple[out, softmax_lse, q_fp8, k_fp8, v_fp8]:
    """
    Fused: RoPE(Q,K) → QK_RMSNorm(Q,K) → FP8_cast(Q,K,V) → MXFP8 Flash Attention

    The kernel prologue (per-tile, before the main attention loop):
      1. Load BF16 Q/K tile
      2. Apply rotary position embedding (element-wise cos/sin multiply + add)
      3. Apply RMSNorm per-head (reduce, normalize, scale by weight)
      4. Cast to float8_e4m3fn with identity scales (values are ~unit scale after RMSNorm)
      5. Load BF16 V tile, cast to float8_e4m3fn
      6. Proceed with standard MXFP8 flash attention

    Returns:
      out:          (B, T, H, D), bfloat16 — attention output
      softmax_lse:  (B, H, T), float32 — log-sum-exp for backward
      q_fp8:        (B, T, H, D), float8_e4m3fn — for backward kernel
      k_fp8:        (B, T, H_kv, D), float8_e4m3fn — for backward kernel
      v_fp8:        (B, T, H_kv, D), float8_e4m3fn — for backward kernel
    """
```

### Backward

```python
def flash_attn_mxfp8_fused_bwd_func(
    dout,                   # (B, T, H, D), bfloat16
    q_fp8,                  # (B, T, H, D), float8_e4m3fn — from fwd
    k_fp8,                  # (B, T, H_kv, D), float8_e4m3fn — from fwd
    v_fp8,                  # (B, T, H_kv, D), float8_e4m3fn — from fwd
    out,                    # (B, T, H, D), bfloat16 — from fwd
    softmax_lse,            # (B, H, T), float32 — from fwd
    softmax_scale=None,
    causal=False,
) -> Tuple[dq, dk, dv]:
    """
    Backward pass. Identical to existing flash_attn_mxfp8_bwd_func but with
    identity scales (no scale tensors needed).

    dQ, dK gradients are w.r.t. the post-RoPE, post-RMSNorm Q and K.
    The caller chains these through RMSNorm and RoPE backward via autograd.

    Returns:
      dq: (B, T, H, D), bfloat16
      dk: (B, T, H_kv, D), bfloat16
      dv: (B, T, H_kv, D), bfloat16
    """
```

### Alternative: Full backward fusion (optional, more complex)

If backward fusion is also desired, the backward kernel would need to:
1. Compute dQ, dK w.r.t. FP8 Q, K (as today)
2. Chain through RMSNorm backward: `dq_pre_norm = rms_norm_bwd(dq, q_pre_norm, weight)`
3. Chain through RoPE backward: `dq_pre_rope = rope_bwd(dq_pre_norm, cos, sin)`

This requires saving additional tensors (pre-norm Q/K, pre-RoPE Q/K) and is significantly
more complex. **Recommend starting with forward-only fusion** and letting PyTorch autograd
handle the backward chain through RoPE and RMSNorm.

## RoPE Specification

Standard rotary position embedding, applied per-head along the headdim dimension.

```python
def apply_rotary_pos_emb(x, cos, sin):
    """
    x:   (B, T, H, D), bfloat16
    cos: (T, 1, 1, D//2) or broadcastable, float32
    sin: (T, 1, 1, D//2) or broadcastable, float32

    For each position t, head h:
      x_rot[..., :D//2]  = x[..., :D//2] * cos[t] - x[..., D//2:] * sin[t]
      x_rot[..., D//2:]  = x[..., D//2:] * cos[t] + x[..., :D//2] * sin[t]

    Applied to Q and K only (not V).
    """
```

The cos/sin tables are precomputed by `te.RotaryPositionEmbedding(dim=D)` and have shape
`(T, 1, 1, D)` with the rotation interleaved. The exact layout follows the TE convention
(`tensor_format="bshd"`). See `transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb`.

## QK RMSNorm Specification

Standard RMSNorm applied per-head independently (not across heads).

```python
def rms_norm(x, weight, eps=1e-6):
    """
    x:      (B, T, H, D), bfloat16
    weight: (D,), float32 — learnable per-element scale

    For each (b, t, h):
      rms = sqrt(mean(x^2) + eps)
      out = (x / rms) * weight
    """
```

- Separate weight tensors for Q and K: `q_norm_weight` (D,) and `k_norm_weight` (D,)
- RMSNorm is applied **after** RoPE (matches TE default `qk_norm_before_rope=False`)
- Applied to Q and K only (not V)
- After RMSNorm, values are ~unit scale, so FP8 cast with identity scales is safe

## FP8 Cast Specification

After RMSNorm, values are in approximately [-2, 2] range. Direct cast to float8_e4m3fn
(which can represent up to ±448) with identity MXFP8 scales (UE8M0 = 127 = 2^0 = 1.0).

```python
q_fp8 = q_after_norm.to(torch.float8_e4m3fn)  # direct truncation, no scaling needed
k_fp8 = k_after_norm.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)             # V is not RoPE'd or normed
```

No per-block MXFP8 quantization needed — identity scales throughout. This eliminates the
need to compute/store/pass scale tensors entirely.

## Integration in NanoGPT (target caller code)

```python
class MXFP8Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cos, sin, q_norm_weight, k_norm_weight):
        # q, k, v: (B, T, H, D) in BF16 — raw from QKV projection
        out, softmax_lse, q_fp8, k_fp8, v_fp8 = flash_attn_mxfp8_fused_func(
            q, k, v, cos, sin, q_norm_weight, k_norm_weight,
            causal=True, norm_eps=1e-6,
        )
        # Save for backward
        ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out, softmax_lse)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q_fp8, k_fp8, v_fp8, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = flash_attn_mxfp8_fused_bwd_func(
            grad_output, q_fp8, k_fp8, v_fp8, out, softmax_lse, causal=True)
        # dq, dk are w.r.t. post-norm post-RoPE inputs
        # autograd handles RMSNorm and RoPE backward automatically
        return dq, dk, dv, None, None, None, None


class Block(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, num_attention_heads):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.num_heads = num_attention_heads
        self.qkv_proj = te.LayerNormLinear(hidden_size, 3 * hidden_size, ...)
        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim))
        self.out_proj = te.Linear(hidden_size, hidden_size, ...)
        self.mlp = te.LayerNormMLP(hidden_size, ffn_hidden_size, ...)

    def forward(self, x, rotary_pos_emb=None, is_first_microbatch=None):
        B, T, C = x.shape
        residual = x
        qkv = self.qkv_proj(x, is_first_microbatch=is_first_microbatch)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        # Single fused kernel: RoPE + QK RMSNorm + FP8 cast + MXFP8 attention
        cos, sin = rotary_pos_emb  # precomputed
        attn_out = MXFP8Attention.apply(
            q, k, v, cos, sin, self.q_norm_weight, self.k_norm_weight)

        attn_out = attn_out.reshape(B, T, C)
        attn_out = self.out_proj(attn_out, is_first_microbatch=is_first_microbatch)
        x = residual + attn_out
        residual = x
        mlp_out = self.mlp(x, is_first_microbatch=is_first_microbatch)
        x = residual + mlp_out
        return x
```

## Shapes Summary

Training config: B=4, T=2048, H=10, D=128, H_kv=H (no GQA)

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| q, k (input) | (B, T, H, D) | bfloat16 | Raw QKV proj output |
| v (input) | (B, T, H, D) | bfloat16 | Not RoPE'd or normed |
| cos, sin | (T, 1, 1, D) | float32 | Precomputed RoPE tables |
| q_norm_weight | (D,) | float32 | Learnable RMSNorm scale |
| k_norm_weight | (D,) | float32 | Learnable RMSNorm scale |
| q_fp8, k_fp8, v_fp8 | (B, T, H, D) | float8_e4m3fn | After RoPE+Norm+Cast |
| out | (B, T, H, D) | bfloat16 | Attention output |
| softmax_lse | (B, H, T) | float32 | For backward |
| dout | (B, T, H, D) | bfloat16 | Gradient input |
| dq, dk, dv | (B, T, H, D) | bfloat16 | Gradient output |

## Implementation Notes

### Kernel prologue (forward)

The fusion happens in the **tile loading stage** of the flash attention kernel. For each
Q tile being loaded from global memory:

1. Load BF16 Q/K tile into shared memory / registers
2. Apply RoPE: element-wise multiply-add with cos/sin (position-dependent, load from global)
3. Apply RMSNorm: reduce across D within each (b,t,h), normalize, scale by weight
4. Cast to FP8: direct truncation to float8_e4m3fn (values are ~unit scale after norm)
5. Proceed with standard MXFP8 MMA (matmul-accumulate)

For V tiles: only do step 1 (load) and step 4 (cast). No RoPE or norm on V.

### Why identity scales work

After QK RMSNorm, values have unit RMS norm. The typical range is [-2, 2]. FP8 E4M3 can
represent values up to ±448, so no per-block scaling is needed. Empirically verified:
identity scales produce identical convergence to per-block MXFP8 quantization over 50 training
steps (final loss 6.11 vs 6.05 for TE baseline).

### Backward considerations

The backward kernel operates on the FP8 Q, K, V tensors saved from the forward pass.
It does not need to undo RoPE or RMSNorm — those are handled by PyTorch autograd on the
caller side. The backward kernel signature is identical to the existing `flash_attn_mxfp8_bwd_func`
but without scale tensor arguments (identity scales are hardcoded).

If full backward fusion is desired later (fusing RMSNorm/RoPE backward into the attention
backward), this would require saving pre-RoPE and pre-norm tensors and implementing the
chain rule inside the kernel. This is a separate, more complex optimization.

## Testing

### Correctness test

```python
# Reference: unfused path
q_rope = apply_rotary_pos_emb(q, (cos, sin), tensor_format="bshd")
k_rope = apply_rotary_pos_emb(k, (cos, sin), tensor_format="bshd")
q_normed = rms_norm(q_rope, q_norm_weight)
k_normed = rms_norm(k_rope, k_norm_weight)
q_fp8 = q_normed.to(torch.float8_e4m3fn)
k_fp8 = k_normed.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)
sf = identity_scales(B, H, T, D)
out_ref, lse_ref = flash_attn_mxfp8_func(q_fp8, k_fp8, v_fp8, sf, sf, sf, causal=True)

# Fused path
out_fused, lse_fused, _, _, _ = flash_attn_mxfp8_fused_func(
    q, k, v, cos, sin, q_norm_weight, k_norm_weight, causal=True)

# Should match exactly (same operations, same precision)
assert torch.allclose(out_ref, out_fused, atol=0, rtol=0)
assert torch.allclose(lse_ref, lse_fused, atol=1e-5, rtol=1e-5)
```

### Performance test

```python
# Unfused: 5 kernel launches + attention = ~0.53 ms fwd
# Fused:   1 kernel launch             = ~0.20 ms fwd (target)
# Savings: ~0.33 ms/layer × 20 layers = ~6.6 ms/iter
```

## Summary

| What | Before (unfused) | After (fused) | Saved |
|---|---|---|---|
| Kernel launches per layer (fwd) | 6 (RoPE×2 + Norm×2 + Cast + Attn) | 1 (fused Attn) | 5 launches |
| Pre-attention fwd time | 0.33 ms/layer | 0 ms (absorbed) | 0.33 ms/layer |
| Training iter time (20 layers) | ~119 ms | ~113 ms (target) | ~6 ms |
| MFU | 58% | ~61% (target) | +3% |
