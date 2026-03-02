"""
MXFP8 Flash Attention Performance Analysis

Three-way comparison:
  1. Custom Block   — MXFP8 flash attention fwd + BF16 SDPA bwd, TE linear layers
  2. TE MXFP8       — te.TransformerLayer with MXFP8BlockScaling recipe
  3. TE Float8Block  — te.TransformerLayer with Float8BlockScaling recipe

Part 1: Operator-level timing breakdown (per-layer fwd + bwd)
Part 2: Head-to-head operator unit tests
Part 3: Summary output
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling, MXFP8BlockScaling

sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")
from flash_attn_interface import flash_attn_mxfp8_func
from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

torch.manual_seed(42)
device = "cuda"

# ── Shapes matching training config ──────────────────────────────────────
B = 4           # batch size
T = 2048        # sequence length
N_HEAD = 10     # num attention heads
D = 128         # head dim
C = N_HEAD * D  # hidden size = 1280
FFN = 4 * C     # ffn hidden size = 5120

WARMUP = 5
TIMED = 20


# ═════════════════════════════════════════════════════════════════════════
# Inline copies from train.py (avoid module-level side effects)
# ═════════════════════════════════════════════════════════════════════════

def te_init_method(weight):
    fan_out, fan_in = weight.size(0), weight.size(1)
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    torch.nn.init.normal_(weight, mean=0.0, std=std)

def te_output_layer_init_method(weight):
    torch.nn.init.zeros_(weight)


def quantize_to_mxfp8(x):
    """BF16 (B,T,H,D) -> float8_e4m3fn data + uint8 UE8M0 scales."""
    B, T, H, D = x.shape
    x_blocks = x.reshape(B, T, H, D // 32, 32)
    amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(amax)).to(torch.int32) + 127 - 8
    shared_exp = shared_exp.clamp(0, 255)
    scale = torch.exp2((127 - shared_exp).to(torch.float32))
    x_scaled = (x_blocks.float() * scale.unsqueeze(-1)).clamp(-448, 448)
    x_fp8 = x_scaled.reshape(B, T, H, D).to(torch.float8_e4m3fn)
    scale_uint8 = shared_exp.to(torch.uint8).permute(0, 2, 1, 3)
    return x_fp8, scale_uint8


class MXFP8Attention(torch.autograd.Function):
    """MXFP8 flash attention forward, BF16 SDPA backward."""
    @staticmethod
    def forward(ctx, q, k, v):
        q_fp8, q_scale = quantize_to_mxfp8(q)
        k_fp8, k_scale = quantize_to_mxfp8(k)
        v_fp8, v_scale = quantize_to_mxfp8(v)
        out, _ = flash_attn_mxfp8_func(
            q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal=True)
        ctx.save_for_backward(q, k, v)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v = ctx.saved_tensors
        q_t = q.transpose(1, 2).detach().requires_grad_(True)
        k_t = k.transpose(1, 2).detach().requires_grad_(True)
        v_t = v.transpose(1, 2).detach().requires_grad_(True)
        grad_output_t = grad_output.transpose(1, 2)
        with torch.enable_grad():
            attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
            attn_out.backward(grad_output_t)
        return q_t.grad.transpose(1, 2), k_t.grad.transpose(1, 2), v_t.grad.transpose(1, 2)


class Block(nn.Module):
    """Custom transformer block: MXFP8 flash attention + TE linear layers."""
    def __init__(self, hidden_size, ffn_hidden_size, num_attention_heads):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.num_heads = num_attention_heads
        self.qkv_proj = te.LayerNormLinear(
            hidden_size, 3 * hidden_size, normalization="RMSNorm",
            bias=False, init_method=te_init_method)
        self.q_norm = te.RMSNorm(self.head_dim)
        self.k_norm = te.RMSNorm(self.head_dim)
        self.out_proj = te.Linear(
            hidden_size, hidden_size, bias=False,
            init_method=te_output_layer_init_method)
        self.mlp = te.LayerNormMLP(
            hidden_size, ffn_hidden_size, normalization="RMSNorm",
            activation="srelu", bias=False, init_method=te_init_method,
            output_layer_init_method=te_output_layer_init_method)

    def forward(self, x, rotary_pos_emb=None, is_first_microbatch=None):
        B, T, C = x.shape
        residual = x
        qkv = self.qkv_proj(x, is_first_microbatch=is_first_microbatch)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb(q, rotary_pos_emb, tensor_format="bshd")
            k = apply_rotary_pos_emb(k, rotary_pos_emb, tensor_format="bshd")
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_out = MXFP8Attention.apply(q, k, v)
        attn_out = attn_out.reshape(B, T, C)
        attn_out = self.out_proj(attn_out, is_first_microbatch=is_first_microbatch)
        x = residual + attn_out
        residual = x
        mlp_out = self.mlp(x, is_first_microbatch=is_first_microbatch)
        x = residual + mlp_out
        return x


# ═════════════════════════════════════════════════════════════════════════
# Timing helpers
# ═════════════════════════════════════════════════════════════════════════

def cuda_timer(fn, warmup=WARMUP, iters=TIMED):
    """Time a callable using CUDA events. Returns (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    import numpy as np
    return float(np.mean(times)), float(np.std(times))


def fmt_ms(mean, std=None):
    if std is not None:
        return f"{mean:7.2f} +/- {std:.2f}"
    return f"{mean:7.2f}"


# ═════════════════════════════════════════════════════════════════════════
# Part 1: Operator-Level Timing Breakdown
# ═════════════════════════════════════════════════════════════════════════

print("=" * 80)
print(" PART 1: OPERATOR-LEVEL TIMING BREAKDOWN (single layer, fwd + bwd)")
print("=" * 80)
print(f"  Shapes: B={B}, T={T}, H={N_HEAD}, D={D}, C={C}, FFN={FFN}")
print(f"  Warmup={WARMUP}, Timed={TIMED}\n")

recipe_mxfp8 = MXFP8BlockScaling()
recipe_f8blk = Float8BlockScaling()
rope = te.RotaryPositionEmbedding(dim=D, pretrained_max_position_embeddings=T)
rotary_emb = rope(T)

# ── 1a. Custom MXFP8 Block — per-operator breakdown ─────────────────────

print("--- Custom MXFP8 Block: per-operator breakdown ---\n")

block = Block(C, FFN, N_HEAD).to(device, dtype=torch.bfloat16)
block.train()

# We'll time each operator in the forward pass individually,
# then time the full backward pass, then the full fwd+bwd end-to-end.

x_input = torch.randn(B, T, C, device=device, dtype=torch.bfloat16)

# Operator timings (forward only — each measured independently)
op_fwd_times = {}

# 1. QKV projection (LayerNormLinear)
def time_qkv_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_mxfp8):
        _ = block.qkv_proj(x_input, is_first_microbatch=True)
op_fwd_times["1. QKV proj (LayerNormLinear)"] = cuda_timer(time_qkv_fwd)

# Prepare QKV output for subsequent ops
with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
     te.autocast(enabled=True, recipe=recipe_mxfp8):
    qkv_out = block.qkv_proj(x_input, is_first_microbatch=True)
q_raw, k_raw, v_raw = qkv_out.chunk(3, dim=-1)
q_4d = q_raw.reshape(B, T, N_HEAD, D)
k_4d = k_raw.reshape(B, T, N_HEAD, D)
v_4d = v_raw.reshape(B, T, N_HEAD, D)

# 2. RoPE (apply_rotary_pos_emb x2)
def time_rope_fwd():
    _ = apply_rotary_pos_emb(q_4d, rotary_emb, tensor_format="bshd")
    _ = apply_rotary_pos_emb(k_4d, rotary_emb, tensor_format="bshd")
op_fwd_times["2. RoPE (x2)"] = cuda_timer(time_rope_fwd)

q_rope = apply_rotary_pos_emb(q_4d, rotary_emb, tensor_format="bshd")
k_rope = apply_rotary_pos_emb(k_4d, rotary_emb, tensor_format="bshd")

# 3. QK RMSNorm (x2)
def time_qknorm_fwd():
    _ = block.q_norm(q_rope)
    _ = block.k_norm(k_rope)
op_fwd_times["3. QK RMSNorm (x2)"] = cuda_timer(time_qknorm_fwd)

q_normed = block.q_norm(q_rope)
k_normed = block.k_norm(k_rope)

# 4a. quantize_to_mxfp8 (x3)
def time_quantize_fwd():
    _ = quantize_to_mxfp8(q_normed)
    _ = quantize_to_mxfp8(k_normed)
    _ = quantize_to_mxfp8(v_4d)
op_fwd_times["4a. quantize_to_mxfp8 (x3)"] = cuda_timer(time_quantize_fwd)

q_fp8, q_scale = quantize_to_mxfp8(q_normed)
k_fp8, k_scale = quantize_to_mxfp8(k_normed)
v_fp8, v_scale = quantize_to_mxfp8(v_4d)

# 4b. flash_attn_mxfp8_func (MXFP8 kernel)
def time_flash_fwd():
    _ = flash_attn_mxfp8_func(
        q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal=True)
op_fwd_times["4b. flash_attn_mxfp8 kernel"] = cuda_timer(time_flash_fwd)

attn_out_4d, _ = flash_attn_mxfp8_func(
    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal=True)
attn_out_3d = attn_out_4d.reshape(B, T, C)

# 5. Output projection (te.Linear)
def time_outproj_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_mxfp8):
        _ = block.out_proj(attn_out_3d, is_first_microbatch=True)
op_fwd_times["5. Output proj (te.Linear)"] = cuda_timer(time_outproj_fwd)

with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
     te.autocast(enabled=True, recipe=recipe_mxfp8):
    outproj_out = block.out_proj(attn_out_3d, is_first_microbatch=True)
after_attn_residual = x_input + outproj_out

# 6. LayerNormMLP
def time_mlp_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_mxfp8):
        _ = block.mlp(after_attn_residual, is_first_microbatch=True)
op_fwd_times["6. LayerNormMLP"] = cuda_timer(time_mlp_fwd)

# 7. Residual adds (x2) — very cheap, measure anyway
res_a = torch.randn_like(x_input)
res_b = torch.randn_like(x_input)
def time_residuals():
    _ = x_input + res_a
    _ = x_input + res_b
op_fwd_times["7. Residual adds (x2)"] = cuda_timer(time_residuals)

# Full fwd pass
def custom_block_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_mxfp8):
        _ = block(x_input, rotary_pos_emb=rotary_emb, is_first_microbatch=True)
custom_fwd_time = cuda_timer(custom_block_fwd)

# Full bwd pass (need grad-enabled fwd first)
def custom_block_fwd_bwd():
    x_in = x_input.detach().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_mxfp8):
        out = block(x_in, rotary_pos_emb=rotary_emb, is_first_microbatch=True)
    out.sum().backward()
custom_fwdbwd_time = cuda_timer(custom_block_fwd_bwd)

# Bwd only = fwd+bwd - fwd
custom_bwd_est = (custom_fwdbwd_time[0] - custom_fwd_time[0],
                  (custom_fwdbwd_time[1]**2 + custom_fwd_time[1]**2)**0.5)

# 4c. BF16 SDPA backward recompute — time it separately
def time_sdpa_bwd():
    q_t = q_normed.transpose(1, 2).detach().requires_grad_(True)
    k_t = k_normed.transpose(1, 2).detach().requires_grad_(True)
    v_t = v_4d.transpose(1, 2).detach().requires_grad_(True)
    grad_out = torch.randn(B, N_HEAD, T, D, device=device, dtype=torch.bfloat16)
    with torch.enable_grad():
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        out.backward(grad_out)
op_fwd_times["4c. BF16 SDPA bwd recompute"] = cuda_timer(time_sdpa_bwd)

# Print custom block breakdown
print(f"{'Operator':<35} {'Fwd (ms)':>18}")
print("-" * 55)
fwd_total = 0.0
for name, (m, s) in op_fwd_times.items():
    if name.startswith("4c"):
        continue  # show separately
    print(f"  {name:<33} {fmt_ms(m, s)}")
    fwd_total += m
print(f"  {'---':<33} {'---':>18}")
print(f"  {'Sum of operators':<33} {fmt_ms(fwd_total)}")
print(f"  {'Measured fwd':<33} {fmt_ms(*custom_fwd_time)}")
print(f"  {'Measured fwd+bwd':<33} {fmt_ms(*custom_fwdbwd_time)}")
print(f"  {'Estimated bwd (fwd+bwd - fwd)':<33} {fmt_ms(*custom_bwd_est)}")
m4c, s4c = op_fwd_times["4c. BF16 SDPA bwd recompute"]
print(f"  {'4c. BF16 SDPA bwd recompute':<33} {fmt_ms(m4c, s4c)}")
print()

# ── 1b. TE TransformerLayer with MXFP8BlockScaling ──────────────────────

print("--- TE TransformerLayer + MXFP8BlockScaling ---\n")

te_layer_mxfp8 = te.TransformerLayer(
    hidden_size=C, ffn_hidden_size=FFN, num_attention_heads=N_HEAD,
    hidden_dropout=0.0, attention_dropout=0.0, fuse_qkv_params=True,
    activation="srelu", normalization="RMSNorm", bias=False,
    attn_input_format="bshd", seq_length=T, micro_batch_size=B,
    init_method=te_init_method, output_layer_init_method=te_output_layer_init_method,
).to(device, dtype=torch.bfloat16)
te_layer_mxfp8.train()

def te_mxfp8_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_mxfp8):
        _ = te_layer_mxfp8(x_input, rotary_pos_emb=rotary_emb,
                           self_attn_mask_type="causal", is_first_microbatch=True)
te_mxfp8_fwd_time = cuda_timer(te_mxfp8_fwd)

def te_mxfp8_fwd_bwd():
    x_in = x_input.detach().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_mxfp8):
        out = te_layer_mxfp8(x_in, rotary_pos_emb=rotary_emb,
                             self_attn_mask_type="causal", is_first_microbatch=True)
    out.sum().backward()
te_mxfp8_fwdbwd_time = cuda_timer(te_mxfp8_fwd_bwd)
te_mxfp8_bwd_est = (te_mxfp8_fwdbwd_time[0] - te_mxfp8_fwd_time[0],
                    (te_mxfp8_fwdbwd_time[1]**2 + te_mxfp8_fwd_time[1]**2)**0.5)

print(f"  {'Fwd':<33} {fmt_ms(*te_mxfp8_fwd_time)}")
print(f"  {'Bwd (estimated)':<33} {fmt_ms(*te_mxfp8_bwd_est)}")
print(f"  {'Fwd+Bwd':<33} {fmt_ms(*te_mxfp8_fwdbwd_time)}")
print()

# ── 1c. TE TransformerLayer with Float8BlockScaling ─────────────────────

print("--- TE TransformerLayer + Float8BlockScaling ---\n")

te_layer_f8blk = te.TransformerLayer(
    hidden_size=C, ffn_hidden_size=FFN, num_attention_heads=N_HEAD,
    hidden_dropout=0.0, attention_dropout=0.0, fuse_qkv_params=True,
    activation="srelu", normalization="RMSNorm", bias=False,
    attn_input_format="bshd", seq_length=T, micro_batch_size=B,
    init_method=te_init_method, output_layer_init_method=te_output_layer_init_method,
).to(device, dtype=torch.bfloat16)
te_layer_f8blk.train()

def te_f8blk_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_f8blk):
        _ = te_layer_f8blk(x_input, rotary_pos_emb=rotary_emb,
                           self_attn_mask_type="causal", is_first_microbatch=True)
te_f8blk_fwd_time = cuda_timer(te_f8blk_fwd)

def te_f8blk_fwd_bwd():
    x_in = x_input.detach().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_f8blk):
        out = te_layer_f8blk(x_in, rotary_pos_emb=rotary_emb,
                             self_attn_mask_type="causal", is_first_microbatch=True)
    out.sum().backward()
te_f8blk_fwdbwd_time = cuda_timer(te_f8blk_fwd_bwd)
te_f8blk_bwd_est = (te_f8blk_fwdbwd_time[0] - te_f8blk_fwd_time[0],
                    (te_f8blk_fwdbwd_time[1]**2 + te_f8blk_fwd_time[1]**2)**0.5)

print(f"  {'Fwd':<33} {fmt_ms(*te_f8blk_fwd_time)}")
print(f"  {'Bwd (estimated)':<33} {fmt_ms(*te_f8blk_bwd_est)}")
print(f"  {'Fwd+Bwd':<33} {fmt_ms(*te_f8blk_fwdbwd_time)}")
print()


# ═════════════════════════════════════════════════════════════════════════
# Part 2: Head-to-Head Operator Unit Tests
# ═════════════════════════════════════════════════════════════════════════

print("=" * 80)
print(" PART 2: HEAD-TO-HEAD OPERATOR UNIT TESTS")
print("=" * 80)
print()

# 2.1 Attention fwd only: MXFP8 flash vs BF16 SDPA
print("--- 2.1 Attention fwd only: MXFP8 flash vs BF16 SDPA ---\n")

q_bench = torch.randn(B, T, N_HEAD, D, device=device, dtype=torch.bfloat16)
k_bench = torch.randn(B, T, N_HEAD, D, device=device, dtype=torch.bfloat16)
v_bench = torch.randn(B, T, N_HEAD, D, device=device, dtype=torch.bfloat16)

# Pre-quantize for fair kernel-only comparison
q_fp8_b, q_sc_b = quantize_to_mxfp8(q_bench)
k_fp8_b, k_sc_b = quantize_to_mxfp8(k_bench)
v_fp8_b, v_sc_b = quantize_to_mxfp8(v_bench)

def attn_mxfp8_fwd():
    _ = flash_attn_mxfp8_func(
        q_fp8_b, k_fp8_b, v_fp8_b, q_sc_b, k_sc_b, v_sc_b, causal=True)
mxfp8_attn_fwd_time = cuda_timer(attn_mxfp8_fwd)

# BF16 SDPA (expects B,H,T,D)
q_bhsd = q_bench.transpose(1, 2).contiguous()
k_bhsd = k_bench.transpose(1, 2).contiguous()
v_bhsd = v_bench.transpose(1, 2).contiguous()

def attn_sdpa_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _ = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, is_causal=True)
sdpa_attn_fwd_time = cuda_timer(attn_sdpa_fwd)

print(f"  {'MXFP8 flash_attn kernel':<40} {fmt_ms(*mxfp8_attn_fwd_time)}")
print(f"  {'BF16 SDPA (F.scaled_dot_product_attention)':<40} {fmt_ms(*sdpa_attn_fwd_time)}")
ratio = mxfp8_attn_fwd_time[0] / sdpa_attn_fwd_time[0]
print(f"  MXFP8/SDPA ratio: {ratio:.2f}x\n")

# 2.2 Quantize overhead: time quantize_to_mxfp8 for one tensor
print("--- 2.2 Quantize overhead: quantize_to_mxfp8 per tensor ---\n")

def quant_one():
    _ = quantize_to_mxfp8(q_bench)
quant_one_time = cuda_timer(quant_one)

def quant_three():
    _ = quantize_to_mxfp8(q_bench)
    _ = quantize_to_mxfp8(k_bench)
    _ = quantize_to_mxfp8(v_bench)
quant_three_time = cuda_timer(quant_three)

print(f"  {'1 tensor (B,T,H,D)':<40} {fmt_ms(*quant_one_time)}")
print(f"  {'3 tensors (Q, K, V)':<40} {fmt_ms(*quant_three_time)}")
print()

# 2.3 Attention fwd+bwd: hybrid (MXFP8 fwd + BF16 SDPA bwd) vs pure BF16 SDPA
print("--- 2.3 Attention fwd+bwd: hybrid vs pure BF16 SDPA ---\n")

def hybrid_fwd_bwd():
    q_in = q_bench.detach().requires_grad_(True)
    k_in = k_bench.detach().requires_grad_(True)
    v_in = v_bench.detach().requires_grad_(True)
    out = MXFP8Attention.apply(q_in, k_in, v_in)
    out.sum().backward()
hybrid_time = cuda_timer(hybrid_fwd_bwd)

def sdpa_fwd_bwd():
    q_in = q_bhsd.detach().requires_grad_(True)
    k_in = k_bhsd.detach().requires_grad_(True)
    v_in = v_bhsd.detach().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = F.scaled_dot_product_attention(q_in, k_in, v_in, is_causal=True)
    out.sum().backward()
sdpa_fwdbwd_time = cuda_timer(sdpa_fwd_bwd)

print(f"  {'Hybrid (MXFP8 fwd + BF16 SDPA bwd)':<40} {fmt_ms(*hybrid_time)}")
print(f"  {'Pure BF16 SDPA fwd+bwd':<40} {fmt_ms(*sdpa_fwdbwd_time)}")
ratio_fb = hybrid_time[0] / sdpa_fwdbwd_time[0]
print(f"  Hybrid/SDPA ratio: {ratio_fb:.2f}x\n")

# 2.4 QKV projection sanity check
print("--- 2.4 QKV proj (te.LayerNormLinear) under TE autocast ---\n")

qkv_layer = te.LayerNormLinear(
    C, 3 * C, normalization="RMSNorm", bias=False,
    init_method=te_init_method).to(device, dtype=torch.bfloat16)

def qkv_mxfp8():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_mxfp8):
        _ = qkv_layer(x_input, is_first_microbatch=True)
qkv_mxfp8_time = cuda_timer(qkv_mxfp8)

def qkv_f8blk():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), \
         te.autocast(enabled=True, recipe=recipe_f8blk):
        _ = qkv_layer(x_input, is_first_microbatch=True)
qkv_f8blk_time = cuda_timer(qkv_f8blk)

print(f"  {'MXFP8BlockScaling':<40} {fmt_ms(*qkv_mxfp8_time)}")
print(f"  {'Float8BlockScaling':<40} {fmt_ms(*qkv_f8blk_time)}")
print()


# ═════════════════════════════════════════════════════════════════════════
# Part 3: Summary
# ═════════════════════════════════════════════════════════════════════════

print("=" * 80)
print(" PART 3: SUMMARY")
print("=" * 80)
print()

# Side-by-side comparison table
print("--- Per-layer fwd+bwd total ---\n")
print(f"  {'Path':<35} {'Fwd (ms)':>12} {'Bwd (ms)':>12} {'Total (ms)':>12}")
print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")

rows = [
    ("Custom MXFP8 Block",  custom_fwd_time[0], custom_bwd_est[0], custom_fwdbwd_time[0]),
    ("TE + MXFP8BlockScaling", te_mxfp8_fwd_time[0], te_mxfp8_bwd_est[0], te_mxfp8_fwdbwd_time[0]),
    ("TE + Float8BlockScaling", te_f8blk_fwd_time[0], te_f8blk_bwd_est[0], te_f8blk_fwdbwd_time[0]),
]

fastest_total = min(r[3] for r in rows)
for name, fwd, bwd, total in rows:
    pct = (total / fastest_total - 1) * 100
    marker = "" if abs(pct) < 0.5 else f"  (+{pct:.0f}%)"
    print(f"  {name:<35} {fwd:>10.2f}ms {bwd:>10.2f}ms {total:>10.2f}ms{marker}")
print()

# Custom block operator breakdown as % of total
print("--- Custom MXFP8 Block: operator % of fwd ---\n")
print(f"  {'Operator':<35} {'ms':>8} {'% of fwd':>10}")
print(f"  {'-'*35} {'-'*8} {'-'*10}")
for name, (m, s) in op_fwd_times.items():
    if name.startswith("4c"):
        continue
    pct = m / custom_fwd_time[0] * 100
    print(f"  {name:<35} {m:>7.2f} {pct:>9.1f}%")
print(f"  {'-'*35} {'-'*8} {'-'*10}")
print(f"  {'Measured fwd total':<35} {custom_fwd_time[0]:>7.2f} {'100.0%':>10}")
print()

# Bottleneck identification
print("--- Top bottlenecks (Custom MXFP8 vs fastest TE path) ---\n")
fastest_te = min(te_mxfp8_fwdbwd_time[0], te_f8blk_fwdbwd_time[0])
fastest_te_name = "TE+MXFP8" if te_mxfp8_fwdbwd_time[0] < te_f8blk_fwdbwd_time[0] else "TE+Float8Blk"
gap = custom_fwdbwd_time[0] - fastest_te
print(f"  Total gap: {custom_fwdbwd_time[0]:.2f} - {fastest_te:.2f} = {gap:.2f} ms "
      f"(Custom Block is {gap/fastest_te*100:.1f}% slower than {fastest_te_name})")
print()

# Sort operators by time
sorted_ops = sorted(
    [(n, m) for n, (m, s) in op_fwd_times.items() if not n.startswith("4c")],
    key=lambda x: x[1], reverse=True)
print(f"  Fwd time ranking:")
for i, (name, ms) in enumerate(sorted_ops, 1):
    print(f"    {i}. {name:<33} {ms:.2f} ms")
print()

# SDPA backward recompute cost
m4c, s4c = op_fwd_times["4c. BF16 SDPA bwd recompute"]
print(f"  BF16 SDPA backward recompute cost: {m4c:.2f} ms")
print(f"  (This runs inside backward, not counted in fwd total above)")
print(f"  Quantize tax (3 tensors):           {quant_three_time[0]:.2f} ms")
print(f"  Combined MXFP8 attention overhead:  {quant_three_time[0] + m4c:.2f} ms")
print(f"    vs TE's integrated attention:     included in TE fwd+bwd above")
print()

# Extrapolate to full model (20 layers)
n_layers = 20
print(f"--- Extrapolated to full model ({n_layers} layers) ---\n")
for name, fwd, bwd, total in rows:
    full_ms = total * n_layers
    tok_per_sec = B * T / (full_ms / 1000)
    mfu = 6 * 302e6 * B * T / (full_ms / 1000) / (404e12) * 100
    print(f"  {name:<35} {full_ms:>8.1f} ms/iter  {tok_per_sec:>10,.0f} tok/s  ~{mfu:.1f}% MFU")
print()
print("  Note: extrapolation excludes embedding, lm_head, and optimizer overhead.")
print("  Actual training iter time will be higher.")
