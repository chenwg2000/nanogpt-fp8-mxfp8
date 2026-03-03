"""
Per-layer breakdown: Linear vs Attention (fwd/bwd) for both paths.

Path 1: MXFP8 Linear + MXFP8 Attention (custom Block)
Path 2: MXFP8 Linear + BF16 Attention (TE TransformerLayer)

Times: QKV proj, attention, output proj, MLP — each fwd and bwd separately.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")
from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func
from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

torch.manual_seed(42)
device = "cuda"

B, T, N_HEAD, D = 4, 2048, 10, 128
C = N_HEAD * D  # 1280
FFN = 4 * C     # 5120
WARMUP = 3
TIMED = 10

recipe = MXFP8BlockScaling()


def te_init_method(weight):
    fan_out, fan_in = weight.size(0), weight.size(1)
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    torch.nn.init.normal_(weight, mean=0.0, std=std)

def te_output_layer_init_method(weight):
    torch.nn.init.zeros_(weight)


def cuda_timer(fn, warmup=WARMUP, iters=TIMED):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.mean(times)), float(np.std(times))


def fmt(m, s=None):
    if s is not None:
        return f"{m:7.2f} ± {s:.2f}"
    return f"{m:7.2f}"


# ═══════════════════════════════════════════════════════════════════════
# Path 1: MXFP8 Linear + MXFP8 Attention (Custom Block)
# ═══════════════════════════════════════════════════════════════════════

print("=" * 80)
print(" PATH 1: MXFP8 Linear + MXFP8 Attention (Custom Block)")
print("=" * 80)

qkv_proj = te.LayerNormLinear(C, 3*C, normalization="RMSNorm", bias=False,
                               init_method=te_init_method).to(device, dtype=torch.bfloat16)
q_norm = te.RMSNorm(D).to(device, dtype=torch.bfloat16)
k_norm = te.RMSNorm(D).to(device, dtype=torch.bfloat16)
out_proj = te.Linear(C, C, bias=False,
                      init_method=te_output_layer_init_method).to(device, dtype=torch.bfloat16)
mlp = te.LayerNormMLP(C, FFN, normalization="RMSNorm", activation="srelu", bias=False,
                       init_method=te_init_method,
                       output_layer_init_method=te_output_layer_init_method).to(device, dtype=torch.bfloat16)
rope = te.RotaryPositionEmbedding(dim=D, pretrained_max_position_embeddings=T)
rotary_emb = rope(T)

x_input = torch.randn(B, T, C, device=device, dtype=torch.bfloat16)

# --- QKV Proj fwd ---
def qkv_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        return qkv_proj(x_input, is_first_microbatch=True)
qkv_fwd_t = cuda_timer(qkv_fwd)

# --- QKV Proj fwd+bwd ---
def qkv_fwdbwd():
    x = x_input.detach().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        out = qkv_proj(x, is_first_microbatch=True)
    out.sum().backward()
qkv_fwdbwd_t = cuda_timer(qkv_fwdbwd)
qkv_bwd_t = (qkv_fwdbwd_t[0] - qkv_fwd_t[0], (qkv_fwdbwd_t[1]**2 + qkv_fwd_t[1]**2)**0.5)

# --- RoPE + QK Norm fwd ---
with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
    qkv_out = qkv_proj(x_input, is_first_microbatch=True)
q_raw, k_raw, v_raw = qkv_out.chunk(3, dim=-1)
q_4d = q_raw.reshape(B, T, N_HEAD, D)
k_4d = k_raw.reshape(B, T, N_HEAD, D)
v_4d = v_raw.reshape(B, T, N_HEAD, D)

def rope_norm_fwd():
    q = apply_rotary_pos_emb(q_4d, rotary_emb, tensor_format="bshd")
    k = apply_rotary_pos_emb(k_4d, rotary_emb, tensor_format="bshd")
    q = q_norm(q)
    k = k_norm(k)
    return q, k
rope_norm_fwd_t = cuda_timer(rope_norm_fwd)

# --- MXFP8 Attention fwd ---
q_normed = q_norm(apply_rotary_pos_emb(q_4d, rotary_emb, tensor_format="bshd"))
k_normed = k_norm(apply_rotary_pos_emb(k_4d, rotary_emb, tensor_format="bshd"))

q_fp8 = q_normed.to(torch.float8_e4m3fn)
k_fp8 = k_normed.to(torch.float8_e4m3fn)
v_fp8 = v_4d.to(torch.float8_e4m3fn)
identity_scale = torch.full((B, N_HEAD, T, D // 32), 127, dtype=torch.uint8, device=device)

def mxfp8_attn_fwd():
    return flash_attn_mxfp8_func(q_fp8, k_fp8, v_fp8,
                                  identity_scale, identity_scale, identity_scale, causal=True)
mxfp8_attn_fwd_t = cuda_timer(mxfp8_attn_fwd)

# --- MXFP8 Attention bwd ---
attn_out, softmax_lse = flash_attn_mxfp8_func(q_fp8, k_fp8, v_fp8,
                                                identity_scale, identity_scale, identity_scale, causal=True)
grad_attn = torch.randn_like(attn_out)

def mxfp8_attn_bwd():
    return flash_attn_mxfp8_bwd_func(grad_attn, q_fp8, k_fp8, v_fp8,
                                      attn_out, softmax_lse,
                                      identity_scale, identity_scale, causal=True)
mxfp8_attn_bwd_t = cuda_timer(mxfp8_attn_bwd)

# --- FP8 cast overhead ---
def fp8_cast():
    _ = q_normed.to(torch.float8_e4m3fn)
    _ = k_normed.to(torch.float8_e4m3fn)
    _ = v_4d.to(torch.float8_e4m3fn)
fp8_cast_t = cuda_timer(fp8_cast)

# --- Output Proj fwd ---
attn_3d = attn_out.reshape(B, T, C)
def outproj_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        return out_proj(attn_3d, is_first_microbatch=True)
outproj_fwd_t = cuda_timer(outproj_fwd)

# --- Output Proj fwd+bwd ---
def outproj_fwdbwd():
    a = attn_3d.detach().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        o = out_proj(a, is_first_microbatch=True)
    o.sum().backward()
outproj_fwdbwd_t = cuda_timer(outproj_fwdbwd)
outproj_bwd_t = (outproj_fwdbwd_t[0] - outproj_fwd_t[0],
                  (outproj_fwdbwd_t[1]**2 + outproj_fwd_t[1]**2)**0.5)

# --- MLP fwd ---
def mlp_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        return mlp(x_input, is_first_microbatch=True)
mlp_fwd_t = cuda_timer(mlp_fwd)

# --- MLP fwd+bwd ---
def mlp_fwdbwd():
    x = x_input.detach().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        o = mlp(x, is_first_microbatch=True)
    o.sum().backward()
mlp_fwdbwd_t = cuda_timer(mlp_fwdbwd)
mlp_bwd_t = (mlp_fwdbwd_t[0] - mlp_fwd_t[0], (mlp_fwdbwd_t[1]**2 + mlp_fwd_t[1]**2)**0.5)

# Print Path 1
print(f"\n{'Operator':<35} {'Fwd (ms)':>14} {'Bwd (ms)':>14} {'Total (ms)':>14}")
print("-" * 80)

p1_rows = [
    ("QKV proj (LayerNormLinear)", qkv_fwd_t, qkv_bwd_t, qkv_fwdbwd_t),
    ("RoPE + QK RMSNorm", rope_norm_fwd_t, None, None),
    ("FP8 cast (Q,K,V)", fp8_cast_t, None, None),
    ("MXFP8 Attention fwd", mxfp8_attn_fwd_t, None, None),
    ("MXFP8 Attention bwd", None, mxfp8_attn_bwd_t, None),
    ("Output proj (te.Linear)", outproj_fwd_t, outproj_bwd_t, outproj_fwdbwd_t),
    ("MLP (LayerNormMLP)", mlp_fwd_t, mlp_bwd_t, mlp_fwdbwd_t),
]

p1_fwd_total = 0.0
p1_bwd_total = 0.0
for name, fwd, bwd, total in p1_rows:
    fwd_s = fmt(*fwd) if fwd else "       —"
    bwd_s = fmt(*bwd) if bwd else "       —"
    tot_s = fmt(*total) if total else "       —"
    print(f"  {name:<33} {fwd_s:>14} {bwd_s:>14} {tot_s:>14}")
    if fwd: p1_fwd_total += fwd[0]
    if bwd: p1_bwd_total += bwd[0]

p1_total = p1_fwd_total + p1_bwd_total
print(f"  {'-'*33} {'-'*14} {'-'*14} {'-'*14}")
print(f"  {'TOTAL':<33} {fmt(p1_fwd_total):>14} {fmt(p1_bwd_total):>14} {fmt(p1_total):>14}")

# Categorize
p1_linear_fwd = qkv_fwd_t[0] + outproj_fwd_t[0] + mlp_fwd_t[0]
p1_linear_bwd = qkv_bwd_t[0] + outproj_bwd_t[0] + mlp_bwd_t[0]
p1_attn_fwd = rope_norm_fwd_t[0] + fp8_cast_t[0] + mxfp8_attn_fwd_t[0]
p1_attn_bwd = mxfp8_attn_bwd_t[0]

print(f"\n  {'Category':<35} {'Fwd (ms)':>10} {'Bwd (ms)':>10} {'Total (ms)':>10} {'%':>8}")
print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
print(f"  {'Linear (QKV+OutProj+MLP)':<35} {p1_linear_fwd:>9.2f} {p1_linear_bwd:>9.2f} {p1_linear_fwd+p1_linear_bwd:>9.2f} {(p1_linear_fwd+p1_linear_bwd)/p1_total*100:>7.1f}%")
print(f"  {'Attention (RoPE+Norm+Cast+Kernel)':<35} {p1_attn_fwd:>9.2f} {p1_attn_bwd:>9.2f} {p1_attn_fwd+p1_attn_bwd:>9.2f} {(p1_attn_fwd+p1_attn_bwd)/p1_total*100:>7.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# Path 2: MXFP8 Linear + BF16 Attention (TE TransformerLayer)
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(" PATH 2: MXFP8 Linear + BF16 Attention (TE TransformerLayer)")
print("=" * 80)

te_layer = te.TransformerLayer(
    hidden_size=C, ffn_hidden_size=FFN, num_attention_heads=N_HEAD,
    hidden_dropout=0.0, attention_dropout=0.0, fuse_qkv_params=True,
    activation="srelu", normalization="RMSNorm", bias=False,
    attn_input_format="bshd", seq_length=T, micro_batch_size=B,
    init_method=te_init_method, output_layer_init_method=te_output_layer_init_method,
).to(device, dtype=torch.bfloat16)
te_layer.train()

# TE fuses everything, so we can only time the full layer and the MLP/attention sub-layers
# by extracting the internal modules.

# --- Full layer fwd ---
def te_full_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        return te_layer(x_input, rotary_pos_emb=rotary_emb,
                        self_attn_mask_type="causal", is_first_microbatch=True)
te_full_fwd_t = cuda_timer(te_full_fwd)

# --- Full layer fwd+bwd ---
def te_full_fwdbwd():
    x = x_input.detach().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        out = te_layer(x, rotary_pos_emb=rotary_emb,
                       self_attn_mask_type="causal", is_first_microbatch=True)
    out.sum().backward()
te_full_fwdbwd_t = cuda_timer(te_full_fwdbwd)
te_full_bwd_t = (te_full_fwdbwd_t[0] - te_full_fwd_t[0],
                  (te_full_fwdbwd_t[1]**2 + te_full_fwd_t[1]**2)**0.5)

# --- Time the self-attention sub-layer alone (LN+QKV+RoPE+Norm+SDPA+OutProj) ---
te_self_attn = te_layer.self_attention

def te_attn_sublayer_fwd():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        return te_self_attn(x_input, rotary_pos_emb=rotary_emb,
                           attn_mask_type="causal", is_first_microbatch=True)
te_attn_fwd_t = cuda_timer(te_attn_sublayer_fwd)

def te_attn_sublayer_fwdbwd():
    x = x_input.detach().requires_grad_(True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16), te.autocast(enabled=True, recipe=recipe):
        out = te_self_attn(x, rotary_pos_emb=rotary_emb,
                          attn_mask_type="causal", is_first_microbatch=True)
    if isinstance(out, tuple): out = out[0]
    out.sum().backward()
te_attn_fwdbwd_t = cuda_timer(te_attn_sublayer_fwdbwd)
te_attn_bwd_t = (te_attn_fwdbwd_t[0] - te_attn_fwd_t[0],
                  (te_attn_fwdbwd_t[1]**2 + te_attn_fwd_t[1]**2)**0.5)

# --- MLP sub-layer (reuse from Path 1 — same module type + recipe) ---
# Already timed above as mlp_fwd_t, mlp_bwd_t, mlp_fwdbwd_t

# Estimate: linear portion = full - attention_sublayer (approximate)
te_mlp_fwd_est = te_full_fwd_t[0] - te_attn_fwd_t[0]
te_mlp_bwd_est = te_full_bwd_t[0] - te_attn_bwd_t[0]

print(f"\n{'Operator':<35} {'Fwd (ms)':>14} {'Bwd (ms)':>14} {'Total (ms)':>14}")
print("-" * 80)
print(f"  {'Self-Attention (QKV+RoPE+Norm+SDPA+OutProj)':<33}")
print(f"  {'  (fused sub-layer)':<33} {fmt(*te_attn_fwd_t):>14} {fmt(*te_attn_bwd_t):>14} {fmt(*te_attn_fwdbwd_t):>14}")
print(f"  {'MLP (remainder: full - attn)':<33} {fmt(te_mlp_fwd_est):>14} {fmt(te_mlp_bwd_est):>14} {fmt(te_mlp_fwd_est+te_mlp_bwd_est):>14}")
print(f"  {'-'*33} {'-'*14} {'-'*14} {'-'*14}")
print(f"  {'TOTAL (full layer)':<33} {fmt(*te_full_fwd_t):>14} {fmt(*te_full_bwd_t):>14} {fmt(*te_full_fwdbwd_t):>14}")


# ═══════════════════════════════════════════════════════════════════════
# Side-by-Side Summary
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(" SIDE-BY-SIDE SUMMARY (per layer)")
print("=" * 80)

print(f"\n{'Component':<30} {'MXFP8 Attn (ms)':>16} {'BF16 Attn/TE (ms)':>18} {'Delta':>10}")
print("-" * 76)

# Attention (everything non-MLP)
p1_attn_total = p1_attn_fwd + p1_attn_bwd + qkv_fwd_t[0] + qkv_bwd_t[0] + outproj_fwd_t[0] + outproj_bwd_t[0]
p2_attn_total = te_attn_fwdbwd_t[0]
print(f"  {'Attn sub-layer (fwd+bwd)':<28} {p1_attn_total:>14.2f} {p2_attn_total:>16.2f} {p1_attn_total-p2_attn_total:>+9.2f}")

# Breakdown within attention
p1_qkv_total = qkv_fwdbwd_t[0]
p1_rope_norm = rope_norm_fwd_t[0]
p1_cast = fp8_cast_t[0]
p1_kernel_fwd = mxfp8_attn_fwd_t[0]
p1_kernel_bwd = mxfp8_attn_bwd_t[0]
p1_outproj_total = outproj_fwdbwd_t[0]

print(f"    {'QKV proj (fwd+bwd)':<26} {p1_qkv_total:>14.2f} {'(included)':>16} {'':>10}")
print(f"    {'RoPE + QK Norm (fwd)':<26} {p1_rope_norm:>14.2f} {'(included)':>16} {'':>10}")
print(f"    {'FP8 cast (fwd)':<26} {p1_cast:>14.2f} {'—':>16} {'':>10}")
print(f"    {'Attn kernel fwd':<26} {p1_kernel_fwd:>14.2f} {'(BF16 SDPA)':>16} {'':>10}")
print(f"    {'Attn kernel bwd':<26} {p1_kernel_bwd:>14.2f} {'(BF16 SDPA)':>16} {'':>10}")
print(f"    {'Output proj (fwd+bwd)':<26} {p1_outproj_total:>14.2f} {'(included)':>16} {'':>10}")

# MLP
p1_mlp_total = mlp_fwdbwd_t[0]
p2_mlp_total = te_mlp_fwd_est + te_mlp_bwd_est
print(f"  {'MLP sub-layer (fwd+bwd)':<28} {p1_mlp_total:>14.2f} {p2_mlp_total:>16.2f} {p1_mlp_total-p2_mlp_total:>+9.2f}")

print(f"  {'-'*28} {'-'*14} {'-'*16} {'-'*10}")
p1_layer_total = p1_total
p2_layer_total = te_full_fwdbwd_t[0]
print(f"  {'LAYER TOTAL (fwd+bwd)':<28} {p1_layer_total:>14.2f} {p2_layer_total:>16.2f} {p1_layer_total-p2_layer_total:>+9.2f}")

# Extrapolate
print(f"\n--- Extrapolated to 20 layers ---\n")
p1_model = p1_layer_total * 20
p2_model = p2_layer_total * 20
print(f"  {'MXFP8 Attn path':<30} {p1_model:>8.1f} ms/iter")
print(f"  {'TE BF16 Attn path':<30} {p2_model:>8.1f} ms/iter")
print(f"  {'Delta':<30} {p1_model - p2_model:>+8.1f} ms/iter")
print(f"\n  Note: excludes embedding, lm_head, optimizer. Actual training adds ~10-20ms.")
