# MXFP8 Flash Attention Performance & Convergence Report

Comparison on NVIDIA RTX 5090 (SM 12.0, Blackwell consumer)

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX 5090 (SM 12.0, 32GB GDDR7) |
| Model | 561M param decoder-only transformer |
| Architecture | 20 layers, 1280 hidden, 10 heads, 128 head dim, SReLU, RMSNorm |
| Sequence length | 2048 |
| Micro-batch size | 4 |
| Total batch size | 516,096 tokens (63 grad accum steps) |
| Optimizer | NorMuon (lr=0.02) + AdamW for embeddings |
| Training steps | 50 |
| Transformer Engine | 2.13.0.dev0+c93ca27a (source build) |
| PyTorch | 2.11.0a0+git5e4d6fb (source build, CUDA 13.0) |
| Flash Attention | MXFP8 branch with native fwd+bwd kernels for SM 12.0 |

## Configurations Tested

| # | Config | Attention Precision | Linear Layer Precision |
|---|---|---|---|
| 1 | **MXFP8 + MXFP8 Attention** | MXFP8 fwd + MXFP8 bwd (custom `Block`, direct FP8 cast, torch.compile) | FP8 GEMMs (MXFP8BlockScaling) |
| 2 | **TE + MXFP8BlockScaling** | BF16 SDPA fwd + bwd (`te.TransformerLayer`) | FP8 GEMMs (MXFP8BlockScaling) |

> **Note:** On SM 12.0, TE disables FP8 FlashAttention and FusedAttention entirely
> (`dot_product_attention/utils.py` line ~511). All TE paths use BF16 SDPA for attention.
> FP8 is only applied to linear layers (QKV projection, output projection, MLP).

## 1. Precision Analysis

### Custom MXFP8 Block — per-stage precision

```
Input (BF16)
  │
  ├─ Self-Attention Sub-layer
  │   ├─ QKV Projection (te.LayerNormLinear)
  │   │     RMSNorm:         BF16 → BF16
  │   │     GEMM:            E4M3 × E4M3 → FP32 accumulate (split acc) → BF16
  │   ├─ RoPE (Q, K)         BF16 → BF16  (torch.compiled with QK Norm)
  │   ├─ QK RMSNorm (Q, K)   BF16 → BF16  (torch.compiled with RoPE)
  │   ├─ FP8 Cast (Q, K, V)  BF16 → E4M3 FP8 (direct cast, identity scales UE8M0=127)
  │   ├─ MXFP8 Flash Attention
  │   │     Q × K^T:         E4M3 × E4M3 → FP32 accumulate
  │   │     Softmax:          FP32
  │   │     P × V:            E4M3 × E4M3 → FP32 accumulate → BF16 output
  │   ├─ Output Projection (te.Linear)
  │   │     GEMM:            E4M3 × E4M3 → FP32 accumulate → BF16
  │   └─ Residual Add        BF16
  │
  ├─ MLP Sub-layer (te.LayerNormMLP)
  │   ├─ RMSNorm              BF16 → BF16
  │   ├─ FC1 GEMM             E4M3 × E4M3 → FP32 accumulate → BF16
  │   ├─ SReLU                BF16
  │   ├─ FC2 GEMM             E4M3 × E4M3 → FP32 accumulate → BF16
  │   └─ Residual Add         BF16
  │
Output (BF16)
```

### TE Baseline — per-stage precision

```
Input (BF16)
  │
  ├─ Self-Attention Sub-layer (te.TransformerLayer)
  │   ├─ QKV Projection
  │   │     RMSNorm:         BF16 → BF16
  │   │     GEMM:            E4M3 × E4M3 → FP32 accumulate (split acc) → BF16
  │   ├─ RoPE (Q, K)         BF16 → BF16  (fused into attention module)
  │   ├─ QK RMSNorm (Q, K)   BF16 → BF16  (fused into attention module)
  │   ├─ BF16 Attention (UnfusedDotProductAttention, FP8 disabled on SM 12.0)
  │   │     Q × K^T:         BF16 × BF16 → BF16
  │   │     Softmax:          BF16
  │   │     P × V:            BF16 × BF16 → BF16
  │   ├─ Output Projection
  │   │     GEMM:            E4M3 × E4M3 → FP32 accumulate → BF16
  │   └─ Residual Add        BF16
  │
  ├─ MLP Sub-layer
  │   ├─ RMSNorm              BF16 → BF16
  │   ├─ FC1 GEMM             E4M3 × E4M3 → FP32 accumulate → BF16
  │   ├─ SReLU                BF16
  │   ├─ FC2 GEMM             E4M3 × E4M3 → FP32 accumulate → BF16
  │   └─ Residual Add         BF16
  │
Output (BF16)
```

### Key precision differences

| Component | Custom MXFP8 Block | TE Baseline |
|---|---|---|
| Attention Q × K^T | **E4M3 × E4M3 → FP32** | BF16 × BF16 → BF16 |
| Softmax | **FP32** | BF16 |
| Attention P × V | **E4M3 × E4M3 → FP32** | BF16 × BF16 → BF16 |
| Attention backward | **E4M3 FP8 GEMMs** | BF16 (autograd) |
| FP8 GEMMs per layer (fwd) | **6** (4 linear + 2 attn) | **4** (linear only) |
| FP8 GEMMs per layer (bwd) | **12** (8 linear + 4 attn) | **8** (linear only) |
| Linear layers | Identical | Identical |
| RMSNorm, RoPE, activations | Identical (BF16) | Identical (BF16) |

The custom path uses FP8 for **all 18 GEMMs per layer** (fwd+bwd), while TE uses FP8 for
only 12 (linear layers only). Notably, the custom path computes softmax in FP32 (vs TE's
BF16), giving higher numerical stability in attention despite lower-precision inputs.

## 2. Evolution of the MXFP8 Attention Path

| Version | Changes | Iter time | MFU | Final loss |
|---|---|---|---|---|
| v1 | MXFP8 fwd + BF16 SDPA bwd, Python quantization | 145 ms | 47% | 7.08 |
| v2 | Native MXFP8 fwd+bwd, Python quantization | 135 ms | 51% | — |
| v3 | Native fwd+bwd, direct FP8 cast (no quantization) | 129 ms | 53% | 6.11 |
| v3+ | v3 + updated flash-attn library | 120 ms | 57% | 6.11 |
| **v3+ compiled** | **v3+ with torch.compile on pre-attention** | **114 ms** | **60%** | **6.12** |

Key fixes at each stage:
- **v1→v2**: Native backward kernel fixed fwd/bwd precision mismatch (loss spikes eliminated)
- **v1→v2**: Scale convention fix (sf = exp+127, not exp+119) fixed NaN gradients
- **v2→v3**: Direct FP8 cast eliminated 0.50 ms/layer Python quantization overhead
- **v3→v3+**: Updated flash-attention library optimized kernel performance
- **v3+→v3+ compiled**: `torch.compile` on `_pre_attention` fuses surrounding tensor ops (~6ms savings)

### Fused kernel investigation (tested, reverted)
A fully fused kernel (`flash_attn_mxfp8_fused_func` + `flash_attn_mxfp8_fused_bwd_func`)
that incorporates RoPE + QK RMSNorm + FP8 cast into the attention kernel was tested.
The fused forward saved 0.037ms/layer, but the fused backward was 0.711ms/layer slower
than autograd's handling of separate TE backward kernels. Net: +0.637ms/layer = +12.7ms/iter
worse. Reverted to unfused v3+ with torch.compile.

## 3. Training Throughput (Latest: v3+ compiled)

| Config | Avg iter time | MFU | Speedup vs BF16 |
|:---|---:|---:|---:|
| **MXFP8 + MXFP8 Attention (v3+ compiled)** | **114 ms** | **60%** | **1.57x** |
| TE + MXFP8BlockScaling | 111 ms | 61% | **1.61x** |
| BF16 baseline | 179 ms | 38% | 1.00x |

The gap between MXFP8 attention and TE baseline is now **~3 ms** (~114 vs ~111 ms).

## 4. Training Convergence (50 steps, latest)

### Loss curve

```
Step  MXFP8 Attn   TE MXFP8
----  ----------  ---------
   1     10.999     10.999
   2     10.073     10.072
   3      8.977      8.982
   4      8.657      8.655
   5      8.536      8.581
  10      9.372      9.845
  15      6.700      7.009
  20      6.977      7.128
  25      6.472      6.609
  30      6.299      6.394
  35      6.259      6.307
  40      6.218      6.245
  45      6.207      6.242
  49      6.121      6.060
```

### Key metrics

| Metric | MXFP8 Attn | TE MXFP8 |
|:---|---:|---:|
| Final train loss | 6.121 | 6.060 |
| Val loss @40 | 6.301 | 6.340 |
| Grad norm @49 | 0.177 | 0.148 |

Convergence is essentially identical. Both paths reach ~6.1 final loss with stable
gradient norms. MXFP8 attention achieves slightly better val loss (6.30 vs 6.34),
likely from the FP32 softmax precision advantage.

## 5. Per-Layer Operator Breakdown

From `benchmark_breakdown.py` (CUDA event timing, 10 runs):

### Custom MXFP8 Block

| Operator | Fwd (ms) | Bwd (ms) | Total (ms) |
|:---|---:|---:|---:|
| QKV proj (te.LayerNormLinear) | 0.27 | 0.41 | 0.68 |
| RoPE + QK RMSNorm (compiled) | 0.29 | — | 0.29 |
| FP8 cast (Q,K,V) | 0.04 | — | 0.04 |
| MXFP8 Attention kernel | 0.20 | 0.78 | 0.98 |
| Output proj (te.Linear) | 0.16 | 0.32 | 0.48 |
| MLP (te.LayerNormMLP) | 0.53 | 0.97 | 1.50 |
| **Total** | **1.48** | **2.49** | **3.97** |

Category breakdown:
- **Linear layers** (QKV + OutProj + MLP): 2.66 ms (67%)
- **Attention** (RoPE + Norm + Cast + Kernel): 1.31 ms (33%)

### TE Baseline

| Operator | Fwd (ms) | Bwd (ms) | Total (ms) |
|:---|---:|---:|---:|
| Self-Attention (fused: LN+QKV+RoPE+Norm+SDPA+OutProj) | 0.72 | 1.44 | 2.16 |
| MLP (LayerNormMLP) | 0.53 | 1.11 | 1.64 |
| **Total** | **1.25** | **2.55** | **3.80** |

### Side-by-side

| Component | MXFP8 Attn | TE Baseline | Delta |
|:---|---:|---:|---:|
| Attention sub-layer | 2.47 ms | 2.16 ms | +0.31 ms |
| MLP sub-layer | 1.50 ms | 1.64 ms | -0.14 ms |
| **Layer total** | **3.97 ms** | **3.80 ms** | **+0.17 ms** |

The per-layer gap is only 0.17 ms, coming from unfused RoPE + QK RMSNorm (0.29 ms as
separate kernels, partially offset by a faster MLP in the custom path).
`torch.compile` on `_pre_attention` reduces this further at training time by fusing
surrounding tensor ops (~6 ms/iter savings).

## 6. Flash Attention GEMM Dimensions

For training config B=4, T=2048, H=10, D=128 (40 independent heads):

### Forward (per head)

| GEMM | Operation | Dimensions (M × N × K) | Precision |
|---|---|---|---|
| S = Q × K^T | Attention scores | 2048 × 2048 × 128 | E4M3 × E4M3 → FP32 |
| O = softmax(S) × V | Attention output | 2048 × 128 × 2048 | E4M3 × E4M3 → FP32 → BF16 |

### Backward (per head)

| GEMM | Operation | Dimensions (M × N × K) |
|---|---|---|
| dS = dO × V^T | Score gradients | 2048 × 2048 × 128 |
| dQ = dS × K | Query gradients | 2048 × 128 × 2048 |
| dK = dS^T × Q | Key gradients | 2048 × 128 × 2048 |
| dV = P^T × dO | Value gradients | 2048 × 128 × 2048 |

All tiled (flash attention never materializes the full 2048×2048 matrix).
Causal masking roughly halves the effective work.

## 7. Key Technical Decisions

### Scale convention fix (v2)
Our original `quantize_to_mxfp8` used `sf = floor(log2(amax)) + 119` (offset by -8 for
E4M3 max range). The flash attention backward kernel expects the standard convention
`sf = floor(log2(amax)) + 127`. The offset caused gradient magnitudes of ~10M (should be
~0.01), producing NaN during training. Fixed by matching the library's convention.

### Direct FP8 cast (v3)
After QK RMSNorm, values are already normalized to ~unit scale. Per-block MXFP8 quantization
(computing per-32-element amax and shared exponents) is unnecessary — a simple
`.to(torch.float8_e4m3fn)` with identity scales (UE8M0=127=2^0=1.0) produces equivalent
convergence at zero quantization cost.

### torch.compile on _pre_attention (v3+ compiled)
`torch.compile(self._pre_attention)` compiles the RoPE + QK RMSNorm method. Dynamo cannot
trace TE's C++ `rmsnorm_fwd` kernel, but it fuses surrounding tensor operations (reshape,
chunk, broadcast). This saves ~6 ms/iter (114ms vs 120ms without compile), closing the gap
to TE baseline from ~9ms to ~3ms.

### Flash attention output stays BF16
The MXFP8 flash attention kernel outputs BF16 (softmax intermediate is FP32, accumulated in
FP32, then cast to BF16). FP8 output would lose too much precision for residual connections.
The subsequent `te.Linear` output projection handles its own BF16→FP8 quantization internally
with fused CUDA kernels.

### Fused kernel: forward helps, backward hurts
A fully fused kernel was tested that combines RoPE + QK RMSNorm + FP8 cast + attention
into a single kernel call. The fused forward saved only 0.037ms/layer (the separate ops
were already fast). The fused backward was 0.711ms/layer slower because autograd's handling
of TE's separate C++ backward kernels is more efficient than the fused backward kernel.
The unfused path with `torch.compile` remains the better architecture.

## 8. Summary

| Rank | Config | Iter time | MFU | Final loss | FP8 GEMMs/layer |
|---|---|---:|---:|---:|---:|
| 1 | TE + MXFP8BlockScaling | 111 ms | 61% | 6.060 | 12 |
| 2 | MXFP8 + MXFP8 Attention (v3+ compiled) | 114 ms | 60% | 6.121 | 18 |
| 3 | BF16 baseline | 179 ms | 38% | 6.017 | 0 |

### Recommendation

**MXFP8 attention with native fwd+bwd kernels and torch.compile** is at near-parity with
the TE baseline on SM 12.0 (114 ms vs 111 ms, 60% vs 61% MFU) with matching convergence
(6.12 vs 6.06 final loss, 6.30 vs 6.34 val loss). It uses FP8 for all 18 GEMMs per layer
(vs TE's 12), with FP32 softmax providing higher intermediate precision than TE's BF16 softmax.

The ~3 ms remaining gap comes from separate RoPE + QK RMSNorm kernel launches that
`torch.compile` partially but not fully fuses. TE avoids this by integrating these ops
directly into its attention module's C++ implementation.
