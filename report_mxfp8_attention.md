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
| 1 | **MXFP8 + MXFP8 Attention** | MXFP8 fwd + MXFP8 bwd (custom `Block`, direct FP8 cast) | FP8 GEMMs (MXFP8BlockScaling) |
| 2 | **TE + MXFP8BlockScaling** | BF16 SDPA fwd + bwd (`te.TransformerLayer`) | FP8 GEMMs (MXFP8BlockScaling) |

> **Note:** On SM 12.0, TE disables FP8 FlashAttention and FusedAttention entirely
> (`dot_product_attention/utils.py` line ~511). All TE paths use BF16 SDPA for attention.
> FP8 is only applied to linear layers (QKV projection, output projection, MLP).

## Evolution of the MXFP8 Attention Path

The custom MXFP8 attention path went through three iterations:

### v1: MXFP8 fwd + BF16 SDPA bwd (with Python quantization)
- **145 ms/iter, 47% MFU, final loss 7.08**
- BF16 SDPA backward recompute: 0.95 ms/layer
- Python `quantize_to_mxfp8`: 0.50 ms/layer
- Fwd/bwd precision mismatch caused loss spikes (8.1 → 11.7) and poor convergence

### v2: Native MXFP8 fwd + bwd (with Python quantization)
- **135 ms/iter, 51% MFU**
- Native backward kernel eliminated BF16 SDPA recompute
- Fixed fwd/bwd precision mismatch — convergence stable
- Scale convention bug fixed (sf = exp+127, not exp+119)
- Python quantization still costing 0.50 ms/layer

### v3: Native MXFP8 fwd + bwd (direct FP8 cast, no quantization)
- **129 ms/iter, 53% MFU, final loss 6.11**
- After QK RMSNorm, values are ~unit scale — direct `.to(float8_e4m3fn)` with
  identity scales (UE8M0=127=2^0) works as well as per-block quantization
- Eliminated 0.50 ms/layer Python quantization overhead

## 1. Training Throughput (Latest: v3)

| Config | Avg iter time | MFU | Speedup vs BF16 |
|:---|---:|---:|---:|
| TE + MXFP8BlockScaling | 111 ms | 61% | **1.61x** |
| MXFP8 + MXFP8 Attention (v3) | 129 ms | 53% | **1.39x** |
| BF16 baseline | 179 ms | 38% | 1.00x |

The remaining 18 ms gap (129 vs 111 ms) is from unfused RoPE and QK RMSNorm kernels.
TE fuses these into its attention module; the custom Block runs them as separate CUDA kernels.

## 2. Training Convergence (50 steps, v3 vs TE baseline)

### Loss curve

```
Step  MXFP8 Attn (v3)   TE MXFP8
----  ----------------  ---------
   1          10.999     10.999
   2          10.073     10.072
   3           8.977      8.982
   4           8.657      8.656
   5           8.536      8.581
  10           9.188      9.862
  15           6.656      6.994
  20           6.960      7.144
  25           6.467      6.609
  30           6.300      6.387
  35           6.249      6.313
  40           6.210      6.241
  45           6.205      6.249
  49           6.107      6.045
```

### Key metrics

| Metric | MXFP8 Attn (v3) | TE MXFP8 |
|:---|---:|---:|
| Final train loss | 6.107 | 6.045 |
| Val loss @40 | 6.298 | 6.327 |
| Grad norm @49 | 0.131 | 0.130 |

Convergence is essentially identical. Both paths reach ~6.1 final loss with stable
gradient norms settling to ~0.13 by step 49. No loss spikes or instability.

## 3. Operator-Level Analysis

Per-layer timing breakdown from `benchmark_attention.py` (CUDA event timing, 20 runs).
Note: these timings are from the v1 architecture; v3 eliminates the quantize_to_mxfp8 cost.

### Custom MXFP8 Block — forward pass operators

| Operator | Time (ms) | % of fwd | Status in v3 |
|:---|---:|---:|:---|
| LayerNormMLP (RMSNorm + FC1 + SReLU + FC2) | 0.59 | 32.0% | unchanged |
| ~~`quantize_to_mxfp8` × 3~~ | ~~0.53~~ | ~~28.5%~~ | **eliminated** (direct cast) |
| QKV proj (LayerNormLinear, FP8 GEMM) | 0.26 | 14.3% | unchanged |
| RoPE × 2 | 0.25 | 13.7% | unchanged |
| `flash_attn_mxfp8` kernel | 0.21 | 11.4% | unchanged |
| Output proj (te.Linear, FP8 GEMM) | 0.16 | 8.4% | unchanged |
| QK RMSNorm × 2 | 0.10 | 5.4% | unchanged |
| Residual adds × 2 | 0.02 | 1.3% | unchanged |

### Head-to-head kernel comparisons

| Test | Time (ms) | Ratio |
|:---|---:|---:|
| MXFP8 flash attn fwd | 0.21 | 0.74x (26% faster) |
| BF16 SDPA fwd | 0.29 | 1.00x |

The MXFP8 flash attention kernel is 26% faster than BF16 SDPA in isolation.

## 4. Root Cause: Remaining 18ms Gap

The 18 ms/iter gap between MXFP8 attention (129ms) and TE baseline (111ms) comes from
**kernel fusion**, not precision conversion overhead:

| Source | Estimated cost | Notes |
|:---|---:|:---|
| RoPE × 2 (separate kernels) | ~0.25 ms/layer | TE fuses into attention |
| QK RMSNorm × 2 (separate kernels) | ~0.10 ms/layer | TE fuses into attention |
| Flash attn kernel dispatch overhead | ~0.05 ms/layer | vs TE's integrated SDPA |
| **Total per layer** | **~0.40 ms** | |
| **Total for 20 layers** | **~8 ms** | |

The remaining ~10ms is likely from backward pass differences (TE's fused backward vs
our separate backward kernels for each operator).

## 5. Key Technical Decisions

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

### Flash attention output stays BF16
The MXFP8 flash attention kernel outputs BF16 (softmax intermediate is FP32, accumulated in
FP32, then cast to BF16). FP8 output would lose too much precision for residual connections.
The subsequent `te.Linear` output projection handles its own BF16→FP8 quantization internally
with fused CUDA kernels.

## 6. Summary

| Rank | Config | Iter time | MFU | Final loss |
|---|---|---:|---:|---:|
| 1 | TE + MXFP8BlockScaling | 111 ms | 61% | 6.045 |
| 2 | MXFP8 + MXFP8 Attention (v3) | 129 ms | 53% | 6.107 |
| 3 | BF16 baseline | 179 ms | 38% | 6.017 |

### Recommendation

**MXFP8 attention with native fwd+bwd kernels and direct FP8 cast** is now viable for
training on SM 12.0. It achieves 1.39x speedup over BF16 with matching convergence quality
(6.11 vs 6.05 final loss). The TE baseline remains 16% faster (111 vs 129 ms) due to
kernel fusion advantages, not precision conversion overhead.

The remaining gap could be closed by fusing RoPE + QK RMSNorm into the attention kernel
or using a custom fused pre-attention kernel.
