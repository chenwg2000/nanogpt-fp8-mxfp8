# MXFP8 Flash Attention Performance & Convergence Report

4-way comparison on NVIDIA RTX 5090 (SM 12.0, Blackwell consumer)

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
| Flash Attention | MXFP8 branch (forward-only kernel for SM 12.0) |

## Configurations Tested

| # | Config | Attention Precision | Linear Layer Precision |
|---|---|---|---|
| 1 | **MXFP8 + MXFP8 Attention** | MXFP8 fwd + BF16 SDPA bwd (custom `Block`) | FP8 GEMMs (MXFP8BlockScaling) |
| 2 | **TE + MXFP8BlockScaling** | BF16 SDPA fwd + bwd (`te.TransformerLayer`) | FP8 GEMMs (MXFP8BlockScaling) |
| 3 | **TE + Float8BlockScaling** | BF16 SDPA fwd + bwd (`te.TransformerLayer`) | FP8 GEMMs (Float8BlockScaling) |
| 4 | **BF16 baseline** | BF16 SDPA fwd + bwd (`te.TransformerLayer`) | BF16 GEMMs |

> **Note:** On SM 12.0, TE disables FP8 FlashAttention and FusedAttention entirely
> (`dot_product_attention/utils.py` line ~511). All TE paths use BF16 SDPA for attention.
> FP8 is only applied to linear layers (QKV projection, output projection, MLP).

## 1. Training Throughput

| Config | Avg iter time | Tokens/sec | MFU | Speedup vs BF16 |
|:---|---:|---:|---:|---:|
| TE + MXFP8BlockScaling | 115 ms | 71,000 | 59.1% | **1.55x** |
| TE + Float8BlockScaling | 116 ms | 70,500 | 58.8% | **1.54x** |
| MXFP8 + MXFP8 Attention | 145 ms | 56,500 | 47.1% | 1.23x |
| BF16 baseline | 179 ms | 45,700 | 38.1% | 1.00x |

TE + MXFP8BlockScaling and TE + Float8BlockScaling are effectively identical in throughput.
Both achieve ~59% MFU, a 1.55x speedup over BF16 — entirely from FP8 linear layer GEMMs.

The custom MXFP8 attention path is 26% slower than the TE baselines due to:
- Python-level `quantize_to_mxfp8()` overhead: ~0.50 ms/layer (3 tensors)
- BF16 SDPA backward recompute: ~0.95 ms/layer (runs full fwd+bwd in backward pass)
- Combined: ~1.45 ms/layer × 20 layers = ~29 ms/iter of pure overhead

## 2. Training Convergence

### Loss curve (50 steps)

```
Step  MXFP8+Attn   TE MXFP8  TE Float8Blk      BF16
----  ----------  ---------  ------------  ---------
   1     10.999     10.999        10.999     10.998
   2     10.207     10.072        10.072     10.060
   3      9.113      8.982         8.982      8.991
   4      8.727      8.656         8.656      8.643
   5      8.234      8.580         8.583      8.574
   6      8.123      8.468         8.484      8.492
   7      8.576      7.665         7.681      7.666
   8     11.091      7.769         7.767      7.843
   9     10.898      8.244         8.234      8.253
  10      9.120      9.862         9.908      9.862
  15      7.347      7.040         6.994      7.060
  20      7.530      7.229         7.144      7.194
  25      7.173      6.620         6.593      6.553
  30      7.113      6.405         6.404      6.343
  35      7.040      6.311         6.305      6.271
  40      6.975      6.245         6.252      6.203
  45      7.212      6.243         6.223      6.156
  49      7.082      6.059         6.056      6.017
```

### Key checkpoints

| Step | MXFP8+Attn | TE MXFP8 | TE Float8Blk | BF16 |
|---:|---:|---:|---:|---:|
| 5 | 8.234 | 8.580 | 8.583 | 8.574 |
| 10 | 9.120 | 9.862 | 9.908 | 9.862 |
| 15 | 7.347 | 7.040 | 6.994 | 7.060 |
| 20 | 7.530 | 7.229 | 7.144 | 7.194 |
| 25 | 7.173 | 6.620 | 6.593 | 6.553 |
| 30 | 7.113 | 6.405 | 6.404 | 6.343 |
| 40 | 6.975 | 6.245 | 6.252 | 6.203 |
| **49 (final)** | **7.082** | **6.059** | **6.056** | **6.017** |

### Validation loss (step 40)

| Config | Val loss |
|:---|---:|
| BF16 baseline | 6.292 |
| TE + Float8BlockScaling | 6.335 |
| TE + MXFP8BlockScaling | 6.338 |
| MXFP8 + MXFP8 Attention | 7.110 |

### Gradient norm stability

```
Step  MXFP8+Attn   TE MXFP8  TE Float8Blk      BF16
----  ----------  ---------  ------------  ---------
   7      2.352     10.629        10.640     10.536
   8      5.999      3.398         3.410      3.259
   9     12.692      3.836         3.816      4.255
  10      8.610      3.345         3.421      3.287
  11      2.630      4.751         4.737      4.239
  30      0.466      0.126         0.176      0.158
  40      0.641      0.157         0.258      0.134
  49      1.043      0.149         0.166      0.111
```

The MXFP8 attention path shows a severe loss spike at steps 8–11 (loss jumps from 8.1 to 11.7)
with grad norms spiking to 12.7. The three non-MXFP8-attention paths all show similar grad
norm trajectories that stabilize below 0.2 by step 30. The MXFP8 attention path's grad norms
remain elevated at ~1.0 even at step 49, indicating persistent training instability.

## 3. Operator-Level Analysis

Per-layer timing breakdown from `benchmark_attention.py` (CUDA event timing, 20 runs):

### Custom MXFP8 Block — forward pass operators

| Operator | Time (ms) | % of fwd |
|:---|---:|---:|
| LayerNormMLP (RMSNorm + FC1 + SReLU + FC2) | 0.59 | 32.0% |
| `quantize_to_mxfp8` × 3 (Python torch ops) | 0.53 | 28.5% |
| QKV proj (LayerNormLinear, FP8 GEMM) | 0.26 | 14.3% |
| RoPE × 2 | 0.25 | 13.7% |
| `flash_attn_mxfp8` kernel | 0.21 | 11.4% |
| Output proj (te.Linear, FP8 GEMM) | 0.16 | 8.4% |
| QK RMSNorm × 2 | 0.10 | 5.4% |
| Residual adds × 2 | 0.02 | 1.3% |
| **Measured fwd total** | **1.85** | |
| **Measured fwd+bwd total** | **4.96** | |

Additional backward-only cost:
- BF16 SDPA backward recompute (fwd+bwd inside backward): **0.95 ms/layer**

### Per-layer totals (fwd + bwd)

| Path | Fwd (ms) | Bwd (ms) | Total (ms) | vs fastest |
|:---|---:|---:|---:|---:|
| Custom MXFP8 Block | 1.85 | 3.11 | 4.96 | +32% |
| TE + MXFP8BlockScaling | 1.21 | 2.56 | 3.77 | fastest |
| TE + Float8BlockScaling | 1.35 | 2.58 | 3.93 | +4% |

### Head-to-head kernel comparisons

| Test | Time (ms) | Ratio |
|:---|---:|---:|
| MXFP8 flash attn fwd | 0.21 | 0.74x (26% faster) |
| BF16 SDPA fwd | 0.29 | 1.00x |
| | | |
| Hybrid attn fwd+bwd (MXFP8 fwd + BF16 SDPA bwd) | 1.63 | 1.72x (72% slower) |
| Pure BF16 SDPA fwd+bwd | 0.94 | 1.00x |
| | | |
| `quantize_to_mxfp8` per tensor | 0.18 | — |
| `quantize_to_mxfp8` × 3 (Q,K,V) | 0.50 | — |

The MXFP8 flash attention kernel is 26% faster than BF16 SDPA in isolation, but the
quantization overhead (0.50 ms) and backward recompute cost (0.95 ms) make the total
hybrid approach 72% slower than pure BF16 SDPA.

## 4. Root Cause Analysis

### Why is the custom MXFP8 Block slower?

The 1.45 ms/layer overhead comes from two sources:

1. **Python quantization tax (0.50 ms/layer):** `quantize_to_mxfp8()` performs 7 operations
   per tensor in Python (reshape → amax → log2 → exp2 → clamp → cast → permute). Three
   tensors (Q, K, V) means 21 kernel launches with Python overhead between each.

2. **Backward recompute (0.95 ms/layer):** `flash_attn_mxfp8_func` is forward-only — no
   backward kernel exists. Our `MXFP8Attention.backward()` must run a full BF16 SDPA
   forward + backward to compute gradients, effectively doubling the attention backward cost.

### Why does convergence degrade?

The forward pass computes attention weights using MXFP8 precision, but the backward pass
recomputes them using BF16 SDPA. This creates a **precision mismatch**: gradients are computed
with respect to BF16 attention outputs, not the MXFP8 outputs that were actually used in the
forward pass. The mismatch causes:

- Incorrect gradient signals that accumulate over training steps
- Loss spikes at steps 8–11 (loss jumps from 8.1 → 11.7)
- Persistently elevated gradient norms (~1.0 vs ~0.15 for consistent-precision paths)
- Final loss 1.0 points worse than TE baseline (7.08 vs 6.06)

### Why are TE MXFP8 and Float8BlockScaling identical?

On SM 12.0, TE internally converts Float8BlockScaling to MXFP8 tensors via
`convert_block_scaling_to_mxfp8_tensor`. Both recipes execute the same MXFP8 GEMM kernels
and use the same BF16 SDPA for attention. The only difference is pre-transposition of
`columnwise_data`, which gives Float8BlockScaling a marginal edge in some benchmarks
(not significant at 50 steps).

## 5. Summary

### Performance ranking

| Rank | Config | Iter time | MFU | Final loss |
|---|---|---:|---:|---:|
| 1 | TE + MXFP8BlockScaling | 115 ms | 59% | 6.059 |
| 2 | TE + Float8BlockScaling | 116 ms | 59% | 6.056 |
| 3 | MXFP8 + MXFP8 Attention | 145 ms | 47% | 7.082 |
| 4 | BF16 baseline | 179 ms | 38% | 6.017 |

### What would be needed to make MXFP8 attention competitive

1. **Native MXFP8 backward kernel** (`mha_bwd_mxfp8` in flash-attention)
   - Eliminates the 0.95 ms/layer BF16 SDPA recompute
   - Fixes the fwd/bwd precision mismatch that degrades convergence
   - Expected: ~0.21 ms if backward kernel achieves similar speedup to forward

2. **Fused CUDA quantize kernel** to replace Python `quantize_to_mxfp8`
   - Current: 0.18 ms/tensor (7 separate kernel launches + Python overhead)
   - Expected: <0.05 ms/tensor with a single fused CUDA kernel

3. **Projected impact** (if both are implemented):
   - Attention: ~0.21 (fwd) + 0.21 (bwd) + 0.15 (quantize) ≈ 0.57 ms
   - vs TE's BF16 SDPA: ~0.94 ms fwd+bwd
   - Potential ~39% speedup in attention, translating to ~5–10% end-to-end

### Recommendation

**TE + MXFP8BlockScaling** (or Float8BlockScaling) is the correct choice for SM 12.0 today.
It delivers 1.55x speedup over BF16 with matching convergence quality. The custom MXFP8
attention path should not be used for training until a native MXFP8 backward kernel is
available, as it hurts both performance (26% slower) and convergence (1.0 point worse loss).
