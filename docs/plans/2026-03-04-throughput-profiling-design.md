# Design: Full Training Loop Profiling & Throughput Optimization

## Goal

Identify and optimize remaining throughput bottlenecks in MXFP8 NanoGPT training
beyond the already-optimized flash attention path. Current: 114ms/iter, 60% MFU.
Target: close the gap to theoretical peak and find non-attention bottlenecks.

## Context

- MXFP8 flash attention is at near-parity with TE baseline (114ms vs 111ms)
- Attention is 33% of per-layer time; linear layers are 67%
- The full training loop includes overhead beyond the 20 transformer layers:
  embedding, lm_head, optimizer, grad accumulation, data loading, DDP sync

## Approach

Two-phase profiling, then targeted optimization.

### Phase 1: CUDA Event Breakdown

Instrument the full training iteration with CUDA events to measure every component:

| Component | What it measures |
|---|---|
| Data loading | `get_batch()` — CPU→GPU transfer |
| Embedding | `token_embedding_table(idx)` |
| RoPE precompute | `self.rope(T)` + cos/sin (if applicable) |
| Per-layer forward (×20) | Each Block or TransformerLayer |
| Final norm + lm_head | `ln_f(x)` + `lm_head(x)` + softcap + loss |
| Backward pass | `loss.backward()` as a whole |
| Grad clipping | `clip_grad_norm_` |
| Optimizer step | NorMuon `.step()` + scheduler |
| Python loop overhead | Total iter time minus sum of GPU ops |

Method: 3 warmup iterations, 10 timed iterations. CUDA events for GPU timing.
Report mean ± std for each component, plus percentage of total.

### Phase 2: PyTorch Profiler Deep Dive

Run `torch.profiler.profile()` on 3 iterations to get kernel-level detail:
- Top 20 CUDA kernels by total time
- Look for: redundant memory allocations, unnecessary syncs, TE quantize/dequantize
  overhead, kernel launch gaps
- Focus on the top 2-3 bottlenecks identified in Phase 1

### Expected Findings

| Component | Estimated cost | Notes |
|---|---|---|
| 20× layer fwd+bwd | ~80ms | Already optimized |
| Optimizer step | ~10-15ms | NorMuon Triton kernels + SVD-like ops |
| Grad accumulation overhead | ~5-10ms | 63 micro-batches, Python loop overhead |
| lm_head + loss | ~5ms | Large vocab (65536) GEMM + cross entropy |
| Data loading | ~1-2ms | Should overlap with GPU |
| Grad clipping | ~1-2ms | Iterates all params |

## Deliverables

1. `profile_training.py` — standalone script, both phases, self-contained
2. Printed tables: time breakdown + top CUDA kernels
3. Optimization plan based on actual profiling data

## Constraints

- Single GPU (RTX 5090, SM 12.0)
- B=4, T=2048, 63 grad accum steps
- Must not change model architecture or convergence behavior
- Pragmatic: either TE modules or custom, whichever gives best throughput/effort
