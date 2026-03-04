# Optimization Targets (from Profiling)

## Profiling Summary

Profiled MXFP8 NanoGPT training: B=4, T=2048, 20 layers, 561M params.
Current: ~114ms/iter, ~60% MFU. 10 optimizer steps profiled.

### Per-iteration breakdown (114ms total)
| Component | ms/iter | % |
|---|---|---|
| Backward | 77.1 | 67.8% |
| Forward | 34.4 | 30.3% |
| Optimizer (amortized) | 1.6 | 1.4% |
| Data load | 0.4 | 0.3% |
| Grad clip (amortized) | 0.1 | 0.1% |

### Top CUDA kernels (per optimizer step, 6.8s total GPU time)
| Kernel | Time (ms) | % |
|---|---|---|
| MXFP8 block-scaled GEMMs (cutlass) | 2265 | 33.1% |
| LayerNormMLP backward | 1188 | 17.4% |
| MXFP8 attention backward | 1019 | 14.9% |
| aten::copy_ (dtype conversions) | 589 | 8.6% |
| aten::add_ (residual/grad accum) | 393 | 5.8% |
| MXFP8 quantize kernel | 308 | 4.5% |
| MXFP8 attention forward | 242 | 3.5% |
| cross_entropy_loss | 238 | 3.5% |

## Target 1: Reduce dtype conversion overhead (aten::copy_)

- **Current cost**: 589ms / 63 iters = **9.3ms/iter (8.6%)**
- **Potential saving**: 3-5ms/iter
- **Root cause**: TE linear layers do BF16→FP8 quantization and FP8→BF16 dequantization
  at every layer boundary. Each conversion is a separate `aten::copy_` kernel.
- **Approach**:
  - Investigate if TE can keep activations in FP8 between consecutive layers
    (avoid dequantize→requantize round-trip)
  - Or use `torch.compile` on the full Block to fuse copy kernels
  - Or replace TE linear layers with direct cuBLAS FP8 GEMM calls that skip the copy
- **Effort**: Medium-High (requires either TE internals or custom GEMM wrappers)

## Target 2: MXFP8 attention backward kernel optimization

- **Current cost**: 1019ms / 63 = **16.2ms/iter (14.9%)**
- **Attention fwd**: 242ms / 63 = 3.8ms/iter
- **Bwd/fwd ratio**: 4.2× (typical flash attention is ~2.5×)
- **Potential saving**: If bwd/fwd ratio reduced to 2.5×, saves ~6ms/iter
- **Approach**: Optimize the flash-attention `bwd_mxfp8` kernel in the external project
  (tile sizes, warp scheduling, memory access patterns)
- **Effort**: High (requires CUTLASS/CUDA kernel work in flash-attention repo)

## Target 3: MXFP8 quantization kernel overhead

- **Current cost**: 308ms / 63 = **4.9ms/iter (4.5%)**
- **What**: TE's `mxfp8::quantize_kernel` runs 7686 times per optimizer step
  (~122 calls per micro-batch = ~6 per layer for fwd+bwd quantization)
- **Potential saving**: 2-3ms/iter
- **Approach**:
  - Fuse quantization into the GEMM prologue (TE already does this partially)
  - Or reduce quantization calls by caching FP8 weights across micro-batches
    (weights don't change within a grad accumulation window)
- **Effort**: Medium (TE configuration or custom weight caching)

## Target 4: Command buffer saturation

- **Current cost**: 1032ms self CPU time on "Command Buffer Full"
- **What**: Too many small CUDA kernel launches saturate the command buffer.
  9475 occurrences in one optimizer step = ~150 per micro-batch.
- **Potential saving**: 1-2ms/iter (reduces CPU-side launch overhead)
- **Approach**:
  - `torch.compile` on more of the model to fuse small kernels
  - CUDA graphs for the forward+backward pass (tricky with dynamic shapes)
  - Reduce Python-side kernel dispatch overhead
- **Effort**: Low-Medium (torch.compile) to High (CUDA graphs)

## Target 5: Optimizer step (NorMuon)

- **Current cost**: 98.5ms per step = **1.6ms/iter amortized**
- **Potential saving**: Limited (~0.5ms/iter)
- **Approach**: Overlap optimizer step with next batch's data loading
  or compile the optimizer
- **Effort**: Low
- **Priority**: Low (only 1.4% of total)

## Prioritized Plan

| Priority | Target | Expected saving | Effort |
|---|---|---|---|
| 1 | dtype conversion (aten::copy_) | 3-5ms/iter | Medium-High |
| 2 | MXFP8 attention bwd kernel | ~6ms/iter | High (external project) |
| 3 | MXFP8 quantize kernel | 2-3ms/iter | Medium |
| 4 | Command buffer saturation | 1-2ms/iter | Low-Medium |
| 5 | Optimizer overlap | ~0.5ms/iter | Low |

**Total potential**: 12-16ms/iter → from 114ms to ~98-102ms (~65-70% MFU)

Note: Targets 1 and 3 are related (both involve TE's FP8 quantization pipeline).
A unified approach replacing TE's linear layers with direct MXFP8 GEMM calls
could address both simultaneously.
