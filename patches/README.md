# Patches for MXFP8 on RTX 5090 (SM 12.0)

These patches enable MXFP8BlockScaling training on the NVIDIA RTX 5090 (Blackwell consumer, SM 12.0) using Transformer Engine and a nanogpt-fp8 training script.

## Patches

### `transformer_engine_mxfp8_sm120.patch`

Apply against Transformer Engine 2.13.0.dev (commit `f8449052` or similar).

**Files modified (4):**

| File | Lines | Change |
|---|---|---|
| `pytorch/quantization.py` | -5/+2 | Remove SM 12.0 block from `check_mxfp8_support()` and `get_default_fp8_recipe()`. TE explicitly returned `False` for SM >= 12.0; this patch allows SM 10.0+ (including 12.0). |
| `pytorch/csrc/extensions/gemm.cpp` | +76/+1 | Add `transpose_mxfp8_operand` lambda before `swizzle_scales_for_gemm`. For MXFP8 non-TN GEMMs on SM 12.0: transposes columnwise data `[M,K] -> [K,M]` using TE's optimized `nvte_transpose` (vectorized tiled loads/stores, near-peak bandwidth), transposes columnwise scales, creates new TensorWrapper with transposed data as rowwise, sets `transa=true, transb=false`. Also adds `#include "transformer_engine/transpose.h"`. |
| `pytorch/csrc/quantizer.cpp` | +6/-2 | Disable `optimize_for_gemm` (pre-swizzled scales) for MXFP8 on SM 12.0. Changed `with_gemm_swizzled_scales = this->optimize_for_gemm` to `this->optimize_for_gemm && nvte_is_non_tn_fp8_gemm_supported()` in both `create_tensor()` and `convert_and_update_tensor()`. Required because the gemm.cpp transpose needs natural (unswizzled) scale format. |
| `common/gemm/cublaslt_gemm.cu` | +18/-12 | Fix `CanonicalizeGemmInput()` TN fallback for MXFP8. For operand A: use `A.scale_inv` (rowwise) instead of `A.columnwise_scale_inv` when applying CUBLAS_OP_T to columnwise data. For operand B: same fix with `B.scale_inv`. cuBLASLt MXFP8 TN GEMM expects scales indexed as `scale[row][col//32]` which matches rowwise layout. |

**Performance note:** The data transpose in `gemm.cpp` uses `nvte_transpose` instead of PyTorch's `.t().contiguous()`. The generic PyTorch elementwise kernel achieved only ~2% of peak memory bandwidth, causing a 43ms/step overhead (280 GB of uint8 copies per step). The optimized kernel eliminates this overhead, bringing MXFP8 to match Float8BlockScaling throughput (~116 ms/step).

**How to apply:**

```bash
cd /path/to/TransformerEngine
git apply /path/to/transformer_engine_mxfp8_sm120.patch
```

Then rebuild:

```bash
# Rebuild PyTorch extension (gemm.cpp, quantizer.cpp):
PATH=/usr/bin:/usr/local/bin:/usr/local/cuda/bin:$PATH \
  NVTE_FRAMEWORK=pytorch MAX_JOBS=16 \
  python setup.py build_ext --inplace

# Rebuild common C++ library (cublaslt_gemm.cu):
PATH=/usr/bin:/usr/local/bin:/usr/local/cuda/bin:$PATH \
  cmake --build /path/to/TransformerEngine/build/cmake --parallel 16
```

---

### `nanogpt_fp8_changes.patch`

Apply against the original [alint77/nanogpt-fp8](https://github.com/alint77/nanogpt-fp8) repo.

**Files modified (2):**

| File | Change |
|---|---|
| `train.py` | Environment variable overrides (`USE_FP8`, `BATCH_SIZE`, `MAX_ITERS`, `RECIPE`); recipe selection via env var (default: MXFP8BlockScaling); optimizer switched from `Muon` to `NorMuon` with `weight_decay=0.01, cautious_wd=True`; gains/biases LR changed from `2e-3` to `0.2` (NorMuon internally scales via `rms_norm`). |
| `run.sh` | Changed from 8 GPU (`--nproc-per-node=8`, `OMP_NUM_THREADS=16`) to single GPU (`--nproc-per-node=1`, `OMP_NUM_THREADS=1`). |

**How to apply:**

```bash
cd /path/to/nanogpt-fp8
git apply /path/to/nanogpt_fp8_changes.patch
```

**Usage after applying:**

```bash
# Default (MXFP8BlockScaling, BS=3):
bash run.sh

# Float8BlockScaling, BS=4, 50 steps:
RECIPE=Float8BlockScaling BATCH_SIZE=4 MAX_ITERS=50 bash run.sh

# BF16 baseline (no FP8):
USE_FP8=false BATCH_SIZE=4 MAX_ITERS=50 bash run.sh

# DelayedScaling:
RECIPE=DelayedScaling BATCH_SIZE=4 MAX_ITERS=50 bash run.sh
```

---

## Prerequisites

- NVIDIA RTX 5090 (SM 12.0) or compatible Blackwell consumer GPU
- CUDA Toolkit 13.0+
- PyTorch 2.11.0a0+ (compiled from source with `TORCH_CUDA_ARCH_LIST="9.0;12.0+PTX"`)
- Transformer Engine 2.13.0.dev (compiled from source with `NVTE_CUDA_ARCHS="90;120"`)
- Dion optimizer library (latest commit, includes NorMuon NaN fix)

## Background

cuBLASLt on SM 12.0 only supports TN (transposed-A, normal-B) layout for MXFP8 GEMM. The forward pass uses TN natively, but backward dgrad (NN) and wgrad (NT) layouts are unsupported. The TE patch converts these to TN by transposing MXFP8 columnwise data and scales at GEMM time using `nvte_transpose`. Float8BlockScaling avoids this because its columnwise data is already pre-transposed `[K,M]`, while MXFP8 stores it as `[M,K]` (same shape as original).
