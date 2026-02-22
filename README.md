# MXFP8 Training on RTX 5090 (SM 12.0)

Enabling, fixing, and benchmarking MXFP8 (Microscaling FP8) training on NVIDIA RTX 5090 with Transformer Engine.

This is a fork of [alint77/nanogpt-fp8](https://github.com/alint77/nanogpt-fp8) — a 561M-param decoder-only transformer trained on FineWeb10B. The original repo targets B200 with FSDP2. This fork focuses on **single-GPU RTX 5090** with MXFP8 support that required patching Transformer Engine.

## What's New

MXFP8BlockScaling was software-blocked on SM 12.0 in Transformer Engine, and the backward pass produced wrong gradients due to cuBLASLt's TN-only layout constraint for MXFP8 GEMM. This project:

1. **Built PyTorch 2.11.0a0 + TE 2.13.0.dev from source** with CUDA 13.0 / SM 12.0 support
2. **Patched TE** (4 files) to enable MXFP8 and fix backward pass correctness
3. **Verified numerical accuracy**: MXFP8 weight gradient cosine = 0.999611 vs BF16
4. **Benchmarked 4 recipes**: BF16, DelayedScaling, Float8BlockScaling, MXFP8BlockScaling

## Benchmark Results (RTX 5090, 561M model, BS=4)

| Recipe | ms/step | MFU | Speedup vs BF16 |
|---|---|---|---|
| BF16 (no FP8) | ~175 | ~39% | 1.00x |
| DelayedScaling | ~124 | ~55% | 1.41x |
| **Float8BlockScaling** | **~113** | **~61%** | **1.55x** |
| MXFP8BlockScaling | ~159 | ~43% | 1.10x |

Float8BlockScaling is fastest because it pre-transposes columnwise data and pre-swizzles scales at quantization time. MXFP8's slower dual-quantization kernel is the bottleneck (not GEMM-time overhead).

See [report_complete.pdf](report_complete.pdf) for the full analysis.

---

## Reproducing This Work

### Prerequisites

- NVIDIA RTX 5090 (SM 12.0, 32 GB) or similar Blackwell consumer GPU
- Ubuntu 22.04+ with GCC 12
- CUDA Toolkit 13.0+ installed at `/usr/local/cuda`
- ~60 GB disk space (PyTorch source + build + TE source + build + dataset)

### Step 1: Create Python Environment

```bash
# Using uv (fast) or python -m venv
uv venv ~/.venv --python 3.10
source ~/.venv/bin/activate

# Install build tools
pip install cmake ninja numpy
```

### Step 2: Build PyTorch from Source

```bash
git clone --depth 1 https://github.com/pytorch/pytorch ~/pytorch
cd ~/pytorch
git submodule update --init --recursive

# Fix GCC 12 ICE (if needed)
# In torch/headeronly/macros/Macros.h, lines 201-202:
# Change static_cast<bool>(expr) to !!(expr) in C10_LIKELY/C10_UNLIKELY

CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST="9.0;12.0+PTX" \
MAX_JOBS=$(nproc) \
BUILD_TEST=0 \
pip install --no-build-isolation -e .
```

**Gotcha**: System cmake must be >= 3.27. If not:
```bash
ln -sf ~/.venv/lib/python3.10/site-packages/cmake/data/bin/cmake ~/.local/bin/cmake
export PATH=~/.local/bin:$PATH
```

### Step 3: Build Transformer Engine from Source

```bash
git clone https://github.com/NVIDIA/TransformerEngine ~/te/TransformerEngine
cd ~/te/TransformerEngine
```

Apply the MXFP8 patch:
```bash
git apply /path/to/patches/transformer_engine_mxfp8_sm120.patch
```

Build:
```bash
CUDA_HOME=/usr/local/cuda \
NVTE_FRAMEWORK=pytorch \
NVTE_CUDA_ARCHS="90;120" \
CUDNN_PATH=~/.venv/lib/python3.10/site-packages/nvidia/cudnn \
MAX_JOBS=$(nproc) \
pip install --no-build-isolation -e .
```

**Gotcha**: `NVTE_CUDA_ARCHS=120` alone fails (cmake removes it as "special"). Must include `"90;120"`.

Remove conflicting packages if present:
```bash
pip uninstall -y transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch
# Also uninstall any nvidia-*-cu12 packages that conflict with CUDA 13
```

### Step 4: Install Training Dependencies

```bash
pip install wandb datasets tqdm ninja setuptools
pip install --no-deps "dion @ git+https://github.com/microsoft/dion.git"
```

**Important**: Install dion with `--no-deps` to avoid pulling in pip PyTorch wheels that would overwrite your source build. The latest dion commit includes a NaN fix for zero-initialized weights.

### Step 5: Clone This Repo and Get Data

```bash
git clone https://github.com/YOUR_USERNAME/nanogpt-fp8-mxfp8 ~/nanogpt-fp8
cd ~/nanogpt-fp8

# Download FineWeb10B (9 shards = 900M tokens)
python data/cachedfineweb10b.py 9
```

### Step 6: Verify MXFP8 Works

```bash
~/.venv/bin/python diag_fwd_bwd.py
```

Expected output:
```
COSINE SIMILARITY of weight gradients vs BF16
  Float8BlockScaling: 0.999611
  MXFP8BlockScaling : 0.999611
```

### Step 7: Run Training

```bash
# Default: MXFP8BlockScaling, BS=3
bash run.sh

# Float8BlockScaling, BS=4, 50 steps:
RECIPE=Float8BlockScaling BATCH_SIZE=4 MAX_ITERS=50 bash run.sh

# BF16 baseline:
USE_FP8=false BATCH_SIZE=4 MAX_ITERS=50 bash run.sh

# DelayedScaling:
RECIPE=DelayedScaling BATCH_SIZE=4 MAX_ITERS=50 bash run.sh
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RECIPE` | `MXFP8BlockScaling` | FP8 recipe: `Float8BlockScaling`, `MXFP8BlockScaling`, `DelayedScaling` |
| `BATCH_SIZE` | `3` | Micro-batch size per GPU |
| `MAX_ITERS` | `5000` | Number of optimizer steps |
| `USE_FP8` | `True` | Set to `false` for BF16 baseline |

## Repository Structure

```
nanogpt-fp8/
├── train.py                  # Main training script
├── diag_fwd_bwd.py           # Diagnostic: FP8 correctness test
├── run.sh                    # Launch script (single GPU)
├── data/
│   └── cachedfineweb10b.py   # Dataset downloader
├── patches/
│   ├── README.md             # Patch documentation
│   ├── transformer_engine_mxfp8_sm120.patch  # TE fix (4 files)
│   └── nanogpt_fp8_changes.patch             # train.py changes
├── report_complete.md        # Full report (markdown)
├── report_complete.pdf       # Full report (PDF with charts)
├── loss_curves.png           # Training loss comparison chart
└── gen_loss_plot.py          # Script to regenerate loss chart
```

## What the TE Patch Does

The patch modifies 4 files in Transformer Engine to enable MXFP8 on SM 12.0:

1. **`quantization.py`** — Removes the Python guard that blocked MXFP8 on SM >= 12.0
2. **`gemm.cpp`** — Adds a 68-line transpose routine that converts MXFP8 non-TN GEMMs (backward dgrad NN, wgrad NT) to TN layout by transposing columnwise data `[M,K] -> [K,M]` and scales
3. **`quantizer.cpp`** — Disables pre-swizzled scales on SM 12.0 so the gemm.cpp transpose gets natural-format scales to work with
4. **`cublaslt_gemm.cu`** — Fixes the scale_inv pointer selection in the existing TN fallback (use rowwise scales, not columnwise)

See [patches/README.md](patches/README.md) for detailed per-file documentation.

## Troubleshooting

**`RuntimeError: Multiple libcudart libraries found`**: Uninstall all `nvidia-*-cu12` pip packages. They conflict with CUDA 13.

**`NVTE_CUDA_ARCHS=120` produces empty cmake list**: Use `"90;120"` instead. cmake treats 120 as special and filters it out if alone.

**NaN after first optimizer step with NorMuon**: Update dion to the latest git commit. Older versions have a division-by-zero bug when `norm_U_new` is zero (from zero-initialized weights).

**TE import fails after PyTorch rebuild**: TE must be rebuilt whenever PyTorch is rebuilt (ABI coupling).

**GCC 12 Internal Compiler Error**: Patch `torch/headeronly/macros/Macros.h` — change `static_cast<bool>(expr)` to `!!(expr)` in `C10_LIKELY`/`C10_UNLIKELY` macros.

## Credits

- Original repo: [alint77/nanogpt-fp8](https://github.com/alint77/nanogpt-fp8)
- Architecture: [nanochat](https://github.com/kellerjordan/nanochat) by Keller Jordan
- Optimizer: [Dion/NorMuon](https://github.com/microsoft/dion) by Microsoft
- Dataset: [FineWeb10B](https://huggingface.co/datasets/HuggingFaceFW/fineweb) via modded-nanogpt
