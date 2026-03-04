# Training Loop Profiling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Profile every component of the MXFP8 NanoGPT training loop to identify throughput bottlenecks beyond flash attention.

**Architecture:** Two-phase profiling script. Phase 1 uses CUDA events to time each training loop component (data load, forward sub-components, backward, optimizer, etc). Phase 2 uses `torch.profiler` to get kernel-level detail on the top bottlenecks. Both phases are self-contained in one script that reuses the model/data setup from train.py inline.

**Tech Stack:** PyTorch CUDA events, torch.profiler, TransformerEngine, MXFP8 flash attention

---

### Task 1: Create profiling script scaffold

**Files:**
- Create: `profile_training.py`

**Step 1: Write the script scaffold with imports and model setup**

```python
"""
Full training loop profiler for MXFP8 NanoGPT.

Phase 1: CUDA event timing for every training loop component
Phase 2: PyTorch profiler for kernel-level detail

Usage: source ~/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 profile_training.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import math
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

import sys
sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")
from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func
from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# Initialize DDP (required by train.py's model setup)
init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
data_parallel_group = dist.new_group(backend='nccl')

torch.manual_seed(42)

# Model config (matching train.py)
batch_size = 4
block_size = 2048
n_layer = 20
n_embd = 1280
n_head = 10
dropout = 0.0
vocab_size = 65536
recipe = MXFP8BlockScaling()

WARMUP = 3
TIMED = 10

print(f"Profiling: B={batch_size}, T={block_size}, layers={n_layer}, C={n_embd}, H={n_head}")
```

**Step 2: Run to verify scaffold loads**

Run: `source ~/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 profile_training.py`
Expected: Prints config line, no errors.

**Step 3: Commit**

```bash
git add profile_training.py
git commit -m "feat: add profiling script scaffold"
```

---

### Task 2: Add model, data loader, and optimizer setup

**Files:**
- Modify: `profile_training.py`

**Step 1: Add model class, data loader, and optimizer inline**

Copy the model components inline from train.py (same pattern as benchmark scripts). Add after the config section:

- `te_init_method`, `te_output_layer_init_method` (from train.py:276-284)
- `MXFP8Attention` class (from train.py:287-325)
- `Block` class with torch.compile (from train.py:327-415)
- `LLM` class (from train.py:425-515)
- Data loader setup (reuse `_load_data_shard`, `DistributedDataLoader` from train.py:166-236)
- Optimizer setup: NorMuon with param groups (from train.py:581-628)

Then instantiate:

```python
# Build model
model = LLM().to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank], process_group=data_parallel_group)
raw_model = model.module

# Data
input_bin = 'data/fineweb10B/fineweb_train_*.bin'
train_loader = DistributedDataLoader(input_bin, batch_size, block_size, ddp_rank, ddp_world_size)

# Optimizer (same as train.py)
hidden_weights = [p for p in raw_model.blocks.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in raw_model.blocks.parameters() if p.ndim < 2]
nonhidden_params = [*raw_model.token_embedding_table.parameters(), *raw_model.ln_f.parameters()]
unembedding_params = [*raw_model.lm_head.parameters()]

from dion import NorMuon
param_groups = [
    dict(params=hidden_weights, lr=0.02),
    dict(params=hidden_gains_biases+nonhidden_params, algorithm="adamw", lr=0.2, betas=(0.9, 0.95), weight_decay=0.01),
    dict(params=unembedding_params, algorithm="adamw", lr=0.004, betas=(0.9, 0.95), weight_decay=0.01)
]
optimizer = NorMuon(param_groups, distributed_mesh=data_parallel_group, use_triton=True, weight_decay=0.01, cautious_wd=True)
```

**Step 2: Run to verify model builds**

Run: `source ~/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 profile_training.py`
Expected: Prints config, builds model, no errors.

**Step 3: Commit**

```bash
git add profile_training.py
git commit -m "feat: add model/data/optimizer to profiling script"
```

---

### Task 3: Implement Phase 1 — CUDA event training loop profiler

**Files:**
- Modify: `profile_training.py`

**Step 1: Add the instrumented training loop**

Add Phase 1 after the model setup. This runs the actual training loop with CUDA events around each component:

```python
print("\n" + "=" * 70)
print(" PHASE 1: CUDA EVENT TRAINING LOOP BREAKDOWN")
print("=" * 70)

grad_accum_steps = 63  # matching train.py
total_iters = (WARMUP + TIMED) * grad_accum_steps  # full optimizer steps

# Storage for per-component timings (per micro-batch iteration)
timings = {
    'data_load': [],
    'forward': [],
    'loss_scale': [],
    'backward': [],
    'grad_clip': [],
    'optimizer_step': [],
    'zero_grad': [],
    'total_iter': [],
}

train_loader.reset()
model.train()

for it in range(total_iters):
    step = it // grad_accum_steps
    is_warmup = step < WARMUP
    is_accum_boundary = ((it + 1) % grad_accum_steps == 0)

    model.require_backward_grad_sync = is_accum_boundary

    # --- Data load ---
    s_data = torch.cuda.Event(enable_timing=True)
    e_data = torch.cuda.Event(enable_timing=True)
    s_data.record()
    xb, yb = train_loader.next_batch()
    e_data.record()

    # --- Forward ---
    s_fwd = torch.cuda.Event(enable_timing=True)
    e_fwd = torch.cuda.Event(enable_timing=True)
    s_fwd.record()
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        with te.autocast(enabled=True, recipe=recipe, amax_reduction_group=data_parallel_group):
            _, loss = model(xb, yb, is_first_microbatch=(it % grad_accum_steps == 0))
    e_fwd.record()

    # --- Loss scale ---
    s_scale = torch.cuda.Event(enable_timing=True)
    e_scale = torch.cuda.Event(enable_timing=True)
    s_scale.record()
    scaled_loss = loss / grad_accum_steps
    e_scale.record()

    # --- Backward ---
    s_bwd = torch.cuda.Event(enable_timing=True)
    e_bwd = torch.cuda.Event(enable_timing=True)
    s_bwd.record()
    scaled_loss.backward()
    e_bwd.record()

    if is_accum_boundary:
        # --- Grad clip ---
        s_clip = torch.cuda.Event(enable_timing=True)
        e_clip = torch.cuda.Event(enable_timing=True)
        s_clip.record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        e_clip.record()

        # --- Optimizer step ---
        s_opt = torch.cuda.Event(enable_timing=True)
        e_opt = torch.cuda.Event(enable_timing=True)
        s_opt.record()
        for group in optimizer.param_groups:
            group["mu"] = min(it / 300, 1) * 0.1 + 0.85
        optimizer.step()
        e_opt.record()

        # --- Zero grad ---
        s_zero = torch.cuda.Event(enable_timing=True)
        e_zero = torch.cuda.Event(enable_timing=True)
        s_zero.record()
        optimizer.zero_grad(set_to_none=True)
        e_zero.record()

    # Sync and collect (only for timed steps)
    torch.cuda.synchronize()
    if not is_warmup:
        timings['data_load'].append(s_data.elapsed_time(e_data))
        timings['forward'].append(s_fwd.elapsed_time(e_fwd))
        timings['loss_scale'].append(s_scale.elapsed_time(e_scale))
        timings['backward'].append(s_bwd.elapsed_time(e_bwd))
        if is_accum_boundary:
            timings['grad_clip'].append(s_clip.elapsed_time(e_clip))
            timings['optimizer_step'].append(s_opt.elapsed_time(e_opt))
            timings['zero_grad'].append(s_zero.elapsed_time(e_zero))

# Print results
print(f"\nTimed {TIMED} optimizer steps × {grad_accum_steps} micro-batches = {TIMED * grad_accum_steps} iterations\n")

print(f"{'Component':<25} {'Per micro-batch (ms)':>20} {'Per opt step (ms)':>20} {'% of step':>10}")
print("-" * 77)

# Per micro-batch components (averaged, then multiplied by grad_accum_steps for per-step)
step_total = 0.0
for key in ['data_load', 'forward', 'loss_scale', 'backward']:
    per_mb = float(np.mean(timings[key]))
    per_step = per_mb * grad_accum_steps
    step_total += per_step
    print(f"  {key:<23} {per_mb:>19.3f} {per_step:>19.2f}")

# Per-step components (only run once per optimizer step)
for key in ['grad_clip', 'optimizer_step', 'zero_grad']:
    if timings[key]:
        per_step = float(np.mean(timings[key]))
        step_total += per_step
        print(f"  {key:<23} {'—':>20} {per_step:>19.2f}")

print(f"  {'-'*23} {'-'*20} {'-'*20}")
print(f"  {'SUM':<23} {'':>20} {step_total:>19.2f}")

# Compare to wall-clock iter time
# Each "step" in train.py prints is grad_accum_steps iterations
# So step_total should approximate avg_iter_time * grad_accum_steps
print(f"\n  Estimated iter time (sum / {grad_accum_steps}): {step_total / grad_accum_steps:.2f} ms")
print(f"  For reference: actual training shows ~114 ms/iter")
```

**Step 2: Run Phase 1**

Run: `source ~/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 profile_training.py`
Expected: Timing table with per-component breakdown.

**Step 3: Commit**

```bash
git add profile_training.py
git commit -m "feat: implement Phase 1 CUDA event profiler"
```

---

### Task 4: Implement Phase 2 — PyTorch profiler

**Files:**
- Modify: `profile_training.py`

**Step 1: Add PyTorch profiler section after Phase 1**

```python
print("\n" + "=" * 70)
print(" PHASE 2: PYTORCH PROFILER — TOP CUDA KERNELS")
print("=" * 70)

from torch.profiler import profile, ProfilerActivity

train_loader.reset()

# Warmup
for _ in range(grad_accum_steps):
    xb, yb = train_loader.next_batch()
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        with te.autocast(enabled=True, recipe=recipe, amax_reduction_group=data_parallel_group):
            _, loss = model(xb, yb, is_first_microbatch=True)
    (loss / grad_accum_steps).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()

# Profile 1 full optimizer step (63 micro-batches + optimizer)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for mb in range(grad_accum_steps):
        model.require_backward_grad_sync = (mb == grad_accum_steps - 1)
        xb, yb = train_loader.next_batch()
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            with te.autocast(enabled=True, recipe=recipe, amax_reduction_group=data_parallel_group):
                _, loss = model(xb, yb, is_first_microbatch=(mb == 0))
        (loss / grad_accum_steps).backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

torch.cuda.synchronize()

# Print top 30 CUDA kernels
print("\nTop 30 CUDA kernels by total GPU time:\n")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

# Print top 10 by self CPU time (catches Python overhead)
print("\nTop 10 by self CPU time:\n")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

# Cleanup
destroy_process_group()
```

**Step 2: Run full profiler**

Run: `source ~/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 profile_training.py`
Expected: Phase 1 timing table + Phase 2 kernel tables.

**Step 3: Commit**

```bash
git add profile_training.py
git commit -m "feat: implement Phase 2 PyTorch profiler"
```

---

### Task 5: Run profiling and analyze results

**Files:**
- No new files — analysis of output

**Step 1: Run the complete profiling script**

Run: `source ~/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 profile_training.py 2>&1 | tee profile_output.txt`

**Step 2: Analyze Phase 1 output**

Look at the timing table. Key questions:
- What % is forward vs backward vs optimizer?
- Is data loading overlapped or blocking?
- How much does grad_clip + optimizer_step cost per step?
- Does the sum match the ~114ms × 63 ≈ 7182ms per optimizer step?

**Step 3: Analyze Phase 2 output**

Look at top CUDA kernels. Key questions:
- Are there unexpected kernels (memory allocation, sync, quantize)?
- What fraction of time is GEMM kernels vs everything else?
- Are there CPU bottlenecks visible in the self_cpu_time table?

**Step 4: Commit output for reference**

```bash
git add profile_output.txt
git commit -m "docs: add profiling output for analysis"
```

---

### Task 6: Write optimization recommendations

**Files:**
- Create: `docs/plans/2026-03-04-optimization-targets.md`

**Step 1: Based on profiling results, document the top 3 optimization targets**

Template:

```markdown
# Optimization Targets (from profiling)

## Profiling Summary
[Insert key numbers from Phase 1 and Phase 2]

## Target 1: [Name]
- Current cost: X ms/iter
- Potential saving: Y ms/iter
- Approach: [specific technique]
- Effort: [low/medium/high]

## Target 2: [Name]
...

## Target 3: [Name]
...

## Prioritized Plan
1. [Target] — [expected saving] — [effort]
2. ...
3. ...
```

**Step 2: Commit**

```bash
git add docs/plans/2026-03-04-optimization-targets.md
git commit -m "docs: add optimization targets from profiling"
```
