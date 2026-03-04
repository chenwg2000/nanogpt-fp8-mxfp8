"""
Profile MXFP8 NanoGPT training: CUDA event breakdown + PyTorch profiler.

Usage:
    source ~/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 profile_training.py
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# MXFP8 Flash Attention imports
import sys
sys.path.insert(0, "/home/nanogpt/prj/fp8_flashattention/flash-attention/hopper")
from flash_attn_interface import flash_attn_mxfp8_func, flash_attn_mxfp8_bwd_func
from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

# ---- Hyperparameters (matching train.py) ----
batch_size = 4
block_size = 2048
total_batch_size = 512000
n_layer = 20
n_embd = n_layer * 64  # 1280
n_head = max(1, (n_embd + 127) // 128)  # 10
dropout = 0.0
vocab_size = 65536
learning_rate = 1e-3
device = 'cuda'

input_bin = 'data/fineweb10B/fineweb_train_*.bin'

# Profiling constants
WARMUP_STEPS = 3
TIMED_STEPS = 10

# ---- DDP Init ----
init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0
data_parallel_group = dist.new_group(backend='nccl')

grad_accum_steps = max(1, math.ceil(total_batch_size / (batch_size * ddp_world_size * block_size)))
total_batch_size = batch_size * grad_accum_steps * ddp_world_size * block_size

recipe = MXFP8BlockScaling()

torch.manual_seed(1337 + ddp_rank)

# ---- Data loading (copied from train.py) ----

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# ---- Weight init functions (copied from train.py) ----

def te_init_method(weight):
    fan_out = weight.size(0)
    fan_in = weight.size(1)
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    torch.nn.init.normal_(weight, mean=0.0, std=std)

def te_output_layer_init_method(weight):
    torch.nn.init.zeros_(weight)


# ---- MXFP8 Attention (copied from train.py) ----

class MXFP8Attention(torch.autograd.Function):
    """MXFP8 flash attention forward and backward."""

    @staticmethod
    def forward(ctx, q, k, v):
        B, T, H, D = q.shape
        q_fp8 = q.to(torch.float8_e4m3fn)
        k_fp8 = k.to(torch.float8_e4m3fn)
        v_fp8 = v.to(torch.float8_e4m3fn)
        identity_scale = torch.full(
            (B, H, T, D // 32), 127, dtype=torch.uint8, device=q.device)
        out, softmax_lse = flash_attn_mxfp8_func(
            q_fp8, k_fp8, v_fp8,
            identity_scale, identity_scale, identity_scale,
            causal=True,
        )
        ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out, softmax_lse, identity_scale)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q_fp8, k_fp8, v_fp8, out, softmax_lse, identity_scale = ctx.saved_tensors
        dq, dk, dv = flash_attn_mxfp8_bwd_func(
            grad_output, q_fp8, k_fp8, v_fp8,
            out, softmax_lse,
            identity_scale, identity_scale,
            causal=True,
        )
        return dq, dk, dv


# ---- Block (copied from train.py) ----

class Block(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, num_attention_heads):
        super().__init__()
        self.head_dim = hidden_size // num_attention_heads
        self.num_heads = num_attention_heads
        self.qkv_proj = te.LayerNormLinear(
            hidden_size, 3 * hidden_size,
            normalization="RMSNorm",
            bias=False,
            init_method=te_init_method,
        )
        self.q_norm = te.RMSNorm(self.head_dim)
        self.k_norm = te.RMSNorm(self.head_dim)
        self.out_proj = te.Linear(
            hidden_size, hidden_size,
            bias=False,
            init_method=te_output_layer_init_method,
        )
        self.mlp = te.LayerNormMLP(
            hidden_size, ffn_hidden_size,
            normalization="RMSNorm",
            activation="srelu",
            bias=False,
            init_method=te_init_method,
            output_layer_init_method=te_output_layer_init_method,
        )
        self._pre_attention = torch.compile(self._pre_attention)

    def _pre_attention(self, q, k, v, rotary_pos_emb):
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb(q, rotary_pos_emb, tensor_format="bshd")
            k = apply_rotary_pos_emb(k, rotary_pos_emb, tensor_format="bshd")
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k, v

    def forward(self, x, rotary_pos_emb=None, is_first_microbatch=None):
        B, T, C = x.shape
        residual = x
        qkv = self.qkv_proj(x, is_first_microbatch=is_first_microbatch)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)
        q, k, v = self._pre_attention(q, k, v, rotary_pos_emb)
        attn_out = MXFP8Attention.apply(q, k, v)
        attn_out = attn_out.reshape(B, T, C)
        attn_out = self.out_proj(attn_out, is_first_microbatch=is_first_microbatch)
        x = residual + attn_out
        residual = x
        mlp_out = self.mlp(x, is_first_microbatch=is_first_microbatch)
        x = residual + mlp_out
        return x


# ---- LLM (copied from train.py, USE_MXFP8_FLASH_ATTN=True path only) ----

class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.rope = te.RotaryPositionEmbedding(
            dim=n_embd // n_head,
            pretrained_max_position_embeddings=block_size
        )
        self.blocks = nn.ModuleDict({f"block_{i}": Block(
            hidden_size=n_embd,
            ffn_hidden_size=4*n_embd,
            num_attention_heads=n_head,
        ) for i in range(n_layer)})
        self.ln_f = te.RMSNorm(n_embd)
        self.lm_head = te.Linear(n_embd, vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)
        if hasattr(self.lm_head, 'weight'):
            torch.nn.init.zeros_(self.lm_head.weight)
        if self.token_embedding_table.weight.device.type == "cuda":
            self.token_embedding_table.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def forward(self, idx, targets=None, is_first_microbatch=False):
        B, T = idx.shape
        rotary_pos_emb = self.rope(T)
        x = self.token_embedding_table(idx)
        for i in range(n_layer):
            x = self.blocks[f"block_{i}"](x, rotary_pos_emb=rotary_pos_emb,
                                          is_first_microbatch=is_first_microbatch)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        logits = 30.0 * torch.tanh(logits / 30.0).to(torch.bfloat16)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


# ---- Build model + optimizer (matching train.py) ----

def print0(*args, **kwargs):
    if master_process:
        print(*args, **kwargs)

print0(f"grad_accum_steps = {grad_accum_steps}")
print0(f"batch_size = {batch_size}, block_size = {block_size}")
print0(f"n_layer = {n_layer}, n_embd = {n_embd}, n_head = {n_head}")

model = LLM().to(device)
num_params = sum(p.numel() for p in model.parameters())
print0(f"{num_params/1e6:.1f}M parameters")

# Wrap with DDP
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[ddp_local_rank],
    process_group=data_parallel_group,
)
raw_model = model.module

# Create param groups (matching train.py)
hidden_weights = [p for p in raw_model.blocks.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in raw_model.blocks.parameters() if p.ndim < 2]
nonhidden_params = [*raw_model.token_embedding_table.parameters(), *raw_model.ln_f.parameters()]
unembedding_params = [*raw_model.lm_head.parameters()]

param_groups = [
    dict(params=hidden_weights, lr=0.02),
    dict(params=hidden_gains_biases+nonhidden_params, algorithm="adamw", lr=0.2, betas=(0.9, 0.95), weight_decay=0.01),
    dict(params=unembedding_params, algorithm="adamw", lr=0.004, betas=(0.9, 0.95), weight_decay=0.01),
]

from dion import NorMuon

optimizer = NorMuon(param_groups,
                    distributed_mesh=data_parallel_group,
                    use_triton=True,
                    weight_decay=0.01,
                    cautious_wd=True)

# Data loader
train_loader = DistributedDataLoader(input_bin, batch_size, block_size, ddp_rank, ddp_world_size)
print0(f"Training DataLoader: {train_loader.ntok_total} tokens across {len(train_loader.files)} files")


# ---- Muon momentum scheduler (from train.py) ----

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


# ---- Helper: create CUDA event pair ----

def cuda_event_pair():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


# ==============================================================================
# Phase 1: CUDA Event Training Loop Breakdown
# ==============================================================================

def phase1_cuda_event_breakdown():
    print0("\n" + "="*80)
    print0("PHASE 1: CUDA Event Training Loop Breakdown")
    print0("="*80)
    print0(f"  Warmup steps: {WARMUP_STEPS}, Timed steps: {TIMED_STEPS}")
    print0(f"  grad_accum_steps: {grad_accum_steps}")
    print0(f"  micro-batch shape: ({batch_size}, {block_size})")
    print0("")

    total_steps = WARMUP_STEPS + TIMED_STEPS

    # Accumulators for timed steps only (per micro-batch components)
    data_load_times = []
    forward_times = []
    loss_scale_times = []
    backward_times = []

    # Accumulators for per-optimizer-step components
    grad_clip_times = []
    optim_step_times = []
    zero_grad_times = []

    # Full optimizer step wall time
    full_step_times = []

    train_loader.reset()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step_idx in range(total_steps):
        is_timed = step_idx >= WARMUP_STEPS

        # Per-step accumulators for micro-batch timings
        step_data_load = []
        step_forward = []
        step_loss_scale = []
        step_backward = []

        # Full step timing
        s_step_start, s_step_end = cuda_event_pair()
        s_step_start.record()

        for micro in range(grad_accum_steps):
            global_iter = step_idx * grad_accum_steps + micro
            is_last_micro = (micro == grad_accum_steps - 1)

            # -- data_load --
            s0, e0 = cuda_event_pair()
            s0.record()
            xb, yb = train_loader.next_batch()
            e0.record()

            # DDP grad sync control
            model.require_backward_grad_sync = is_last_micro

            # -- forward --
            s1, e1 = cuda_event_pair()
            s1.record()
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                with te.autocast(enabled=True, recipe=recipe, amax_reduction_group=data_parallel_group):
                    _, loss = model(xb, yb, is_first_microbatch=(micro == 0))
            e1.record()

            # -- loss_scale --
            s2, e2 = cuda_event_pair()
            s2.record()
            scaled_loss = loss / grad_accum_steps
            e2.record()

            # -- backward --
            s3, e3 = cuda_event_pair()
            s3.record()
            scaled_loss.backward()
            e3.record()

            if is_timed:
                # Sync to get accurate times for this micro-batch
                e3.synchronize()
                step_data_load.append(s0.elapsed_time(e0))
                step_forward.append(s1.elapsed_time(e1))
                step_loss_scale.append(s2.elapsed_time(e2))
                step_backward.append(s3.elapsed_time(e3))

        # -- grad_clip --
        sg, eg = cuda_event_pair()
        sg.record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        eg.record()

        # -- optimizer_step --
        so, eo = cuda_event_pair()
        so.record()
        for group in optimizer.param_groups:
            group["mu"] = get_muon_momentum(step_idx)
        optimizer.step()
        eo.record()

        # -- zero_grad --
        sz, ez = cuda_event_pair()
        sz.record()
        optimizer.zero_grad(set_to_none=True)
        ez.record()

        s_step_end.record()

        if is_timed:
            # Sync to get optimizer-level times
            s_step_end.synchronize()
            grad_clip_times.append(sg.elapsed_time(eg))
            optim_step_times.append(so.elapsed_time(eo))
            zero_grad_times.append(sz.elapsed_time(ez))
            full_step_times.append(s_step_start.elapsed_time(s_step_end))

            data_load_times.extend(step_data_load)
            forward_times.extend(step_forward)
            loss_scale_times.extend(step_loss_scale)
            backward_times.extend(step_backward)

        if master_process:
            tag = "TIMED" if is_timed else "WARMUP"
            print(f"  [{tag}] step {step_idx}: loss = {loss.item():.4f}")

    # ---- Print results ----
    print0("\n" + "-"*70)
    print0("Per Micro-Batch Breakdown (averaged over timed steps)")
    print0("-"*70)

    def stats(arr, label):
        a = np.array(arr)
        print0(f"  {label:20s}  mean={a.mean():8.3f} ms  std={a.std():7.3f} ms  "
               f"min={a.min():8.3f} ms  max={a.max():8.3f} ms")

    stats(data_load_times, "data_load")
    stats(forward_times, "forward")
    stats(loss_scale_times, "loss_scale")
    stats(backward_times, "backward")

    per_micro_total = (np.array(data_load_times) + np.array(forward_times) +
                       np.array(loss_scale_times) + np.array(backward_times))
    stats(per_micro_total.tolist(), "micro_batch_total")

    print0("\n" + "-"*70)
    print0("Per Optimizer Step Breakdown (averaged over timed steps)")
    print0("-"*70)

    stats(grad_clip_times, "grad_clip")
    stats(optim_step_times, "optimizer_step")
    stats(zero_grad_times, "zero_grad")
    stats(full_step_times, "full_step_wall")

    # Estimated iter time = full_step_wall / grad_accum_steps
    full_arr = np.array(full_step_times)
    est_iter = full_arr / grad_accum_steps
    print0(f"\n  Estimated iter time (full_step / {grad_accum_steps}):")
    print0(f"    mean = {est_iter.mean():.2f} ms   (compare to ~114 ms actual)")
    print0(f"    min  = {est_iter.min():.2f} ms")
    print0(f"    max  = {est_iter.max():.2f} ms")
    print0("")


# ==============================================================================
# Phase 2: PyTorch Profiler
# ==============================================================================

def phase2_pytorch_profiler():
    print0("\n" + "="*80)
    print0("PHASE 2: PyTorch Profiler (1 warmup step + 1 profiled step)")
    print0("="*80 + "\n")

    from torch.profiler import profile, ProfilerActivity

    train_loader.reset()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    def run_one_optimizer_step(step_idx):
        """Run one full optimizer step (grad_accum_steps micro-batches + optimizer)."""
        for micro in range(grad_accum_steps):
            is_last_micro = (micro == grad_accum_steps - 1)
            xb, yb = train_loader.next_batch()
            model.require_backward_grad_sync = is_last_micro

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                with te.autocast(enabled=True, recipe=recipe, amax_reduction_group=data_parallel_group):
                    _, loss = model(xb, yb, is_first_microbatch=(micro == 0))
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for group in optimizer.param_groups:
            group["mu"] = get_muon_momentum(step_idx)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.item()

    # Warmup step (outside profiler)
    loss_val = run_one_optimizer_step(0)
    print0(f"  Warmup step: loss = {loss_val:.4f}")
    torch.cuda.synchronize()

    # Profiled step
    print0("  Profiling 1 optimizer step...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        loss_val = run_one_optimizer_step(1)
    print0(f"  Profiled step: loss = {loss_val:.4f}\n")

    # Top 30 CUDA kernels by cuda_time_total
    print0("-"*70)
    print0("Top 30 CUDA Kernels by cuda_time_total")
    print0("-"*70)
    print0(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Top 10 by self_cpu_time_total
    print0("\n" + "-"*70)
    print0("Top 10 by self_cpu_time_total")
    print0("-"*70)
    print0(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    phase1_cuda_event_breakdown()
    phase2_pytorch_profiler()

    print0("\n" + "="*80)
    print0("Profiling complete.")
    print0("="*80)

    destroy_process_group()
