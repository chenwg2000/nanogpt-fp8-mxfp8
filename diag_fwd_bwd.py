"""
Diagnostic: isolate whether MXFP8 convergence issue is in forward or backward pass.

Strategy:
  - Create a single Linear layer with identical weights.
  - Run forward + backward in 3 modes: BF16, Float8BlockScaling, MXFP8BlockScaling.
  - Compare:
      (a) Forward output difference vs BF16  → forward pass error
      (b) Weight gradient difference vs BF16 → backward pass error
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8BlockScaling, MXFP8BlockScaling

torch.manual_seed(42)
device = "cuda"

IN_FEATURES  = 1280
OUT_FEATURES = 1280
BATCH        = 64   # large enough to stress quantization

# ── helpers ──────────────────────────────────────────────────────────────────

def make_layer(seed=0):
    torch.manual_seed(seed)
    layer = te.Linear(IN_FEATURES, OUT_FEATURES, bias=False).to(device, dtype=torch.bfloat16)
    return layer

def run(recipe, x, w_data):
    """Forward + backward with a given recipe (None = BF16)."""
    layer = make_layer()
    with torch.no_grad():
        layer.weight.copy_(w_data)

    x_in = x.clone().requires_grad_(False)

    if recipe is None:
        # Pure BF16
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = layer(x_in)
            loss = out.mean()
        loss.backward()
    else:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                out = layer(x_in)
                loss = out.mean()
        loss.backward()

    return out.detach().float(), layer.weight.grad.detach().float()

# ── reference weights and input ──────────────────────────────────────────────

torch.manual_seed(0)
ref_weight = torch.randn(OUT_FEATURES, IN_FEATURES, device=device, dtype=torch.bfloat16)
x = torch.randn(BATCH, IN_FEATURES, device=device, dtype=torch.bfloat16)

# ── run three modes ───────────────────────────────────────────────────────────

out_bf16,   grad_bf16   = run(None,                   x, ref_weight)
out_f8blk,  grad_f8blk  = run(Float8BlockScaling(),   x, ref_weight)
out_mxfp8,  grad_mxfp8  = run(MXFP8BlockScaling(),    x, ref_weight)

# ── report ────────────────────────────────────────────────────────────────────

def stats(name, tensor, ref):
    diff = (tensor - ref).abs()
    rel  = diff / (ref.abs() + 1e-6)
    print(f"  {name}: max_abs={diff.max():.5f}  mean_abs={diff.mean():.5f}  "
          f"mean_rel={rel.mean()*100:.2f}%  std_ref={ref.std():.4f}")

print("=" * 60)
print("FORWARD OUTPUT  diff vs BF16")
print("=" * 60)
stats("Float8BlockScaling", out_f8blk,  out_bf16)
stats("MXFP8BlockScaling ", out_mxfp8,  out_bf16)

print()
print("=" * 60)
print("WEIGHT GRADIENT  diff vs BF16")
print("=" * 60)
stats("Float8BlockScaling", grad_f8blk, grad_bf16)
stats("MXFP8BlockScaling ", grad_mxfp8, grad_bf16)

print()
print("=" * 60)
print("COSINE SIMILARITY of weight gradients vs BF16")
print("=" * 60)
def cosine(a, b):
    a, b = a.flatten(), b.flatten()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-9)

print(f"  Float8BlockScaling: {cosine(grad_f8blk,  grad_bf16):.6f}")
print(f"  MXFP8BlockScaling : {cosine(grad_mxfp8, grad_bf16):.6f}")
