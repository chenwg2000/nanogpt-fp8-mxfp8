#!/usr/bin/env python3
"""Generate loss curve plot from benchmark data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Loss data from 50-step benchmarks (BS=4, source-compiled)
steps = list(range(1, 50))

bf16 = [10.9984, 10.0601, 8.9909, 8.6433, 8.5735, 8.4919, 7.6662, 7.8454, 8.2496, 9.8594,
        7.7713, 7.5519, 7.0706, 6.9021, 7.0494, 7.3966, 7.1131, 6.7607, 6.8209, 7.0603,
        6.5684, 6.6142, 6.5033, 7.5121, 6.5584, 6.9853, 6.0797, 6.3477, 6.2931, 6.3409,
        6.3310, 6.2066, 6.4709, 6.2967, 6.2576, 6.0601, 6.0389, 5.8797, 5.8274, 6.2133,
        5.9120, 5.9496, 6.2791, 6.3341, 6.1376, 5.9041, 6.4738, 6.1944, 6.2593]

f8block = [10.9989, 10.0716, 8.9823, 8.6559, 8.5833, 8.4840, 7.6824, 7.7560, 8.2501, 9.8703,
           7.6559, 7.5848, 7.6125, 7.1396, 6.9365, 7.2834, 7.1791, 7.0731, 6.9838, 7.1719,
           6.6751, 6.6264, 6.5436, 7.6035, 6.5918, 7.0558, 6.1192, 6.4002, 6.3726, 6.4227,
           6.4045, 6.2613, 6.5159, 6.3552, 6.3156, 6.1183, 6.0888, 5.9276, 5.8734, 6.2712,
           5.9870, 6.0120, 6.3314, 6.3528, 6.2536, 5.9412, 6.5165, 6.2466, 6.3025]

mxfp8 = [10.9989, 10.0716, 8.9822, 8.6558, 8.5808, 8.4687, 7.6704, 7.7778, 8.2423, 9.8657,
         7.6331, 7.5545, 7.5963, 7.2619, 7.0405, 7.3591, 7.2059, 7.0427, 6.9302, 7.1447,
         6.6346, 6.6152, 6.5486, 7.7417, 6.5966, 7.0549, 6.1237, 6.4064, 6.3578, 6.3891,
         6.3847, 6.2458, 6.4961, 6.3407, 6.3064, 6.1264, 6.0977, 5.9135, 5.8660, 6.2428,
         5.9858, 6.0121, 6.3295, 6.3662, 6.2292, 5.9320, 6.5045, 6.2417, 6.3027]

delayed = [11.0625, 10.9746, 10.0314, 8.9400, 8.0834, 8.0773, 7.7666, 7.9866, 8.5133, 8.5122,
           7.7356, 7.9095, 7.1407, 6.8475, 6.6914, 7.2064, 6.9483, 6.7537, 6.6969, 7.0054,
           6.4817, 6.5101, 6.4291, 7.4694, 6.4607, 6.9342, 6.0110, 6.2874, 6.2439, 6.2878,
           6.2886, 6.1337, 6.4008, 6.2744, 6.2267, 6.0013, 5.9942, 5.8067, 5.7702, 6.1321,
           5.8788, 5.8875, 6.2054, 6.2348, 6.0594, 5.8460, 6.4410, 6.1368, 6.1800]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

# Main loss curve
ax1.plot(steps, bf16, 'k-', alpha=0.6, linewidth=1.2, label='BF16 (no FP8)')
ax1.plot(steps, f8block, 'b-', linewidth=1.5, label='Float8BlockScaling')
ax1.plot(steps, mxfp8, 'r--', linewidth=1.5, label='MXFP8BlockScaling')
ax1.plot(steps, delayed, 'g-.', linewidth=1.5, label='DelayedScaling')
ax1.set_xlabel('Step', fontsize=11)
ax1.set_ylabel('Train Loss', fontsize=11)
ax1.set_title('Training Loss Curves — 561M Model, BS=4, RTX 5090 (SM 12.0)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 49)

# Zoomed view (steps 20-49)
ax2.plot(steps[19:], bf16[19:], 'k-', alpha=0.6, linewidth=1.2, label='BF16')
ax2.plot(steps[19:], f8block[19:], 'b-', linewidth=1.5, label='Float8Block')
ax2.plot(steps[19:], mxfp8[19:], 'r--', linewidth=1.5, label='MXFP8')
ax2.plot(steps[19:], delayed[19:], 'g-.', linewidth=1.5, label='Delayed')
ax2.set_xlabel('Step', fontsize=11)
ax2.set_ylabel('Train Loss', fontsize=11)
ax2.set_title('Zoomed View (Steps 20–49)', fontsize=11)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(20, 49)
ax2.set_ylim(5.5, 7.0)

plt.tight_layout()
plt.savefig('/home/nanogpt/nanogpt-fp8/loss_curves.png', dpi=150, bbox_inches='tight')
print("Saved loss_curves.png")
