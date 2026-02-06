"""
Davis Manifold Visualizations — Static Plot 5: GPU Throughput Scaling
=====================================================================
Puzzles/sec vs batch size — shows GPU saturation curve.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

# Data from the batch benchmark runs
batch_sizes = [1, 100, 1000, 10000, 65536]
throughputs = [
    1000 / 20.39,      # 1 puzzle in 20.39ms = ~49 puzzles/sec
    55791,              # from benchmark
    175091,
    226553,
    232130,
]
per_puzzle_us = [
    20390,              # 20.39 ms = 20390 µs
    17.92,
    5.71,
    4.41,
    4.31,
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0a0a')
for ax in (ax1, ax2):
    ax.set_facecolor('#111111')

# Left: Throughput
ax1.plot(batch_sizes, throughputs, 'o-', color='#ff006e', linewidth=3, markersize=10,
        markeredgecolor='white', markeredgewidth=1.5, zorder=5)
ax1.fill_between(batch_sizes, throughputs, alpha=0.15, color='#ff006e')

# Saturation line
ax1.axhline(232130, color='#ffbe0b', linewidth=1.5, linestyle='--', alpha=0.5)
ax1.text(2, 240000, '232K peak throughput', fontsize=10, color='#ffbe0b', fontweight='bold')

# Labels on points
for x, y in zip(batch_sizes, throughputs):
    label = f'{y:,.0f}'
    ax1.text(x, y * 1.12, label, ha='center', fontsize=9, color='white', fontweight='bold',
            path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

ax1.set_xscale('log')
ax1.set_xlabel('Batch Size', fontsize=12, color='white', fontweight='bold')
ax1.set_ylabel('Puzzles / Second', fontsize=12, color='white', fontweight='bold')
ax1.set_title('GPU Throughput vs Batch Size', fontsize=14, fontweight='bold', color='white', pad=12)
ax1.tick_params(colors='white')
ax1.spines['bottom'].set_color('#333')
ax1.spines['left'].set_color('#333')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(alpha=0.1, color='white')
ax1.set_ylim(0, 280000)

# Annotate the SM count
ax1.text(50, 50000, '36 SMs saturated\nat ~10K batch',
        fontsize=10, color='#778da9', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1b263b', edgecolor='#778da9', alpha=0.8))

# Right: Per-puzzle latency
ax2.plot(batch_sizes, per_puzzle_us, 's-', color='#06d6a0', linewidth=3, markersize=10,
        markeredgecolor='white', markeredgewidth=1.5, zorder=5)
ax2.fill_between(batch_sizes, per_puzzle_us, alpha=0.15, color='#06d6a0')

# Asymptote
ax2.axhline(4.1, color='#ffbe0b', linewidth=1.5, linestyle='--', alpha=0.5)
ax2.text(2, 3.0, '4.1 µs asymptotic latency', fontsize=10, color='#ffbe0b', fontweight='bold')

for x, y in zip(batch_sizes, per_puzzle_us):
    if y > 100:
        label = f'{y/1000:.1f} ms'
    else:
        label = f'{y:.1f} µs'
    offset = y * 1.15 if y < 1000 else y * 1.08
    ax2.text(x, offset, label, ha='center', fontsize=9, color='white', fontweight='bold',
            path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Batch Size', fontsize=12, color='white', fontweight='bold')
ax2.set_ylabel('Time per Puzzle (µs)', fontsize=12, color='white', fontweight='bold')
ax2.set_title('Per-Puzzle Latency vs Batch Size', fontsize=14, fontweight='bold', color='white', pad=12)
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('#333')
ax2.spines['left'].set_color('#333')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(alpha=0.1, color='white')

fig.text(0.5, 0.01,
         'RTX 5070 Laptop GPU · 36 SMs · sm_120 · 15-clue extreme puzzles · Curvature-guided Phase 3 DFS',
         ha='center', fontsize=9, color='#778da9', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
path = os.path.join(OUT, "05_gpu_throughput_scaling.png")
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {path}")
