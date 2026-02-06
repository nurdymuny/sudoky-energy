"""
Davis Manifold Visualizations — Static Plot 4: Three-Phase Pipeline
====================================================================
Shows the GPU solver architecture: Phase 1 → 2 → 3 with Γ routing.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9), facecolor='#0a0a0a')
ax.set_facecolor('#0a0a0a')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')

def draw_box(ax, x, y, w, h, color, label, sublabel='', time_ms=''):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.65, label, ha='center', va='center',
           fontsize=14, fontweight='bold', color='white',
           path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    if sublabel:
        ax.text(x + w/2, y + h*0.35, sublabel, ha='center', va='center',
               fontsize=9, color='white', alpha=0.8, style='italic')
    if time_ms:
        ax.text(x + w/2, y + h*0.12, time_ms, ha='center', va='center',
               fontsize=10, fontweight='bold', color='#ffbe0b',
               path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

def draw_arrow(ax, x1, y1, x2, y2, color='white', label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=2.5, 
                              connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + 0.2
        ax.text(mx, my, label, ha='center', va='center', fontsize=9,
               color=color, fontweight='bold',
               path_effects=[patheffects.withStroke(linewidth=2, foreground='#0a0a0a')])

# Title
ax.text(8, 8.5, 'Davis Manifold GPU Solver — Three-Phase Pipeline',
       ha='center', fontsize=18, fontweight='bold', color='white',
       path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
ax.text(8, 8.0, 'NVIDIA RTX 5070 · 36 SMs · sm_120 Blackwell',
       ha='center', fontsize=11, color='#778da9')

# Input
draw_box(ax, 0.3, 5.0, 2.5, 1.8, '#1b263b', 'INPUT', '65,536 puzzles', '')

# Phase 1
draw_box(ax, 4.0, 5.0, 3.0, 1.8, '#3a86ff', 'PHASE 1', 'Wavefront CP', '0.17 ms')

# Trichotomy diamond
diamond_x, diamond_y = 8.5, 5.9
diamond = plt.Polygon([
    [diamond_x, diamond_y + 0.8],
    [diamond_x + 1.0, diamond_y],
    [diamond_x, diamond_y - 0.8],
    [diamond_x - 1.0, diamond_y],
], facecolor='#f77f00', edgecolor='white', linewidth=2, alpha=0.9)
ax.add_patch(diamond)
ax.text(diamond_x, diamond_y, 'Γ', ha='center', va='center',
       fontsize=20, fontweight='bold', color='white',
       path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])

# Phase 2
draw_box(ax, 10.5, 6.5, 3.0, 1.5, '#06d6a0', 'PHASE 2', 'Manifold Relaxation', '0.16 ms')

# Phase 3
draw_box(ax, 10.5, 3.5, 3.0, 1.5, '#ff006e', 'PHASE 3', 'Jackknife DFS', '21.4 ms')

# Output
draw_box(ax, 10.5, 1.0, 3.0, 1.5, '#ffbe0b', 'SOLVED', '65,536 / 65,536', '267 ms total')

# Arrows
draw_arrow(ax, 2.8, 5.9, 4.0, 5.9)                    # input → phase 1
draw_arrow(ax, 7.0, 5.9, 7.5, 5.9)                     # phase 1 → trichotomy
draw_arrow(ax, 9.5, 6.3, 10.5, 7.0, '#06d6a0', 'Γ > 0.35')   # → phase 2
draw_arrow(ax, 9.5, 5.5, 10.5, 4.5, '#ff006e', 'Γ < 0.35')   # → phase 3
draw_arrow(ax, 13.5, 6.5, 14.0, 5.5, '#06d6a0')        # phase 2 down
draw_arrow(ax, 14.0, 5.5, 14.0, 5.0, '#06d6a0')        # continuing down
draw_arrow(ax, 14.0, 4.0, 13.5, 3.5, '#ff006e')        # to phase 3 if needed
draw_arrow(ax, 12.0, 3.5, 12.0, 2.5, '#ffbe0b')        # phase 3 → solved

# Γ threshold labels
ax.text(1.5, 3.5, 'Trichotomy Thresholds:', fontsize=11, fontweight='bold', color='white')
thresholds = [
    ('Γ > 1.0', 'Easy — CP solves it', '#3a86ff'),
    ('0.35 < Γ < 1.0', 'Medium — Relaxation helps', '#06d6a0'),
    ('Γ < 0.35', 'Hard — Need branching', '#ff006e'),
    ('Γ < 0.20', 'Expert — Deep DFS', '#d62828'),
]
for i, (g, desc, col) in enumerate(thresholds):
    y = 2.8 - i * 0.5
    ax.plot([1.5, 1.8], [y, y], color=col, linewidth=4, solid_capstyle='round')
    ax.text(2.0, y, f'{g}', fontsize=10, fontweight='bold', color=col, va='center')
    ax.text(5.0, y, desc, fontsize=9, color='#adb5bd', va='center')

# GPU info box
ax.text(8, 0.5, '232K puzzles/sec  ·  4.1 µs/puzzle  ·  15-clue extreme',
       ha='center', fontsize=12, fontweight='bold', color='#ffbe0b',
       path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

path = os.path.join(OUT, "04_pipeline_architecture.png")
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {path}")
