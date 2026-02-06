"""
Davis Manifold Visualizations — Static Plot 4: Three-Phase Pipeline (v2)
=========================================================================
Clean vertical-flow layout. Fixed overlapping arrows and messy routing.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0a0a0a')
ax.set_facecolor('#0a0a0a')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

pe_black = [patheffects.withStroke(linewidth=2, foreground='black')]
pe_bg = [patheffects.withStroke(linewidth=2, foreground='#0a0a0a')]

def draw_box(ax, x, y, w, h, color, label, sublabel='', time_ms=''):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(box)
    cy = y + h * 0.6
    ax.text(x + w/2, cy, label, ha='center', va='center',
           fontsize=15, fontweight='bold', color='white', path_effects=pe_black)
    if sublabel:
        ax.text(x + w/2, cy - 0.38, sublabel, ha='center', va='center',
               fontsize=10, color='white', alpha=0.8, style='italic')
    if time_ms:
        ax.text(x + w/2, y + 0.25, time_ms, ha='center', va='center',
               fontsize=11, fontweight='bold', color='#ffbe0b', path_effects=pe_black)

def arrow(ax, x1, y1, x2, y2, color='white', lw=2.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# ── Title ──
ax.text(7, 9.55, 'Davis Manifold GPU Solver — Three-Phase Pipeline',
       ha='center', fontsize=17, fontweight='bold', color='white', path_effects=pe_black)
ax.text(7, 9.15, 'NVIDIA RTX 5070 · 36 SMs · sm_120 Blackwell',
       ha='center', fontsize=10, color='#778da9')

# ── Boxes ──
# Row 1: INPUT → PHASE 1 → Γ
draw_box(ax, 0.5, 7.2, 2.4, 1.4, '#1b263b', 'INPUT', '65,536 puzzles')
draw_box(ax, 4.2, 7.2, 2.8, 1.4, '#3a86ff', 'PHASE 1', 'Wavefront CP', '0.17 ms')

# Trichotomy diamond
dx, dy = 9.0, 7.9
diamond = plt.Polygon([
    [dx, dy + 0.7],
    [dx + 0.9, dy],
    [dx, dy - 0.7],
    [dx - 0.9, dy],
], facecolor='#f77f00', edgecolor='white', linewidth=2, alpha=0.9)
ax.add_patch(diamond)
ax.text(dx, dy, 'Γ', ha='center', va='center',
       fontsize=22, fontweight='bold', color='white', path_effects=pe_black)

# Row 2: PHASE 2 (left) and PHASE 3 (right)
draw_box(ax, 1.5, 4.2, 3.2, 1.6, '#06d6a0', 'PHASE 2', 'Manifold Relaxation', '0.16 ms')
draw_box(ax, 9.3, 4.2, 3.2, 1.6, '#ff006e', 'PHASE 3', 'Jackknife DFS', '21.4 ms')

# Row 3: SOLVED (center)
draw_box(ax, 5.0, 1.3, 4.0, 1.6, '#c9a227', 'SOLVED', '65,536 / 65,536', '267 ms total')

# ── Arrows ──
# input → phase 1
arrow(ax, 2.9, 7.9, 4.2, 7.9)

# phase 1 → trichotomy
arrow(ax, 7.0, 7.9, 8.1, 7.9)

# trichotomy → phase 2 (down-left)
arrow(ax, 8.1, 7.5, 4.7, 5.8, '#06d6a0')
ax.text(5.8, 7.0, 'Γ > 0.35', fontsize=10, fontweight='bold', color='#06d6a0',
       path_effects=pe_bg, ha='center')

# trichotomy → phase 3 (down-right, short)
arrow(ax, 9.7, 7.2, 10.9, 5.8, '#ff006e')
ax.text(10.9, 6.8, 'Γ ≤ 0.35', fontsize=10, fontweight='bold', color='#ff006e',
       path_effects=pe_bg, ha='center')

# phase 2 → phase 3 (horizontal, feeds unsolved)
arrow(ax, 4.7, 5.0, 9.3, 5.0, '#adb5bd')
ax.text(7.0, 5.25, 'unsolved residual', fontsize=8, color='#adb5bd',
       ha='center', style='italic', path_effects=pe_bg)

# phase 2 → solved (diagonal down)
arrow(ax, 3.1, 4.2, 5.5, 2.9, '#06d6a0')
ax.text(3.4, 3.5, 'solved', fontsize=8, color='#06d6a0', ha='center',
       style='italic', path_effects=pe_bg)

# phase 3 → solved (diagonal down)
arrow(ax, 10.9, 4.2, 8.5, 2.9, '#ff006e')

# ── Trichotomy legend ──
lx, ly0 = 0.5, 3.2
ax.text(lx, ly0, 'Trichotomy Thresholds:', fontsize=10, fontweight='bold', color='white')
thresholds = [
    ('Γ > 1.0', 'Easy — CP alone', '#3a86ff'),
    ('0.35 < Γ ≤ 1.0', 'Medium — Relaxation', '#06d6a0'),
    ('Γ ≤ 0.35', 'Hard — Branching', '#ff006e'),
    ('Γ ≈ 0.19', 'Expert (15-clue)', '#d62828'),
]
for i, (g, desc, col) in enumerate(thresholds):
    y = ly0 - 0.45 * (i + 1)
    ax.plot([lx, lx + 0.35], [y, y], color=col, linewidth=4, solid_capstyle='round')
    ax.text(lx + 0.5, y, g, fontsize=9, fontweight='bold', color=col, va='center')
    ax.text(lx + 3.2, y, desc, fontsize=9, color='#adb5bd', va='center')

# ── Bottom stat line ──
ax.text(7, 0.45, '245K puzzles/sec  ·  4.08 µs/puzzle  ·  15-clue extreme  ·  100% solve rate',
       ha='center', fontsize=11, fontweight='bold', color='#ffbe0b', path_effects=pe_black)

path = os.path.join(OUT, "04_pipeline_architecture.png")
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {path}")
