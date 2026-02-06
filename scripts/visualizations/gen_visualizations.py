"""
Davis Manifold Visualizations — Static Plot 1: Curvature Heatmap
================================================================
Shows K_loc(c) for each cell of the extreme puzzle.
High curvature = high constraint tension = picked first by the solver.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap
from sudoku.solvers.davis_solver import (
    get_candidates, local_curvature, information_value,
    select_next_cell_davis, davis_energy, trichotomy_parameter,
    DavisSolver, holonomy_prune
)

# Output directory
OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

# The 15-clue extreme puzzle
PUZZLE = [
    [0, 0, 0, 0, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 3, 5, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 7, 0],
    [7, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 8],
    [0, 4, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0],
    [6, 5, 0, 0, 0, 0, 0, 0, 0],
]

def draw_grid(ax, board, data, cmap, title, label, vmin=None, vmax=None):
    """Draw a 9x9 Sudoku grid with colored cells and values."""
    data = np.array(data)
    if vmin is None: vmin = np.nanmin(data[~np.isnan(data)]) if np.any(~np.isnan(data)) else 0
    if vmax is None: vmax = np.nanmax(data[~np.isnan(data)]) if np.any(~np.isnan(data)) else 1
    
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val != 0:
                # Given clue — bold white
                ax.text(c, r, str(val), ha='center', va='center',
                       fontsize=16, fontweight='bold', color='white',
                       path_effects=[
                           patheffects.withStroke(linewidth=3, foreground='black')
                       ])
            else:
                # Empty cell — show data value
                v = data[r][c]
                if not np.isnan(v):
                    ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                           fontsize=8, color='white', alpha=0.9,
                           path_effects=[
                               patheffects.withStroke(linewidth=2, foreground='black')
                           ])
    
    # Draw 3x3 box borders
    for i in range(4):
        lw = 3 if i % 3 == 0 else 1
        ax.axhline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.8)
        ax.axvline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.8)
    
    # Thin grid lines
    for i in range(10):
        ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(label, fontsize=10)


# ── PLOT 1: Curvature Heatmap ──
print("Generating curvature heatmap...")
K_grid = np.full((9, 9), np.nan)
for r in range(9):
    for c in range(9):
        if PUZZLE[r][c] == 0:
            K_grid[r][c] = local_curvature(PUZZLE, r, c)
        else:
            K_grid[r][c] = 0.0  # solved cells have 0 curvature

davis_cmap = LinearSegmentedColormap.from_list('davis', [
    '#0d1b2a', '#1b263b', '#415a77', '#778da9',
    '#e07a5f', '#f2545b', '#ff006e', '#ffbe0b'
])

fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#0a0a0a')
ax.set_facecolor('#0a0a0a')
fig.patch.set_facecolor('#0a0a0a')

draw_grid(ax, PUZZLE, K_grid, davis_cmap,
          'Local Curvature  $K_{loc}(c)$\n15-Clue Extreme Puzzle',
          '$K_{loc}$')

fig.text(0.5, 0.02,
         'Davis Field Equations — High curvature = high constraint tension = solved first',
         ha='center', fontsize=9, color='#778da9', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
path1 = os.path.join(OUT, "01_curvature_heatmap.png")
fig.savefig(path1, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {path1}")
