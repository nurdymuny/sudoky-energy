"""
Davis Manifold Visualizations — Static Plot 2: Information Value V(c)
=====================================================================
Shows V(c) = K_loc(c) * log|R_c| — the Davis cell selection criterion.
Higher V = picked first by the solver. Numbers show solve order.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap
from sudoku.solvers.davis_solver import (
    get_candidates, local_curvature, information_value,
    select_next_cell_davis, DavisSolver
)

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

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

# Compute V(c) for every empty cell
V_grid = np.full((9, 9), np.nan)
for r in range(9):
    for c in range(9):
        if PUZZLE[r][c] == 0:
            V_grid[r][c] = information_value(PUZZLE, r, c)
        else:
            V_grid[r][c] = 0.0

# Simulate solve order by repeatedly picking highest V(c) cell
import copy
board_copy = copy.deepcopy(PUZZLE)
solver = DavisSolver()
order_grid = np.full((9, 9), np.nan)

# Record the order cells get selected (first 20 picks for clarity)
temp_board = copy.deepcopy(PUZZLE)
pick_order = []
for step in range(66):  # 81 - 15 clues = 66 empty cells
    cell = select_next_cell_davis(temp_board)
    if cell is None:
        break
    r, c = cell
    cands = get_candidates(temp_board, r, c)
    if not cands:
        break
    val = min(cands)  # just pick one for the ordering demo
    temp_board[r][c] = val
    pick_order.append((r, c, step + 1))
    order_grid[r][c] = step + 1

# Color map: bright for early picks (high V), dim for late picks
v_cmap = LinearSegmentedColormap.from_list('davis_v', [
    '#ffbe0b', '#f77f00', '#d62828', '#6a0572', '#1b0a2e', '#0a0a0a'
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0a0a0a')
for ax in (ax1, ax2):
    ax.set_facecolor('#0a0a0a')

# Left: V(c) values
V_display = V_grid.copy()
im1 = ax1.imshow(V_display, cmap='inferno', aspect='equal')
for r in range(9):
    for c in range(9):
        if PUZZLE[r][c] != 0:
            ax1.text(c, r, str(PUZZLE[r][c]), ha='center', va='center',
                    fontsize=16, fontweight='bold', color='white',
                    path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
        elif not np.isnan(V_grid[r][c]) and V_grid[r][c] > 0:
            ax1.text(c, r, f'{V_grid[r][c]:.1f}', ha='center', va='center',
                    fontsize=8, color='white', alpha=0.8,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

for i in range(4):
    lw = 3 if i % 3 == 0 else 1
    ax1.axhline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.7)
    ax1.axvline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.7)
for i in range(10):
    ax1.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.2)
    ax1.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.2)

ax1.set_xlim(-0.5, 8.5); ax1.set_ylim(8.5, -0.5)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_title('Information Value  $V(c) = K_{loc} \\cdot \\log|R_c|$', 
              fontsize=13, fontweight='bold', color='white', pad=12)
cb1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
cb1.set_label('$V(c)$', fontsize=10, color='white')
cb1.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cb1.ax.axes, 'yticklabels'), color='white')

# Right: Solve order
order_display = order_grid.copy()
# Invert so early picks are bright
max_order = np.nanmax(order_display)
order_inv = max_order - order_display + 1
order_inv = np.where(np.isnan(order_display), np.nan, order_inv)

im2 = ax2.imshow(order_inv, cmap=v_cmap, aspect='equal')
for r in range(9):
    for c in range(9):
        if PUZZLE[r][c] != 0:
            ax2.text(c, r, str(PUZZLE[r][c]), ha='center', va='center',
                    fontsize=14, fontweight='bold', color='#555555',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='#222')])
        elif not np.isnan(order_grid[r][c]):
            order_num = int(order_grid[r][c])
            # First 10 picks get big numbers
            if order_num <= 10:
                ax2.text(c, r, f'#{order_num}', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white',
                        path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
            elif order_num <= 30:
                ax2.text(c, r, f'{order_num}', ha='center', va='center',
                        fontsize=8, color='white', alpha=0.7,
                        path_effects=[patheffects.withStroke(linewidth=1, foreground='black')])

for i in range(4):
    lw = 3 if i % 3 == 0 else 1
    ax2.axhline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.7)
    ax2.axvline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.7)
for i in range(10):
    ax2.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.2)
    ax2.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.2)

ax2.set_xlim(-0.5, 8.5); ax2.set_ylim(8.5, -0.5)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_title('Davis Cell Selection Order\n(bright = picked first)',
              fontsize=13, fontweight='bold', color='white', pad=12)
cb2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
cb2.set_label('Priority (bright = early)', fontsize=10, color='white')
cb2.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='white')

fig.text(0.5, 0.01,
         'Davis Field Equations — Cells with highest V(c) are solved first, guided by manifold geometry',
         ha='center', fontsize=9, color='#778da9', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
path = os.path.join(OUT, "02_information_value.png")
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {path}")
