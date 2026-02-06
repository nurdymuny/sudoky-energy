"""
Davis Manifold Visualizations — Static composite for paper
============================================================
4-panel composite showing key frames from the solve animation:
  Step 0 (initial), Step 15, Step 40, Step 66 (complete).
Also: separate energy descent static plot for the paper.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import copy, math
import numpy as np
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

# Solve
solver = DavisSolver()
solution = copy.deepcopy(PUZZLE)
solver.solve(solution)

# Simulate step-by-step, record snapshots
board = copy.deepcopy(PUZZLE)
clue_set = {(r, c) for r in range(9) for c in range(9) if PUZZLE[r][c] != 0}
snapshots = {}  # step -> board
energy_trace, curv_trace, ent_trace = [], [], []

def energy_functional(b):
    E = 0.0
    for r in range(9):
        for c in range(9):
            if b[r][c] == 0:
                K = local_curvature(b, r, c)
                cands = get_candidates(b, r, c)
                n = len(cands)
                H = math.log2(n) if n > 1 else 0
                E += K * H
    return E

def total_curvature(b):
    return sum(local_curvature(b, r, c) for r in range(9) for c in range(9) if b[r][c] == 0)

def total_entropy(b):
    S = 0.0
    for r in range(9):
        for c in range(9):
            if b[r][c] == 0:
                cands = get_candidates(b, r, c)
                n = len(cands)
                if n > 1: S += math.log2(n)
    return S

snapshots[0] = copy.deepcopy(board)
energy_trace.append(energy_functional(board))
curv_trace.append(total_curvature(board))
ent_trace.append(total_entropy(board))

for step in range(66):
    cell = select_next_cell_davis(board)
    if cell is None: break
    r, c = cell
    board[r][c] = solution[r][c]
    snapshots[step + 1] = copy.deepcopy(board)
    energy_trace.append(energy_functional(board))
    curv_trace.append(total_curvature(board))
    ent_trace.append(total_entropy(board))

print(f"Recorded {len(snapshots)} snapshots")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: 4-panel solve composite
# ═══════════════════════════════════════════════════════════════════
cmap = LinearSegmentedColormap.from_list('davis', [
    '#0d1b2a', '#1b263b', '#415a77', '#778da9',
    '#e07a5f', '#f2545b', '#ff006e', '#ffbe0b'
])

key_steps = [0, 15, 40, 66]
titles = ['(a) Initial: 66 empty', '(b) Step 15: 51 empty',
          '(c) Step 40: 26 empty', '(d) Step 66: Solved']

fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor='#0a0a0a')
fig.suptitle('Curvature-Guided Solve Process — Davis Field Equations',
            fontsize=14, fontweight='bold', color='white', y=0.98)

for idx, (step, title) in enumerate(zip(key_steps, titles)):
    ax = axes[idx]
    ax.set_facecolor('#0a0a0a')
    brd = snapshots[step]

    K_grid = np.zeros((9, 9))
    for r in range(9):
        for c in range(9):
            if brd[r][c] == 0:
                K_grid[r][c] = local_curvature(brd, r, c)

    ax.imshow(K_grid, cmap=cmap, vmin=0, vmax=1.0, aspect='equal')

    for r in range(9):
        for c in range(9):
            val = brd[r][c]
            if val != 0:
                if (r, c) in clue_set:
                    color, fs = '#adb5bd', 10
                else:
                    color, fs = '#06d6a0', 9
                ax.text(c, r, str(val), ha='center', va='center',
                       fontsize=fs, fontweight='bold', color=color,
                       path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')])

    for i in range(4):
        lw = 2.5 if i % 3 == 0 else 0.5
        ax.axhline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.5)
        ax.axvline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.5)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=6)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
path = os.path.join(OUT, "08_solve_composite.png")
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Energy descent (static, for paper)
# ═══════════════════════════════════════════════════════════════════
x = np.arange(len(energy_trace))

fig, axes = plt.subplots(3, 1, figsize=(8, 7), facecolor='#0a0a0a',
                          gridspec_kw={'hspace': 0.35})

panels = [
    (energy_trace, r'$E[\gamma]$', 'Energy Functional', '#ff006e'),
    (curv_trace,   r'$\Sigma\, K(c)$', 'Total Curvature', '#06d6a0'),
    (ent_trace,    r'$\Sigma\, H(c)$', 'Total Entropy', '#ffbe0b'),
]
for ax, (trace, ylabel, title, color) in zip(axes, panels):
    ax.set_facecolor('#0a0a0a')
    y = np.array(trace)
    ax.fill_between(x, y, alpha=0.25, color=color)
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_xlim(0, 66)
    ax.set_ylim(-0.5, y[0] * 1.15)
    ax.set_ylabel(ylabel, fontsize=11, color='white', fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', color='white', pad=4)
    ax.tick_params(colors='#adb5bd', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.grid(True, alpha=0.12, color='white')

axes[2].set_xlabel('Solve Step', fontsize=10, color='white')
fig.suptitle('Energy Descent During Curvature-Guided Solve',
            fontsize=13, fontweight='bold', color='white', y=1.01)
fig.text(0.5, -0.01, f'E[γ]: {energy_trace[0]:.1f} → {energy_trace[-1]:.1f}  |  '
         f'ΣK: {curv_trace[0]:.1f} → {curv_trace[-1]:.1f}  |  '
         f'ΣH: {ent_trace[0]:.1f} → {ent_trace[-1]:.1f}',
        ha='center', fontsize=9, color='#778da9', style='italic')

path = os.path.join(OUT, "09_energy_descent_static.png")
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {path}")
