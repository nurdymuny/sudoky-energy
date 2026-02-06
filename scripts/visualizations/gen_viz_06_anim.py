"""
Davis Manifold Visualizations — Animated GIF 1: Solve Process
==============================================================
Step-by-step animation showing cells being filled in Davis order.
Each frame: cell with highest V(c) gets solved, curvature recomputed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sudoku.solvers.davis_solver import (
    get_candidates, local_curvature, information_value,
    select_next_cell_davis, DavisSolver
)

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)
FRAMES_DIR = os.path.join(OUT, "_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

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

# First solve it to get the actual solution
solver = DavisSolver()
solution = copy.deepcopy(PUZZLE)
solver.solve(solution)

# Now simulate the solve step by step using Davis ordering
# At each step, pick the cell Davis would pick, fill with the correct solution value
board = copy.deepcopy(PUZZLE)
clue_set = {(r, c) for r in range(9) for c in range(9) if PUZZLE[r][c] != 0}
steps = []  # (board_snapshot, highlighted_cell, step_num, V_value)

# Record initial state
steps.append((copy.deepcopy(board), None, 0, 0))

for step in range(66):  # 66 empty cells
    cell = select_next_cell_davis(board)
    if cell is None:
        break
    r, c = cell
    V = information_value(board, r, c)
    # Use the actual solution value
    board[r][c] = solution[r][c]
    steps.append((copy.deepcopy(board), (r, c), step + 1, V))

print(f"Recorded {len(steps)} frames")

# Generate frames
cmap = LinearSegmentedColormap.from_list('davis', [
    '#0d1b2a', '#1b263b', '#415a77', '#778da9',
    '#e07a5f', '#f2545b', '#ff006e', '#ffbe0b'
])

frame_paths = []
for idx, (brd, highlight, step_num, V_val) in enumerate(steps):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Compute curvature for current state
    K_grid = np.zeros((9, 9))
    for r in range(9):
        for c in range(9):
            if brd[r][c] == 0:
                K_grid[r][c] = local_curvature(brd, r, c)
    
    ax.imshow(K_grid, cmap=cmap, vmin=0, vmax=1.0, aspect='equal')
    
    # Draw values
    for r in range(9):
        for c in range(9):
            val = brd[r][c]
            if val != 0:
                if (r, c) in clue_set:
                    color = '#adb5bd'
                    fs = 15
                    fw = 'bold'
                elif highlight and (r, c) == highlight:
                    color = '#ff006e'
                    fs = 18
                    fw = 'bold'
                else:
                    color = '#06d6a0'
                    fs = 14
                    fw = 'bold'
                ax.text(c, r, str(val), ha='center', va='center',
                       fontsize=fs, fontweight=fw, color=color,
                       path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    # Highlight the just-filled cell
    if highlight:
        hr, hc = highlight
        rect = plt.Rectangle((hc - 0.5, hr - 0.5), 1, 1,
                             fill=False, edgecolor='#ff006e', linewidth=3)
        ax.add_patch(rect)
    
    # Grid lines
    for i in range(4):
        lw = 3 if i % 3 == 0 else 1
        ax.axhline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.6)
        ax.axvline(i * 3 - 0.5, color='white', linewidth=lw, alpha=0.6)
    for i in range(10):
        ax.axhline(i - 0.5, color='white', linewidth=0.3, alpha=0.2)
        ax.axvline(i - 0.5, color='white', linewidth=0.3, alpha=0.2)
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Title with step info
    empty = sum(1 for r in range(9) for c in range(9) if brd[r][c] == 0)
    if step_num == 0:
        title = f'Step 0: Initial Puzzle  ({empty} empty cells)'
    else:
        title = f'Step {step_num}: V(c) = {V_val:.2f}  ({empty} remaining)'
    ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=8)
    
    # Progress bar
    progress = (66 - empty) / 66
    ax.barh(-1.2, progress * 9, height=0.3, left=-0.5,
           color='#ff006e', alpha=0.8)
    ax.barh(-1.2, 9, height=0.3, left=-0.5,
           fill=False, edgecolor='#333', linewidth=1)
    ax.text(4, -1.2, f'{progress*100:.0f}%', ha='center', va='center',
           fontsize=8, color='white', fontweight='bold')
    ax.set_ylim(9.0, -1.8)
    
    fig.text(0.5, 0.01, 'Davis Field Equations — Curvature-guided cell selection',
            ha='center', fontsize=8, color='#778da9', style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fpath = os.path.join(FRAMES_DIR, f"frame_{idx:03d}.png")
    fig.savefig(fpath, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    frame_paths.append(fpath)
    
    if idx % 10 == 0:
        print(f"  Frame {idx}/{len(steps)}")

print(f"Generated {len(frame_paths)} frames, assembling GIF...")

# Assemble GIF
frames = [Image.open(fp) for fp in frame_paths]
# Hold first and last frames longer
durations = [800] + [150] * (len(frames) - 2) + [2000]

gif_path = os.path.join(OUT, "06_solve_animation.gif")
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=0,
    optimize=True,
)
print(f"Saved: {gif_path}")

# Cleanup frames
import shutil
shutil.rmtree(FRAMES_DIR)
print("Cleaned up frame files.")
