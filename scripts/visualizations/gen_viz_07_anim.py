"""
Davis Manifold Visualizations — Animated GIF 2: Energy Descent
==============================================================
Shows E[γ] decreasing step-by-step as the solver fills cells.
The energy functional collapses toward 0 as the puzzle converges.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from PIL import Image
from sudoku.solvers.davis_solver import (
    get_candidates, local_curvature, information_value,
    select_next_cell_davis, DavisSolver
)

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)
FRAMES_DIR = os.path.join(OUT, "_frames7")
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

def energy_functional(board):
    """
    E[γ] = Σ_c  K(c) · H(c)
    where H(c) = log2(|candidates|) is the local entropy
    and K(c) is the curvature. For filled cells, E = 0.
    """
    E = 0.0
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                K = local_curvature(board, r, c)
                cands = get_candidates(board, r, c)
                n = len(cands)
                H = math.log2(n) if n > 1 else 0
                E += K * H
    return E

def total_curvature(board):
    """Σ K(c) over empty cells."""
    return sum(local_curvature(board, r, c)
               for r in range(9) for c in range(9)
               if board[r][c] == 0)

def total_entropy(board):
    """Σ H(c) over empty cells."""
    S = 0.0
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                cands = get_candidates(board, r, c)
                n = len(cands)
                if n > 1:
                    S += math.log2(n)
    return S

# Solve to get solution
solver = DavisSolver()
solution = copy.deepcopy(PUZZLE)
solver.solve(solution)

# Simulate step-by-step
board = copy.deepcopy(PUZZLE)
energy_trace = []
curvature_trace = []
entropy_trace = []
step_labels = []

E0 = energy_functional(board)
K0 = total_curvature(board)
S0 = total_entropy(board)
energy_trace.append(E0)
curvature_trace.append(K0)
entropy_trace.append(S0)
step_labels.append(0)

for step in range(66):
    cell = select_next_cell_davis(board)
    if cell is None:
        break
    r, c = cell
    board[r][c] = solution[r][c]
    E = energy_functional(board)
    K = total_curvature(board)
    S = total_entropy(board)
    energy_trace.append(E)
    curvature_trace.append(K)
    entropy_trace.append(S)
    step_labels.append(step + 1)

print(f"Traced {len(energy_trace)} steps, E: {energy_trace[0]:.2f} → {energy_trace[-1]:.2f}")

# Generate animated frames
frame_paths = []
n_steps = len(energy_trace)
x_all = np.array(step_labels)
e_all = np.array(energy_trace)
k_all = np.array(curvature_trace)
s_all = np.array(entropy_trace)

for frame_i in range(n_steps):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), facecolor='#0a0a0a',
                              gridspec_kw={'hspace': 0.35})
    
    x = x_all[:frame_i + 1]
    e = e_all[:frame_i + 1]
    k = k_all[:frame_i + 1]
    s = s_all[:frame_i + 1]
    
    # --- Panel 1: Energy E[γ] ---
    ax = axes[0]
    ax.set_facecolor('#0a0a0a')
    ax.fill_between(x, e, alpha=0.3, color='#ff006e')
    ax.plot(x, e, color='#ff006e', linewidth=2.5)
    ax.scatter([x[-1]], [e[-1]], color='#ff006e', s=80, zorder=5,
              edgecolors='white', linewidth=1.5)
    ax.set_xlim(0, 66)
    ax.set_ylim(-0.5, e_all[0] * 1.1)
    ax.set_ylabel(r'$E[\gamma]$', fontsize=12, color='white', fontweight='bold')
    ax.set_title('Energy Functional Descent', fontsize=13, fontweight='bold',
                color='white', pad=6)
    ax.tick_params(colors='#adb5bd', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.grid(True, alpha=0.15, color='white')
    ax.text(0.98, 0.85, f'E = {e[-1]:.1f}', transform=ax.transAxes,
           ha='right', fontsize=14, fontweight='bold', color='#ff006e',
           path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    # --- Panel 2: Curvature Σ K ---
    ax = axes[1]
    ax.set_facecolor('#0a0a0a')
    ax.fill_between(x, k, alpha=0.3, color='#06d6a0')
    ax.plot(x, k, color='#06d6a0', linewidth=2.5)
    ax.scatter([x[-1]], [k[-1]], color='#06d6a0', s=80, zorder=5,
              edgecolors='white', linewidth=1.5)
    ax.set_xlim(0, 66)
    ax.set_ylim(-0.5, k_all[0] * 1.1)
    ax.set_ylabel(r'$\Sigma\, K(c)$', fontsize=12, color='white', fontweight='bold')
    ax.set_title('Total Curvature Collapse', fontsize=13, fontweight='bold',
                color='white', pad=6)
    ax.tick_params(colors='#adb5bd', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.grid(True, alpha=0.15, color='white')
    ax.text(0.98, 0.85, f'ΣK = {k[-1]:.1f}', transform=ax.transAxes,
           ha='right', fontsize=14, fontweight='bold', color='#06d6a0',
           path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    # --- Panel 3: Entropy Σ H ---
    ax = axes[2]
    ax.set_facecolor('#0a0a0a')
    ax.fill_between(x, s, alpha=0.3, color='#ffbe0b')
    ax.plot(x, s, color='#ffbe0b', linewidth=2.5)
    ax.scatter([x[-1]], [s[-1]], color='#ffbe0b', s=80, zorder=5,
              edgecolors='white', linewidth=1.5)
    ax.set_xlim(0, 66)
    ax.set_ylim(-0.5, s_all[0] * 1.1)
    ax.set_xlabel('Solve Step', fontsize=11, color='white')
    ax.set_ylabel(r'$\Sigma\, H(c)$', fontsize=12, color='white', fontweight='bold')
    ax.set_title('Entropy Drainage', fontsize=13, fontweight='bold',
                color='white', pad=6)
    ax.tick_params(colors='#adb5bd', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.grid(True, alpha=0.15, color='white')
    ax.text(0.98, 0.85, f'ΣH = {s[-1]:.1f}', transform=ax.transAxes,
           ha='right', fontsize=14, fontweight='bold', color='#ffbe0b',
           path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    fig.text(0.5, 0.005, 'Davis Field Equations — geometric energy minimization on the constraint manifold',
            ha='center', fontsize=8, color='#778da9', style='italic')
    
    fpath = os.path.join(FRAMES_DIR, f"frame_{frame_i:03d}.png")
    fig.savefig(fpath, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    frame_paths.append(fpath)
    
    if frame_i % 10 == 0:
        print(f"  Frame {frame_i}/{n_steps}")

print(f"Generated {len(frame_paths)} frames, assembling GIF...")

# Assemble GIF
frames = [Image.open(fp) for fp in frame_paths]
durations = [1000] + [120] * (len(frames) - 2) + [2500]

gif_path = os.path.join(OUT, "07_energy_descent.gif")
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
