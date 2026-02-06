"""
Davis Manifold Visualizations — Static Plot 3: Benchmark Bar Chart
===================================================================
Head-to-head comparison of all solvers on the 15-clue extreme puzzle.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

OUT = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUT, exist_ok=True)

# Benchmark results from the head-to-head run (Feb 6, 2026)
solvers_extreme = [
    ("Davis GPU",  20.39,   True),
    ("DLX",        77.77,   True),
    ("CP",         3066.32, True),
    ("DFS",        4194.67, True),
    ("Davis CPU",  25012.11, True),
    ("Annealing",  60000,   False),  # timeout
]

solvers_easy = [
    ("Davis GPU",  25.23,   True),
    ("DFS",        62.86,   True),
    ("DLX",        71.77,   True),
    ("CP",         279.77,  True),
    ("Annealing",  4303.15, True),
    ("Davis CPU",  9031.54, True),
]

# Color scheme
colors = {
    'Davis GPU':  '#ff006e',
    'DLX':        '#3a86ff',
    'CP':         '#f77f00',
    'DFS':        '#8338ec',
    'Davis CPU':  '#06d6a0',
    'Annealing':  '#adb5bd',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0a0a')
for ax in (ax1, ax2):
    ax.set_facecolor('#111111')

# ── EXTREME PUZZLE ──
names = [s[0] for s in solvers_extreme]
times = [s[1] for s in solvers_extreme]
solved = [s[2] for s in solvers_extreme]
bar_colors = [colors.get(n, '#adb5bd') for n in names]

bars1 = ax1.barh(range(len(names)), times, color=bar_colors, edgecolor='white', linewidth=0.5, height=0.6)
ax1.set_xscale('log')
ax1.set_yticks(range(len(names)))
ax1.set_yticklabels(names, fontsize=11, fontweight='bold', color='white')
ax1.invert_yaxis()
ax1.set_xlabel('Time (ms) — log scale', fontsize=11, color='white')
ax1.set_title('Extreme Puzzle (15 clues)\n$\\Gamma \\approx 0.19$', 
              fontsize=14, fontweight='bold', color='white', pad=15)

# Add time labels
for i, (t, s) in enumerate(zip(times, solved)):
    label = f'{t:.1f} ms' if s else 'TIMEOUT'
    x = t * 1.3
    ax1.text(x, i, label, va='center', fontsize=10, color='white', fontweight='bold',
            path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

# Speedup annotation for GPU
gpu_time = solvers_extreme[0][1]
slowest = max(t for _, t, s in solvers_extreme if s)
ax1.annotate(f'{slowest/gpu_time:.0f}× faster', xy=(gpu_time, 0), xytext=(gpu_time * 5, -0.6),
            fontsize=12, fontweight='bold', color='#ff006e',
            arrowprops=dict(arrowstyle='->', color='#ff006e', lw=2),
            path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

ax1.tick_params(colors='white')
ax1.spines['bottom'].set_color('#333')
ax1.spines['left'].set_color('#333')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.15, color='white')

# ── EASY PUZZLE ──
names2 = [s[0] for s in solvers_easy]
times2 = [s[1] for s in solvers_easy]
solved2 = [s[2] for s in solvers_easy]
bar_colors2 = [colors.get(n, '#adb5bd') for n in names2]

bars2 = ax2.barh(range(len(names2)), times2, color=bar_colors2, edgecolor='white', linewidth=0.5, height=0.6)
ax2.set_xscale('log')
ax2.set_yticks(range(len(names2)))
ax2.set_yticklabels(names2, fontsize=11, fontweight='bold', color='white')
ax2.invert_yaxis()
ax2.set_xlabel('Time (ms) — log scale', fontsize=11, color='white')
ax2.set_title('Easy Puzzle (36 clues)\n$\\Gamma > 1.0$',
              fontsize=14, fontweight='bold', color='white', pad=15)

for i, (t, s) in enumerate(zip(times2, solved2)):
    label = f'{t:.1f} ms'
    x = t * 1.3
    ax2.text(x, i, label, va='center', fontsize=10, color='white', fontweight='bold',
            path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('#333')
ax2.spines['left'].set_color('#333')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.15, color='white')

fig.text(0.5, 0.01, 
         'Davis Manifold Sudoku Solver — Head-to-Head Benchmark (RTX 5070 Laptop GPU, 36 SMs)',
         ha='center', fontsize=9, color='#778da9', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
path = os.path.join(OUT, "03_benchmark_comparison.png")
fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {path}")
