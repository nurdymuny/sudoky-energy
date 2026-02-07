"""
Head-to-Head Benchmark: All Solvers vs Davis GPU
=================================================
Compares every solver on the same puzzle(s).
"""
import time
import subprocess
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sudoku.core.board import SudokuBoard
from sudoku.solvers.dfs_solver import DFSSolver
from sudoku.solvers.dlx_solver import DLXSolver
from sudoku.solvers.cp_solver import CPSolver
from sudoku.solvers.annealing_solver import AnnealingSolver
from sudoku.solvers.davis_solver import DavisManifoldSolver

# The same 15-clue extreme puzzle the GPU demo uses
EXTREME_PUZZLE = [
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

# An easy puzzle for contrast
EASY_PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


def puzzle_to_board(puzzle):
    """Convert 2D list to SudokuBoard."""
    board = SudokuBoard()
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0:
                board.set(r, c, puzzle[r][c])
    return board


def run_cpu_solver(solver_class, puzzle, timeout=60):
    """Run a CPU solver, return (solved, time_ms, stats)."""
    board = puzzle_to_board(puzzle)
    solver = solver_class()
    
    t0 = time.perf_counter()
    try:
        solution, stats = solver.solve(board)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        solved = stats.solved
        return solved, elapsed_ms, stats
    except Exception as e:
        t1 = time.perf_counter()
        return False, (t1 - t0) * 1000, None


def run_gpu_solver(n_puzzles=1):
    """Run the GPU solver binary, parse output."""
    gpu_exe = os.path.join(
        os.path.dirname(__file__),
        "sudoku", "solvers", "davis_gpu_solver", "davis_solver.exe"
    )
    if not os.path.exists(gpu_exe):
        return None, 0, "Binary not found"
    
    args = [gpu_exe]
    if n_puzzles > 1:
        args.append(str(n_puzzles))
    
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        output = result.stdout
        
        # Parse single puzzle time
        for line in output.split('\n'):
            if 'Wall time:' in line and 'Batch' not in output.split(line)[0][-200:]:
                time_ms = float(line.strip().split()[-2])
            if 'Batch Benchmark' in line:
                break
        
        # Parse batch results if present
        batch_throughput = None
        batch_time = None
        batch_per = None
        in_batch = False
        for line in output.split('\n'):
            if 'Batch Benchmark' in line:
                in_batch = True
            if in_batch and 'Wall time:' in line:
                batch_time = float(line.strip().split()[-2])
            if in_batch and 'Throughput:' in line:
                batch_throughput = float(line.strip().split()[-2])
            if in_batch and 'Per puzzle:' in line:
                # Handle the µs symbol encoding
                parts = line.strip().split()
                batch_per = float(parts[-2])
        
        if n_puzzles > 1 and batch_time:
            return True, batch_time, {
                'throughput': batch_throughput,
                'per_puzzle_us': batch_per,
                'batch_size': n_puzzles
            }
        else:
            return True, time_ms, None
            
    except Exception as e:
        return None, 0, str(e)


def print_separator(char='═', width=72):
    print(char * width)


def main():
    print()
    print_separator()
    print("  HEAD-TO-HEAD BENCHMARK: All Solvers")
    print("  Davis Manifold Sudoku Solver — Bee Rosa Davis (2025)")
    print_separator()
    print()

    # ─── ROUND 1: Single extreme puzzle ───
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ROUND 1: Single Extreme Puzzle (15 clues)                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    solvers = [
        ("DFS",        DFSSolver,           30),
        ("DLX",        DLXSolver,           30),
        ("CP",         CPSolver,            30),
        ("Davis CPU",  DavisManifoldSolver, 30),
    ]

    # Skip Annealing on extreme puzzle — it needs 60s+ and rarely solves 15-clue

    results = []

    for name, cls, timeout in solvers:
        print(f"  Running {name}...", end="", flush=True)
        solved, ms, stats = run_cpu_solver(cls, EXTREME_PUZZLE, timeout=timeout)
        status = "✓" if solved else "✗ (timeout/fail)"
        print(f" {ms:>10.2f} ms  {status}")
        results.append((name, solved, ms))

    # GPU single puzzle
    print(f"  Running Davis GPU...", end="", flush=True)
    solved, ms, _ = run_gpu_solver(1)
    status = "✓" if solved else "✗"
    print(f" {ms:>10.2f} ms  {status}")
    results.append(("Davis GPU", solved, ms))

    # Sort by time
    results.sort(key=lambda x: (not x[1], x[2]))

    print()
    print(f"  {'Rank':<6} {'Solver':<14} {'Time (ms)':>12} {'Speedup':>10} {'Status':>8}")
    print(f"  {'─'*6} {'─'*14} {'─'*12} {'─'*10} {'─'*8}")

    baseline = results[-1][2] if results else 1  # slowest as baseline
    for i, (name, solved, ms) in enumerate(results):
        rank = f"#{i+1}"
        speedup = f"{baseline / ms:.1f}×" if ms > 0 and solved else "—"
        status = "SOLVED" if solved else "FAILED"
        marker = " ◄" if "GPU" in name else ""
        print(f"  {rank:<6} {name:<14} {ms:>12.2f} {speedup:>10} {status:>8}{marker}")

    # ─── ROUND 2: Easy puzzle ───
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ROUND 2: Single Easy Puzzle (36 clues)                            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    easy_results = []
    easy_solvers = solvers + [("Annealing", AnnealingSolver, 60)]
    for name, cls, timeout in easy_solvers:
        print(f"  Running {name}...", end="", flush=True)
        solved, ms, stats = run_cpu_solver(cls, EASY_PUZZLE, timeout=timeout)
        status = "✓" if solved else "✗"
        print(f" {ms:>10.2f} ms  {status}")
        easy_results.append((name, solved, ms))

    print(f"  Running Davis GPU...", end="", flush=True)
    solved, ms, _ = run_gpu_solver(1)
    status = "✓" if solved else "✗"
    print(f" {ms:>10.2f} ms  {status}")
    easy_results.append(("Davis GPU", solved, ms))

    easy_results.sort(key=lambda x: (not x[1], x[2]))
    baseline = easy_results[-1][2] if easy_results else 1

    print()
    print(f"  {'Rank':<6} {'Solver':<14} {'Time (ms)':>12} {'Speedup':>10} {'Status':>8}")
    print(f"  {'─'*6} {'─'*14} {'─'*12} {'─'*10} {'─'*8}")
    for i, (name, solved, ms) in enumerate(easy_results):
        rank = f"#{i+1}"
        speedup = f"{baseline / ms:.1f}×" if ms > 0 and solved else "—"
        status = "SOLVED" if solved else "FAILED"
        marker = " ◄" if "GPU" in name else ""
        print(f"  {rank:<6} {name:<14} {ms:>12.2f} {speedup:>10} {status:>8}{marker}")

    # ─── ROUND 3: GPU batch throughput ───
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ROUND 3: GPU Batch Throughput (extreme puzzle × N)                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    for batch in [100, 1000, 10000, 65536]:
        print(f"  Batch {batch:>6}...", end="", flush=True)
        solved, ms, info = run_gpu_solver(batch)
        if info and isinstance(info, dict):
            print(f"  {ms:>9.1f} ms   {info['throughput']:>10,.0f} puzzles/sec   "
                  f"{info['per_puzzle_us']:.2f} µs/puzzle")
        else:
            print(f"  {ms:.1f} ms")

    print()
    print_separator()
    print("  Benchmark complete.")
    print_separator()
    print()


if __name__ == "__main__":
    main()
