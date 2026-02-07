"""
Benchmark the world's hardest known Sudoku puzzles.
Tests Davis GPU (CUDA), Davis CPU, DLX, DFS, and CP solvers against famous constructions.
"""
import sys, time, os, re, subprocess, threading
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sudoku.core.board import SudokuBoard
from sudoku.solvers.dlx_solver import DLXSolver
from sudoku.solvers.dfs_solver import DFSSolver
from sudoku.solvers.cp_solver import CPSolver
from sudoku.solvers.davis_solver import DavisManifoldSolver

# Paths to the compiled GPU solvers
GPU_EXE = os.path.join(os.path.dirname(__file__), '..', 'sudoku', 'solvers',
                       'davis_gpu_solver', 'davis_solver.exe')
GPU_V2_EXE = os.path.join(os.path.dirname(__file__), '..', 'sudoku', 'solvers',
                          'davis_gpu_solver', 'davis_solver_v2.exe')
GPU_V3_EXE = os.path.join(os.path.dirname(__file__), '..', 'sudoku', 'solvers',
                          'davis_gpu_solver', 'davis_solver_v3.exe')

# World's hardest Sudoku puzzles (published sources)
HARDEST_PUZZLES = {
    "AI Escargot (Inkala 2006)":       "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
    "Inkala 2010":                     "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    "Golden Nugget":                   "000000039000001005003050800008090006070002000100400000009080050020000600400700000",
    "Easter Monster":                  "100000002090400050006000700050903000000070000000850040700000600030009080002000001",
    "Platinum Blonde":                 "000000012000000003002300400001800005060070800000009000008500000900040500470006000",
    "Tarek071":                        "000000000000003085001020000000507000004000100090000000500000073002010000000040009",
    "17-clue Coloin":                  "000000010400000000020000000000050407008000300001090000300400200050100000000806000",
    "Escargot variant":                "000200000060010700008000050500004000700000002009300000000700400040060009001000030",
    "Norvig hard1":                    "400000805030000000000700000020000060000080400000010000000603070500200000104000000",
    "Inkala (Norvig hardest)":         "850002400720000009004000000000107002305000900040000000000080070017000000000036040",
    "champagne 2010":                  "000000000000000000009010800000700360060300008001006000080020900000005020070000004",
}

# Solver ordering: GPU v1, GPU v2, Davis CPU, then baselines
SOLVER_NAMES = ["Davis GPU", "Davis GPU v2", "Davis GPU v3", "Davis CPU", "DLX", "DFS", "CP"]

# Per-solver timeout in seconds
SOLVER_TIMEOUT = {
    "Davis CPU": 30,
    "DLX":       120,
    "DFS":        60,
    "CP":         60,
}

def solve_with_timeout(solver, board, timeout_sec):
    """Run a solver in a thread with a timeout. Returns (solution, stats) or None on timeout."""
    result = [None, None]
    exc = [None]
    def target():
        try:
            sol, st = solver.solve(board)
            result[0] = sol
            result[1] = st
        except Exception as e:
            exc[0] = e
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        return "TIMEOUT"
    if exc[0]:
        raise exc[0]
    return result[0], result[1]

def string_to_grid(s):
    """Convert 81-char string to 9x9 numpy array."""
    return np.array([[int(s[r*9+c]) for c in range(9)] for r in range(9)])

def count_clues(s):
    return sum(1 for c in s if c != '0' and c != '.')

def verify_solution(grid):
    """Check if a 9x9 grid is a valid complete Sudoku solution."""
    for i in range(9):
        row = set(grid[i])
        col = set(grid[:, i])
        if row != set(range(1, 10)) or col != set(range(1, 10)):
            return False
    for br in range(3):
        for bc in range(3):
            box = set(grid[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten())
            if box != set(range(1, 10)):
                return False
    return True

def run_gpu_solver(puzzle_str: str, exe_path: str | None = None) -> dict:
    """Run a Davis GPU (CUDA) solver on a single puzzle string via CLI."""
    exe = os.path.abspath(exe_path or GPU_EXE)
    if not os.path.isfile(exe):
        return {"solved": False, "time_ms": -1, "error": f"GPU exe not found: {exe}"}
    try:
        result = subprocess.run(
            [exe, puzzle_str],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout

        # Parse times from output
        wall_match = re.search(r'Wall time:\s+([\d.]+)\s*ms', output)
        gpu_match  = re.search(r'GPU time:\s+([\d.]+)\s*ms', output)
        p1_match   = re.search(r'Phase 1 \(CP\):\s+([\d.]+)\s*ms', output)
        p2_match   = re.search(r'Phase 2 \(Relax\):\s*([\d.]+)\s*ms', output)
        p3_match   = re.search(r'Phase 3 \(DFS\):\s+([\d.]+)\s*ms', output)

        wall_ms = float(wall_match.group(1)) if wall_match else -1
        gpu_ms  = float(gpu_match.group(1))  if gpu_match  else wall_ms
        p1_ms   = float(p1_match.group(1))   if p1_match   else 0
        p2_ms   = float(p2_match.group(1))   if p2_match   else 0
        p3_ms   = float(p3_match.group(1))   if p3_match   else 0

        # Check solved status (e.g. "Solved: 1 / 1  (P1: 0, P2: 0, P3: 1)")
        solved_match = re.search(r'Solved:\s*(\d+)\s*/\s*(\d+)', output)
        solved = (solved_match is not None
                  and solved_match.group(1) == solved_match.group(2)
                  and int(solved_match.group(1)) > 0)

        # Which phase solved it?
        phase_match = re.search(r'P1:\s*(\d+),\s*P2:\s*(\d+),\s*P3:\s*(\d+)', output)
        if phase_match:
            phase = ("P1" if int(phase_match.group(1)) > 0
                     else "P2" if int(phase_match.group(2)) > 0
                     else "P3")
        else:
            phase = "?"

        return {
            "solved": solved, "time_ms": gpu_ms, "wall_ms": wall_ms,
            "p1_ms": p1_ms, "p2_ms": p2_ms, "p3_ms": p3_ms, "phase": phase,
        }
    except subprocess.TimeoutExpired:
        return {"solved": False, "time_ms": 30000, "error": "TIMEOUT"}
    except Exception as e:
        return {"solved": False, "time_ms": -1, "error": str(e)}

def main():
    print("=" * 96)
    print("  World's Hardest Sudoku — Multi-Solver Benchmark")
    print("  Davis GPU v1 · Davis GPU v2 [E1-E6] · Davis GPU v3 [E1-E8] · Davis CPU · DLX · DFS · CP")
    print("=" * 96)

    gpu_available = os.path.isfile(os.path.abspath(GPU_EXE))
    gpu_v2_available = os.path.isfile(os.path.abspath(GPU_V2_EXE))
    gpu_v3_available = os.path.isfile(os.path.abspath(GPU_V3_EXE))
    if gpu_available:
        print(f"  ✓ GPU v1 solver: {os.path.abspath(GPU_EXE)}")
    else:
        print(f"  ✗ GPU v1 solver not found — skipping")
    if gpu_v2_available:
        print(f"  ✓ GPU v2 solver: {os.path.abspath(GPU_V2_EXE)}")
    else:
        print(f"  ✗ GPU v2 solver not found — skipping")
    if gpu_v3_available:
        print(f"  ✓ GPU v3 solver: {os.path.abspath(GPU_V3_EXE)}")
    else:
        print(f"  ✗ GPU v3 solver not found — skipping")

    python_solvers = [
        ("Davis CPU", DavisManifoldSolver),
        ("DLX",       DLXSolver),
        ("DFS",       DFSSolver),
        ("CP",        CPSolver),
    ]

    results = {}

    for name, puzzle_str in HARDEST_PUZZLES.items():
        clues = count_clues(puzzle_str)
        print(f"\n{'─' * 96}")
        print(f"  {name}  ({clues} clues)")
        print(f"{'─' * 96}")
        grid = string_to_grid(puzzle_str)
        results[name] = {"clues": clues}

        # ── Davis GPU (CUDA) ──
        if gpu_available:
            gpu_result = run_gpu_solver(puzzle_str)
            if "error" in gpu_result:
                print(f"  {'Davis GPU':10s}: ERROR — {gpu_result['error']}")
            elif gpu_result["solved"]:
                g = gpu_result
                print(f"  {'Davis GPU':10s}: ✓ SOLVED  {g['time_ms']:8.1f} ms  "
                      f"(P1: {g['p1_ms']:.1f}  P2: {g['p2_ms']:.1f}  "
                      f"P3: {g['p3_ms']:.1f} ms, solved by {g['phase']})")
            else:
                print(f"  {'Davis GPU':10s}: ✗ FAILED  {gpu_result['time_ms']:8.1f} ms")
            results[name]["Davis GPU"] = gpu_result

        # ── Davis GPU v2 (CUDA, thermodynamic enhancements [E1-E6]) ──
        if gpu_v2_available:
            gpu2_result = run_gpu_solver(puzzle_str, exe_path=GPU_V2_EXE)
            if "error" in gpu2_result:
                print(f"  {'Davis GPU v2':14s}: ERROR — {gpu2_result['error']}")
            elif gpu2_result["solved"]:
                g = gpu2_result
                # Show speedup vs v1 if both solved
                speedup = ""
                if gpu_available and "Davis GPU" in results[name] and results[name]["Davis GPU"].get("solved"):
                    v1_ms = results[name]["Davis GPU"]["time_ms"]
                    if g['time_ms'] > 0:
                        ratio = v1_ms / g['time_ms']
                        speedup = f"  [{ratio:.2f}× vs v1]"
                print(f"  {'Davis GPU v2':14s}: ✓ SOLVED  {g['time_ms']:8.1f} ms  "
                      f"(P1: {g['p1_ms']:.1f}  P2: {g['p2_ms']:.1f}  "
                      f"P3: {g['p3_ms']:.1f} ms, solved by {g['phase']}){speedup}")
            else:
                print(f"  {'Davis GPU v2':14s}: ✗ FAILED  {gpu2_result['time_ms']:8.1f} ms")
            results[name]["Davis GPU v2"] = gpu2_result

        # ── Davis GPU v3 (CUDA, speculative branching [E1-E8]) ──
        if gpu_v3_available:
            gpu3_result = run_gpu_solver(puzzle_str, exe_path=GPU_V3_EXE)
            if "error" in gpu3_result:
                print(f"  {'Davis GPU v3':14s}: ERROR — {gpu3_result['error']}")
            elif gpu3_result["solved"]:
                g = gpu3_result
                # Show speedup vs v2 if both solved
                speedup = ""
                if gpu_v2_available and "Davis GPU v2" in results[name] and results[name]["Davis GPU v2"].get("solved"):
                    v2_ms = results[name]["Davis GPU v2"]["time_ms"]
                    if g['time_ms'] > 0:
                        ratio = v2_ms / g['time_ms']
                        speedup = f"  [{ratio:.2f}× vs v2]"
                print(f"  {'Davis GPU v3':14s}: ✓ SOLVED  {g['time_ms']:8.1f} ms  "
                      f"(P1: {g['p1_ms']:.1f}  P2: {g['p2_ms']:.1f}  "
                      f"P3: {g['p3_ms']:.1f} ms, solved by {g['phase']}){speedup}")
            else:
                print(f"  {'Davis GPU v3':14s}: ✗ FAILED  {gpu3_result['time_ms']:8.1f} ms")
            results[name]["Davis GPU v3"] = gpu3_result

        # ── Python solvers (Davis CPU, DLX, DFS, CP) ──
        for solver_name, solver_cls in python_solvers:
            solver = solver_cls()  # fresh instance each puzzle
            board = SudokuBoard(size=9, grid=grid.copy())
            timeout = SOLVER_TIMEOUT.get(solver_name, 120)
            try:
                t0 = time.perf_counter()
                outcome = solve_with_timeout(solver, board, timeout)
                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000

                if outcome == "TIMEOUT":
                    print(f"  {solver_name:10s}: ⏱ TIMEOUT  ({timeout}s limit)")
                    results[name][solver_name] = {"solved": False, "time_ms": timeout * 1000, "error": "TIMEOUT"}
                    continue

                solution, stats = outcome

                if solution is not None:
                    valid = verify_solution(np.array(solution.grid))
                    status = "✓ SOLVED" if valid else "⚠ INVALID"
                else:
                    valid = False
                    status = "✗ FAILED"

                extra = ""
                if solver_name == "Davis CPU" and hasattr(stats, 'extra'):
                    prunes = stats.extra.get("holonomy_prunes", 0)
                    extra = f", hol_prunes: {prunes:,}"

                print(f"  {solver_name:10s}: {status}  {ms:10.1f} ms"
                      f"  (nodes: {stats.nodes_explored:,}, bt: {stats.backtracks:,}{extra})")
                results[name][solver_name] = {
                    "solved": valid,
                    "time_ms": ms,
                    "nodes": stats.nodes_explored,
                    "backtracks": stats.backtracks,
                }
            except Exception as e:
                print(f"  {solver_name:10s}: ERROR — {e}")
                results[name][solver_name] = {"solved": False, "time_ms": -1, "error": str(e)}

    # ── Summary Table ──
    width = 120
    print(f"\n\n{'=' * width}")
    print("  SUMMARY")
    print(f"{'=' * width}")
    header = f"\n{'Puzzle':<30s} {'Clues':>5s}"
    for s in SOLVER_NAMES:
        header += f" {s:>14s}"
    header += f" {'v1→v2':>8s} {'v2→v3':>8s}"
    print(header)
    print("─" * width)

    for name, data in results.items():
        row = f"{name:<30s} {data['clues']:>5d}"
        for s in SOLVER_NAMES:
            if s in data and data[s].get("solved"):
                row += f" {data[s]['time_ms']:>12.1f}ms"
            elif s in data and data[s].get("error") == "TIMEOUT":
                row += f" {'TIMEOUT':>14s}"
            elif s in data and "error" in data[s]:
                row += f" {'ERROR':>14s}"
            elif s in data:
                row += f" {'FAILED':>14s}"
            else:
                row += f" {'N/A':>14s}"
        # Speedup column: v1 / v2
        v1 = data.get("Davis GPU", {})
        v2 = data.get("Davis GPU v2", {})
        v3 = data.get("Davis GPU v3", {})
        if v1.get("solved") and v2.get("solved") and v2["time_ms"] > 0:
            ratio = v1["time_ms"] / v2["time_ms"]
            row += f" {ratio:>7.2f}×"
        else:
            row += f" {'—':>8s}"
        # Speedup column: v2 / v3
        if v2.get("solved") and v3.get("solved") and v3["time_ms"] > 0:
            ratio = v2["time_ms"] / v3["time_ms"]
            row += f" {ratio:>7.2f}×"
        else:
            row += f" {'—':>8s}"
        print(row)

    # Solve counts
    print()
    for s in SOLVER_NAMES:
        solved = sum(1 for d in results.values() if s in d and d[s].get("solved", False))
        total = sum(1 for d in results.values() if s in d)
        print(f"  {s}: {solved}/{total} solved")

if __name__ == "__main__":
    main()
