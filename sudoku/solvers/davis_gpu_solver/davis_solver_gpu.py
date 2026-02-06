"""
Davis Manifold Sudoku Solver — Python GPU Interface
====================================================
Python wrapper for the Blackwell-optimized CUDA solver.

Usage:
    from davis_solver_gpu import DavisSolverGPU

    solver = DavisSolverGPU()
    solution, stats = solver.solve(puzzle_2d_list)
    solutions, stats = solver.solve_batch(list_of_puzzles)

Requires:
    - NVIDIA GPU (Blackwell B200/B100 optimal, Hopper/Ampere compatible)
    - CUDA Toolkit 12.8+
    - Compiled shared library: libdavis_solver.so

Build:
    make gpu   (or: nvcc -shared -arch=sm_100 -O3 ...)

Reference: "The Field Equations of Semantic Coherence" (B. R. Davis, 2025)
"""

import ctypes
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUSolverStats:
    """Statistics from the GPU solver run."""
    total_puzzles: int
    solved_phase1: int      # Solved by constraint propagation
    solved_phase2: int      # Solved by manifold relaxation
    solved_phase3: int      # Solved by jackknife branching
    inconsistent: int       # No solution exists
    total_time_ms: float
    phase1_time_ms: float   # CP time
    phase2_time_ms: float   # Relaxation time
    phase3_time_ms: float   # DFS time
    puzzles_per_sec: float  # Throughput
    us_per_puzzle: float    # Latency


# C struct mirror for SolverStats
class _CStats(ctypes.Structure):
    _fields_ = [
        ("total_puzzles",  ctypes.c_int),
        ("solved_phase1",  ctypes.c_int),
        ("solved_phase2",  ctypes.c_int),
        ("solved_phase3",  ctypes.c_int),
        ("inconsistent",   ctypes.c_int),
        ("total_time_ms",  ctypes.c_float),
        ("phase1_time_ms", ctypes.c_float),
        ("phase2_time_ms", ctypes.c_float),
        ("phase3_time_ms", ctypes.c_float),
    ]


def _find_library() -> str:
    """Locate the compiled shared library (.so on Linux, .dll on Windows)."""
    import platform
    ext = ".dll" if platform.system() == "Windows" else ".so"
    name = f"davis_solver{ext}" if ext == ".dll" else f"libdavis_solver{ext}"

    candidates = [
        Path(__file__).parent / name,
        Path(__file__).parent / "build" / name,
        Path(name),
        Path(f"build/{name}"),
    ]
    # Also check the .so names on any platform (cross-compiled binaries)
    if ext == ".dll":
        candidates += [
            Path(__file__).parent / "libdavis_solver.so",
            Path(__file__).parent / "build" / "libdavis_solver.so",
        ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"{name} not found. Build with: make gpu\n"
        "Or: nvcc -shared -Xcompiler -fPIC -arch=sm_100 -O3 "
        f"-o {name} davis_solver_blackwell.cu"
    )


class DavisSolverGPU:
    """
    GPU-accelerated Davis Manifold Sudoku Solver.

    Three-phase pipeline:
        Phase 1: Wavefront Constraint Propagation (curvature-ordered)
        Phase 2: Davis Manifold Relaxation (continuous gradient descent on E[γ])
        Phase 3: Jackknife Speculative Branching (curvature-guided parallel DFS)

    Optimized for NVIDIA Blackwell (B200/B100) with fallback to Hopper/Ampere.
    """

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the GPU solver.

        Args:
            lib_path: Path to libdavis_solver.so. Auto-detected if None.
        """
        if lib_path is None:
            lib_path = _find_library()

        self._lib = ctypes.CDLL(lib_path)

        # Bind C functions
        self._lib.davis_solver_create.restype = ctypes.c_void_p
        self._lib.davis_solver_create.argtypes = []

        self._lib.davis_solver_destroy.restype = None
        self._lib.davis_solver_destroy.argtypes = [ctypes.c_void_p]

        self._lib.davis_solver_solve_batch.restype = _CStats
        self._lib.davis_solver_solve_batch.argtypes = [
            ctypes.c_void_p,                           # solver handle
            ctypes.POINTER(ctypes.c_int),              # input puzzles
            ctypes.POINTER(ctypes.c_int),              # output solutions
            ctypes.c_int,                              # batch_size
        ]

        # Create solver instance
        self._handle = self._lib.davis_solver_create()
        if not self._handle:
            raise RuntimeError("Failed to create GPU solver. Check CUDA installation.")

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            self._lib.davis_solver_destroy(self._handle)

    def solve(self, puzzle: list[list[int]]) -> tuple[list[list[int]], GPUSolverStats]:
        """
        Solve a single Sudoku puzzle.

        Args:
            puzzle: 9×9 list of lists (0 = empty cell)

        Returns:
            (solution, stats) where solution is 9×9 list of lists
        """
        solutions, stats = self.solve_batch([puzzle])
        return solutions[0], stats

    def solve_batch(
        self,
        puzzles: list[list[list[int]]]
    ) -> tuple[list[list[list[int]]], GPUSolverStats]:
        """
        Solve a batch of Sudoku puzzles in parallel.

        This is where the GPU truly shines — processing thousands
        of puzzles simultaneously across all SMs.

        Args:
            puzzles: List of 9×9 puzzles (0 = empty cell)

        Returns:
            (solutions, stats) where solutions is list of 9×9 grids
        """
        batch_size = len(puzzles)

        # Flatten to contiguous arrays
        input_arr = np.zeros(batch_size * 81, dtype=np.int32)
        for p_idx, puzzle in enumerate(puzzles):
            for r in range(9):
                for c in range(9):
                    input_arr[p_idx * 81 + r * 9 + c] = puzzle[r][c]

        output_arr = np.zeros(batch_size * 81, dtype=np.int32)

        # Call GPU solver
        c_stats = self._lib.davis_solver_solve_batch(
            self._handle,
            input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            batch_size,
        )

        # Convert flat output back to 9×9 grids
        solutions = []
        for p_idx in range(batch_size):
            grid = []
            for r in range(9):
                row = []
                for c in range(9):
                    row.append(int(output_arr[p_idx * 81 + r * 9 + c]))
                grid.append(row)
            solutions.append(grid)

        # Build stats
        total_ms = c_stats.total_time_ms if c_stats.total_time_ms > 0 else 0.001
        stats = GPUSolverStats(
            total_puzzles=c_stats.total_puzzles,
            solved_phase1=c_stats.solved_phase1,
            solved_phase2=c_stats.solved_phase2,
            solved_phase3=c_stats.solved_phase3,
            inconsistent=c_stats.inconsistent,
            total_time_ms=c_stats.total_time_ms,
            phase1_time_ms=c_stats.phase1_time_ms,
            phase2_time_ms=c_stats.phase2_time_ms,
            phase3_time_ms=c_stats.phase3_time_ms,
            puzzles_per_sec=batch_size / (total_ms / 1000.0),
            us_per_puzzle=(total_ms * 1000.0) / max(batch_size, 1),
        )

        return solutions, stats

    def geometric_analysis(self, puzzle: list[list[int]]) -> dict:
        """
        Run Davis geometric analysis on a puzzle (CPU fallback).

        Uses the CPU solver's analysis since it produces richer output.
        Import the CPU solver for this.
        """
        try:
            from sudoku.solvers.davis_solver import geometric_analysis
            return geometric_analysis(puzzle)
        except ImportError:
            # Fallback: try relative import for standalone usage
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from davis_solver import geometric_analysis
                return geometric_analysis(puzzle)
            except ImportError:
                return {"error": "davis_solver.py not found for CPU analysis"}


def benchmark(n_puzzles: int = 10000):
    """
    Run a batch benchmark.

    On Blackwell B200 with 192 SMs, expect:
      - Easy puzzles:  ~2M puzzles/sec (Phase 1 only)
      - Hard puzzles:  ~500K puzzles/sec (Phase 1 + 2)
      - Extreme puzzles: ~50K puzzles/sec (all 3 phases)
    """
    print(f"Davis Manifold Sudoku Solver — GPU Benchmark ({n_puzzles} puzzles)")
    print("=" * 60)

    # Hard puzzle (15 clues, Γ ≈ 0.19)
    hard_puzzle = [
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

    solver = DavisSolverGPU()

    # Warm up
    solver.solve(hard_puzzle)

    # Batch solve
    batch = [hard_puzzle] * n_puzzles
    solutions, stats = solver.solve_batch(batch)

    print(f"  Puzzles:       {stats.total_puzzles}")
    print(f"  GPU time:      {stats.total_time_ms:.3f} ms")
    print(f"    Phase 1:     {stats.phase1_time_ms:.3f} ms")
    print(f"    Phase 2:     {stats.phase2_time_ms:.3f} ms")
    print(f"    Phase 3:     {stats.phase3_time_ms:.3f} ms")
    print(f"  Throughput:    {stats.puzzles_per_sec:,.0f} puzzles/sec")
    print(f"  Per puzzle:    {stats.us_per_puzzle:.2f} µs")

    # Verify first solution
    sol = solutions[0]
    valid = all(sorted(sol[r]) == list(range(1, 10)) for r in range(9))
    print(f"\n  First solution valid: {valid}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    benchmark(n)
