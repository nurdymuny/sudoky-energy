"""Benchmarking framework for comparing Sudoku solvers."""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
import os

from tqdm import tqdm

from ..core.board import SudokuBoard
from ..generator import SudokuGenerator, Difficulty
from ..solvers import BaseSolver, DFSSolver, MCTSSolver, DLXSolver, AnnealingSolver, CPSolver, DavisManifoldSolver


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    puzzle_id: int
    difficulty: str
    algorithm: str
    solved: bool
    time_seconds: float
    memory_bytes: int
    iterations: int
    backtracks: int
    nodes_explored: int
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "puzzle_id": self.puzzle_id,
            "difficulty": self.difficulty,
            "algorithm": self.algorithm,
            "solved": self.solved,
            "time_seconds": self.time_seconds,
            "memory_bytes": self.memory_bytes,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "iterations": self.iterations,
            "backtracks": self.backtracks,
            "nodes_explored": self.nodes_explored,
            **self.extra
        }


class Benchmark:
    """
    Benchmark framework for comparing Sudoku solving algorithms.
    
    Runs multiple solvers on generated puzzles and collects performance metrics.
    """
    
    DEFAULT_SOLVERS: Dict[str, Type[BaseSolver]] = {
        "DFS": DFSSolver,
        "MCTS": MCTSSolver,
        "DLX": DLXSolver,
        "Annealing": AnnealingSolver,
        "CP": CPSolver,
        "Davis": DavisManifoldSolver
    }
    
    def __init__(
        self,
        puzzles_per_difficulty: int = 10,
        difficulties: Optional[List[Difficulty]] = None,
        solvers: Optional[Dict[str, BaseSolver]] = None,
        timeout_seconds: float = 60.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the benchmark.
        
        Args:
            puzzles_per_difficulty: Number of puzzles to generate per difficulty.
            difficulties: List of difficulties to test (default: all).
            solvers: Dict of solver_name -> solver_instance (default: all).
            timeout_seconds: Maximum time per puzzle per solver.
            seed: Random seed for reproducibility.
        """
        self.puzzles_per_difficulty = puzzles_per_difficulty
        self.difficulties = difficulties or list(Difficulty)
        self.timeout_seconds = timeout_seconds
        self.seed = seed
        
        # Initialize solvers
        if solvers is None:
            self.solvers = {
                "DFS": DFSSolver(),
                "MCTS": MCTSSolver(max_iterations=5000),
                "DLX": DLXSolver(),
                "Annealing": AnnealingSolver(max_iterations=100000, restarts=3),
                "CP": CPSolver(),
                "Davis": DavisManifoldSolver()
            }
        else:
            self.solvers = solvers
        
        self.puzzles: Dict[str, List[SudokuBoard]] = {}
        
        # Load optimal parameters for Annealing if they exist
        self.optimal_params = {}
        opt_file = "results/tuning/optimal_parameters.json"
        if os.path.exists(opt_file):
            try:
                with open(opt_file, "r") as f:
                    self.optimal_params = json.load(f)
                print(f"Loaded optimal parameters for Simulated Annealing from {opt_file}")
            except Exception as e:
                print(f"Warning: Could not load optimal parameters: {e}")
    
    def generate_puzzles(self) -> None:
        """Generate all puzzles for benchmarking."""
        generator = SudokuGenerator(seed=self.seed)
        
        print("Generating puzzles...")
        for difficulty in tqdm(self.difficulties, desc="Difficulties"):
            self.puzzles[difficulty.value] = generator.generate_batch(
                self.puzzles_per_difficulty, 
                difficulty
            )
    
    def run(self, show_progress: bool = True) -> List[BenchmarkResult]:
        """
        Run the full benchmark suite.
        
        Returns:
            List of BenchmarkResult objects.
        """
        if not self.puzzles:
            self.generate_puzzles()
        
        self.results = []
        
        total_tests = (
            len(self.puzzles) * 
            self.puzzles_per_difficulty * 
            len(self.solvers)
        )
        
        pbar = tqdm(total=total_tests, desc="Benchmarking", disable=not show_progress)
        
        for difficulty_name, puzzles in self.puzzles.items():
            for puzzle_id, puzzle in enumerate(puzzles):
                for solver_name, solver in self.solvers.items():
                    result = self._run_single(
                        puzzle, puzzle_id, difficulty_name, solver_name, solver
                    )
                    self.results.append(result)
                    pbar.update(1)
        
        pbar.close()
        return self.results
    
    def _run_single(
        self,
        puzzle: SudokuBoard,
        puzzle_id: int,
        difficulty: str,
        solver_name: str,
        solver: BaseSolver
    ) -> BenchmarkResult:
        """Run a single solver on a single puzzle."""
        # Apply optimal parameters for Annealing based on difficulty
        if solver_name == "Annealing" and difficulty in self.optimal_params:
            params = self.optimal_params[difficulty].get("params", {})
            for key, val in params.items():
                if hasattr(solver, key):
                    setattr(solver, key, val)
        
        # Use ThreadPoolExecutor to enforce timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(solver.solve, puzzle)
            try:
                solution, stats = future.result(timeout=self.timeout_seconds)
                
                return BenchmarkResult(
                    puzzle_id=puzzle_id,
                    difficulty=difficulty,
                    algorithm=solver_name,
                    solved=stats.solved,
                    time_seconds=stats.time_seconds,
                    memory_bytes=stats.memory_bytes,
                    iterations=stats.iterations,
                    backtracks=stats.backtracks,
                    nodes_explored=stats.nodes_explored,
                    extra=stats.extra
                )
            except TimeoutError:
                return BenchmarkResult(
                    puzzle_id=puzzle_id,
                    difficulty=difficulty,
                    algorithm=solver_name,
                    solved=False,
                    time_seconds=self.timeout_seconds,
                    memory_bytes=0,
                    iterations=0,
                    backtracks=0,
                    nodes_explored=0,
                    extra={"error": "Timeout"}
                )
            except Exception as e:
                return BenchmarkResult(
                    puzzle_id=puzzle_id,
                    difficulty=difficulty,
                    algorithm=solver_name,
                    solved=False,
                    time_seconds=self.timeout_seconds,
                    memory_bytes=0,
                    iterations=0,
                    backtracks=0,
                    nodes_explored=0,
                    extra={"error": str(e)}
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from benchmark results."""
        summary = {
            "total_puzzles": len(self.results) // len(self.solvers),
            "solvers_tested": list(self.solvers.keys()),
            "difficulties": [d.value for d in self.difficulties],
            "results_by_algorithm": {},
            "results_by_difficulty": {}
        }
        
        # Group by algorithm
        for solver_name in self.solvers:
            solver_results = [r for r in self.results if r.algorithm == solver_name]
            if solver_results:
                solved = [r for r in solver_results if r.solved]
                times = [r.time_seconds for r in solver_results]
                memory = [r.memory_bytes for r in solver_results]
                
                summary["results_by_algorithm"][solver_name] = {
                    "accuracy": len(solved) / len(solver_results) * 100,
                    "avg_time_seconds": sum(times) / len(times),
                    "max_time_seconds": max(times),
                    "min_time_seconds": min(times),
                    "avg_memory_mb": sum(memory) / len(memory) / (1024 * 1024),
                    "total_solved": len(solved),
                    "total_tested": len(solver_results)
                }
        
        # Group by difficulty
        for difficulty in self.difficulties:
            diff_results = [r for r in self.results if r.difficulty == difficulty.value]
            if diff_results:
                summary["results_by_difficulty"][difficulty.value] = {}
                
                for solver_name in self.solvers:
                    solver_diff_results = [
                        r for r in diff_results if r.algorithm == solver_name
                    ]
                    if solver_diff_results:
                        solved = [r for r in solver_diff_results if r.solved]
                        times = [r.time_seconds for r in solver_diff_results]
                        
                        summary["results_by_difficulty"][difficulty.value][solver_name] = {
                            "accuracy": len(solved) / len(solver_diff_results) * 100,
                            "avg_time_seconds": sum(times) / len(times),
                            "solved": len(solved),
                            "tested": len(solver_diff_results)
                        }
        
        return summary
    
    def save_results(self, output_dir: str) -> None:
        """Save benchmark results and generated puzzles to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results as JSON
        results_file = os.path.join(output_dir, "benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        # Save summary
        summary_file = os.path.join(output_dir, "benchmark_summary.json")
        with open(summary_file, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
        
        # Save puzzles by difficulty
        puzzles_dir = os.path.join(output_dir, "puzzles")
        os.makedirs(puzzles_dir, exist_ok=True)
        
        for difficulty, puzzles in self.puzzles.items():
            diff_dir = os.path.join(puzzles_dir, difficulty)
            SudokuGenerator.save_to_folder(puzzles, diff_dir, prefix=f"puzzle_{difficulty}")
        
        print(f"Results and puzzles saved to {output_dir}")
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
            return pd.DataFrame([r.to_dict() for r in self.results])
        except ImportError:
            raise ImportError("pandas is required for DataFrame conversion")
