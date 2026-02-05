"""Base solver interface and common utilities."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time
import tracemalloc

from ..core.board import SudokuBoard


@dataclass
class SolverStats:
    """Statistics from a solver run."""
    # Core metrics
    solved: bool = False
    time_seconds: float = 0.0
    memory_bytes: int = 0
    iterations: int = 0
    
    # Algorithm-specific metrics
    backtracks: int = 0
    nodes_explored: int = 0
    
    # Additional metadata
    algorithm: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "solved": self.solved,
            "time_seconds": self.time_seconds,
            "memory_bytes": self.memory_bytes,
            "iterations": self.iterations,
            "backtracks": self.backtracks,
            "nodes_explored": self.nodes_explored,
            "algorithm": self.algorithm,
            **self.extra
        }


class BaseSolver(ABC):
    """Abstract base class for Sudoku solvers."""
    
    name: str = "BaseSolver"
    
    def __init__(self):
        self.stats = SolverStats(algorithm=self.name)
    
    def solve(self, board: SudokuBoard) -> tuple[Optional[SudokuBoard], SolverStats]:
        """
        Solve a Sudoku puzzle with timing and memory tracking.
        
        Args:
            board: The puzzle to solve.
            
        Returns:
            Tuple of (solution or None, stats).
        """
        self.stats = SolverStats(algorithm=self.name)
        
        # Start memory tracking
        tracemalloc.start()
        
        # Start timing
        start_time = time.perf_counter()
        
        try:
            solution = self._solve(board.copy())
            self.stats.solved = solution is not None and solution.is_solved()
        except Exception as e:
            self.stats.extra["error"] = str(e)
            solution = None
        
        # End timing
        self.stats.time_seconds = time.perf_counter() - start_time
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.stats.memory_bytes = peak
        
        return solution, self.stats
    
    @abstractmethod
    def _solve(self, board: SudokuBoard) -> Optional[SudokuBoard]:
        """
        Internal solve method to be implemented by subclasses.
        
        Args:
            board: A copy of the puzzle to solve (can be modified).
            
        Returns:
            The solved board, or None if no solution found.
        """
        pass
    
    def reset_stats(self) -> None:
        """Reset solver statistics."""
        self.stats = SolverStats(algorithm=self.name)
