"""Tests for the Davis Manifold solver."""

from sudoku.core.board import SudokuBoard
from sudoku.solvers.davis_solver import DavisManifoldSolver


# Known easy puzzle
EASY_PUZZLE = (
    "941586073"
    "805704090"
    "060000000"
    "180000030"
    "700000080"
    "352801700"
    "629410307"
    "013070620"
    "570360009"
)

# Known hard puzzle (only 15 clues)
HARD_PUZZLE = (
    "000000012"
    "000035000"
    "000600070"
    "700000300"
    "000000000"
    "001000008"
    "040001000"
    "000200000"
    "650000000"
)


def test_davis_solver_easy():
    """Davis solver should handle easy puzzles."""
    board = SudokuBoard.from_string(EASY_PUZZLE)
    solver = DavisManifoldSolver()
    solution, stats = solver.solve(board)
    assert stats.solved
    assert solution is not None


def test_davis_solver_hard():
    """Davis solver should handle hard puzzles (15-clue extreme)."""
    board = SudokuBoard.from_string(HARD_PUZZLE)
    solver = DavisManifoldSolver()
    solution, stats = solver.solve(board)
    assert stats.solved
    assert solution is not None


def test_davis_solver_stats():
    """Davis solver should report geometric stats."""
    board = SudokuBoard.from_string(EASY_PUZZLE)
    solver = DavisManifoldSolver()
    solution, stats = solver.solve(board)
    assert stats.solved
    assert stats.nodes_explored > 0
    assert stats.backtracks >= 0
    assert "holonomy_prunes" in stats.extra


def test_davis_solver_invalid():
    """Davis solver should handle an invalid (unsolvable) puzzle."""
    # Two 5s in the same row
    invalid = "550000000" + "0" * 72
    board = SudokuBoard.from_string(invalid)
    solver = DavisManifoldSolver()
    solution, stats = solver.solve(board)
    assert not stats.solved
