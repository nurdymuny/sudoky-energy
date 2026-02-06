"""Tests for the Constraint Programming (CP) solver."""

import pytest
from sudoku.core.board import SudokuBoard
from sudoku.solvers.cp_solver import CPSolver

def test_cp_solver_easy():
    """Test CP solver on a simple puzzle."""
    puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    board = SudokuBoard.from_string(puzzle_str)
    solver = CPSolver()
    solution, stats = solver.solve(board)
    assert stats.solved
    assert solution.is_solved()

def test_cp_solver_no_guess_easy():
    """Test that CP solver solves easy puzzles without backtracks."""
    # This puzzle is from a source known to be solvable by simple logic
    puzzle_str = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
    board = SudokuBoard.from_string(puzzle_str)
    solver = CPSolver()
    solution, stats = solver.solve(board)
    assert stats.solved
    # Note: stats.iterations counts backtracking calls. 
    # If it solves by pure logic, iterations should be 0 (or 1 depending on implementation).
    # My current implementation sets iterations = 0 and then increments in _backtrack.
    assert stats.iterations == 0

def test_cp_solver_hard():
    """Test CP solver on a hard puzzle."""
    puzzle_str = "000000000000003085001020000000507000004000100090000000500009007070040000300000008"
    board = SudokuBoard.from_string(puzzle_str)
    solver = CPSolver()
    solution, stats = solver.solve(board)
    assert stats.solved
    assert solution.is_solved()

def test_cp_solver_invalid():
    """Test CP solver on an invalid puzzle."""
    puzzle_str = "550070000600195000098000060800060003400803001700020006060000280000419005000080079"
    # (Two 5s in first row)
    board = SudokuBoard.from_string(puzzle_str)
    solver = CPSolver()
    solution, stats = solver.solve(board)
    assert not stats.solved
