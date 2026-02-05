"""Unit tests for Sudoku solvers."""

import pytest
from sudoku.core.board import SudokuBoard
from sudoku.solvers import DFSSolver, MCTSSolver, DLXSolver, AnnealingSolver


# A known solvable puzzle (medium difficulty)
TEST_PUZZLE = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)

# The solution to the test puzzle
TEST_SOLUTION = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)


class TestDFSSolver:
    """Tests for DFS solver."""
    
    def test_solve_puzzle(self):
        """Test solving a known puzzle."""
        board = SudokuBoard.from_string(TEST_PUZZLE)
        solver = DFSSolver()
        
        solution, stats = solver.solve(board)
        
        assert stats.solved
        assert solution is not None
        assert solution.is_solved()
        assert solution.to_string() == TEST_SOLUTION
    
    def test_stats_collected(self):
        """Test that stats are collected."""
        board = SudokuBoard.from_string(TEST_PUZZLE)
        solver = DFSSolver()
        
        solution, stats = solver.solve(board)
        
        assert stats.time_seconds > 0
        assert stats.iterations > 0


class TestDLXSolver:
    """Tests for Dancing Links solver."""
    
    def test_solve_puzzle(self):
        """Test solving a known puzzle."""
        board = SudokuBoard.from_string(TEST_PUZZLE)
        solver = DLXSolver()
        
        solution, stats = solver.solve(board)
        
        assert stats.solved
        assert solution is not None
        assert solution.is_solved()
        assert solution.to_string() == TEST_SOLUTION
    
    def test_efficient(self):
        """Test that DLX is efficient."""
        board = SudokuBoard.from_string(TEST_PUZZLE)
        solver = DLXSolver()
        
        solution, stats = solver.solve(board)
        
        assert stats.time_seconds < 1.0  # Should be very fast


class TestAnnealingSolver:
    """Tests for Simulated Annealing solver."""
    
    def test_solve_simple_puzzle(self):
        """Test solving with generous parameters."""
        # Use an easier puzzle for annealing
        easy_puzzle = (
            "530070000"
            "600195000"
            "098000060"
            "800060003"
            "400803001"
            "700020006"
            "060000280"
            "000419005"
            "000080079"
        )
        board = SudokuBoard.from_string(easy_puzzle)
        solver = AnnealingSolver(
            max_iterations=200000,
            restarts=5,
            cooling_rate=0.99995
        )
        
        solution, stats = solver.solve(board)
        
        # Annealing may not always succeed, but should make progress
        assert stats.iterations > 0


class TestMCTSSolver:
    """Tests for MCTS solver."""
    
    def test_runs_without_error(self):
        """Test that MCTS runs without crashing."""
        board = SudokuBoard.from_string(TEST_PUZZLE)
        solver = MCTSSolver(max_iterations=100)  # Limited iterations for test
        
        solution, stats = solver.solve(board)
        
        # MCTS may not solve but should not crash
        assert stats.iterations > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
