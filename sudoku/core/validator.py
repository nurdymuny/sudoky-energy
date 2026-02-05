"""Validation utilities for Sudoku puzzles."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .board import SudokuBoard


def is_valid_placement(board: SudokuBoard, row: int, col: int, value: int) -> bool:
    """
    Check if placing a value at (row, col) is valid.
    
    Args:
        board: The Sudoku board.
        row: Row index.
        col: Column index.
        value: Value to check (1 to board.size).
        
    Returns:
        True if the placement is valid.
    """
    if value < 1 or value > board.size:
        return False
    
    # Check row
    if value in board.get_row(row):
        return False
    
    # Check column
    if value in board.get_col(col):
        return False
    
    # Check box
    if value in board.get_box(row, col):
        return False
    
    return True


def is_valid_board(board: SudokuBoard) -> bool:
    """
    Check if the entire board state is valid (no conflicts).
    
    Args:
        board: The Sudoku board to validate.
        
    Returns:
        True if no constraints are violated.
    """
    return board.is_valid()


def count_solutions(board: SudokuBoard, limit: int = 2) -> int:
    """
    Count the number of solutions for a puzzle (up to limit).
    
    Uses backtracking to count solutions efficiently.
    Stops early once limit is reached.
    
    Args:
        board: The puzzle board.
        limit: Maximum solutions to count before stopping.
        
    Returns:
        Number of solutions found (up to limit).
    """
    work_board = board.copy()
    count = [0]  # Use list to allow modification in nested function
    
    def backtrack() -> bool:
        """Returns True if limit reached."""
        # Find next empty cell
        empty_cells = work_board.get_empty_cells()
        if not empty_cells:
            count[0] += 1
            return count[0] >= limit
        
        # Use MRV heuristic - pick cell with fewest candidates
        min_candidates = board.size + 1
        best_cell = empty_cells[0]
        for cell in empty_cells:
            candidates = work_board.get_candidates(cell[0], cell[1])
            if len(candidates) < min_candidates:
                min_candidates = len(candidates)
                best_cell = cell
                if min_candidates == 0:
                    return False  # No valid candidates, backtrack
        
        row, col = best_cell
        candidates = work_board.get_candidates(row, col)
        
        for val in candidates:
            work_board.set(row, col, val)
            if backtrack():
                return True
            work_board.clear(row, col)
        
        return False
    
    backtrack()
    return count[0]


def has_unique_solution(board: SudokuBoard) -> bool:
    """
    Check if a puzzle has exactly one solution.
    
    Args:
        board: The puzzle board.
        
    Returns:
        True if the puzzle has exactly one solution.
    """
    return count_solutions(board, limit=2) == 1


def validate_solution(puzzle: SudokuBoard, solution: SudokuBoard) -> bool:
    """
    Validate that a solution correctly solves the puzzle.
    
    Args:
        puzzle: The original puzzle.
        solution: The proposed solution.
        
    Returns:
        True if solution is valid and matches puzzle clues.
    """
    if puzzle.size != solution.size:
        return False
    
    # Check that solution respects original clues
    for i in range(puzzle.size):
        for j in range(puzzle.size):
            if not puzzle.is_empty(i, j):
                if puzzle.get(i, j) != solution.get(i, j):
                    return False
    
    # Check that solution is complete and valid
    return solution.is_solved()


def difficulty_score(board: SudokuBoard) -> float:
    """
    Estimate the difficulty of a puzzle based on various factors.
    
    Factors considered:
    - Number of empty cells
    - Distribution of clues
    - Number of naked singles at start
    
    Returns:
        A difficulty score (higher = harder).
    """
    empty_count = board.count_empty()
    
    # Base difficulty from empty cell count
    score = empty_count / 10.0
    
    # Check distribution of clues per row/col/box
    min_clues_per_unit = board.size
    
    for i in range(board.size):
        row_clues = len([x for x in board.get_row(i) if x != 0])
        col_clues = len([x for x in board.get_col(i) if x != 0])
        min_clues_per_unit = min(min_clues_per_unit, row_clues, col_clues)
    
    # Fewer minimum clues = harder
    score += (board.size - min_clues_per_unit) * 0.5
    
    # Count cells with only one candidate (naked singles)
    naked_singles = 0
    for row, col in board.get_empty_cells():
        if len(board.get_candidates(row, col)) == 1:
            naked_singles += 1
    
    # Fewer naked singles = harder
    score -= naked_singles * 0.2
    
    return max(0, score)
