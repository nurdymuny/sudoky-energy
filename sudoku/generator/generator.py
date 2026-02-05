"""Sudoku puzzle generator with configurable difficulty levels."""

from __future__ import annotations
import random
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np

from ..core.board import SudokuBoard
from ..core.validator import has_unique_solution


class Difficulty(Enum):
    """Difficulty levels for Sudoku puzzles."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    
    @property
    def clue_range(self) -> Tuple[int, int]:
        """Get the range of clues for this difficulty (min, max)."""
        ranges = {
            Difficulty.EASY: (36, 45),      # 35-45 clues
            Difficulty.MEDIUM: (28, 35),    # 28-34 clues
            Difficulty.HARD: (22, 27),      # 22-27 clues
            Difficulty.EXPERT: (17, 21),    # 17-21 clues (17 is minimum for unique solution)
        }
        return ranges[self]


class SudokuGenerator:
    """
    Generator for Sudoku puzzles with various difficulty levels.
    
    Algorithm:
    1. Generate a complete valid Sudoku solution using backtracking
    2. Remove cells based on difficulty level
    3. Verify the puzzle has a unique solution
    """
    
    def __init__(self, size: int = 9, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            size: Board size (default 9 for standard Sudoku).
            seed: Random seed for reproducibility.
        """
        self.size = size
        self.box_size = int(np.sqrt(size))
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate(self, difficulty: Difficulty = Difficulty.MEDIUM) -> SudokuBoard:
        """
        Generate a Sudoku puzzle with the specified difficulty.
        
        Args:
            difficulty: Desired difficulty level.
            
        Returns:
            A SudokuBoard with the puzzle (with clues only, no solution).
        """
        # Step 1: Generate complete solution
        solution = self._generate_complete_board()
        
        # Step 2: Remove cells to create puzzle
        puzzle = self._remove_cells(solution, difficulty)
        
        return puzzle
    
    def generate_batch(self, count: int, difficulty: Difficulty = Difficulty.MEDIUM) -> List[SudokuBoard]:
        """
        Generate multiple puzzles of the same difficulty.
        
        Args:
            count: Number of puzzles to generate.
            difficulty: Desired difficulty level.
            
        Returns:
            List of SudokuBoard puzzles.
        """
        return [self.generate(difficulty) for _ in range(count)]
    
    def generate_with_solution(self, difficulty: Difficulty = Difficulty.MEDIUM) -> Tuple[SudokuBoard, SudokuBoard]:
        """
        Generate a puzzle along with its solution.
        
        Args:
            difficulty: Desired difficulty level.
            
        Returns:
            Tuple of (puzzle, solution) SudokuBoards.
        """
        solution = self._generate_complete_board()
        puzzle = self._remove_cells(solution, difficulty)
        return puzzle, solution
    
    def _generate_complete_board(self) -> SudokuBoard:
        """Generate a complete valid Sudoku board using backtracking."""
        board = SudokuBoard(self.size)
        self._fill_board(board)
        return board
    
    def _fill_board(self, board: SudokuBoard) -> bool:
        """
        Fill the board using randomized backtracking.
        
        Uses optimized filling: start with diagonal boxes which have no constraints,
        then fill the rest.
        """
        # Fill diagonal boxes first (they are independent)
        for box_idx in range(self.box_size):
            box_row = box_idx * self.box_size
            box_col = box_idx * self.box_size
            self._fill_box(board, box_row, box_col)
        
        # Fill remaining cells using backtracking
        return self._solve_remaining(board)
    
    def _fill_box(self, board: SudokuBoard, start_row: int, start_col: int) -> None:
        """Fill a single box with random values."""
        values = list(range(1, self.size + 1))
        random.shuffle(values)
        
        idx = 0
        for i in range(self.box_size):
            for j in range(self.box_size):
                board.set(start_row + i, start_col + j, values[idx])
                idx += 1
    
    def _solve_remaining(self, board: SudokuBoard) -> bool:
        """Fill remaining cells using backtracking with random ordering."""
        # Find empty cell
        empty_cells = board.get_empty_cells()
        if not empty_cells:
            return True
        
        # Use MRV (Minimum Remaining Values) heuristic
        min_candidates = self.size + 1
        best_cell = empty_cells[0]
        for cell in empty_cells:
            candidates = board.get_candidates(cell[0], cell[1])
            if len(candidates) < min_candidates:
                min_candidates = len(candidates)
                best_cell = cell
        
        row, col = best_cell
        candidates = list(board.get_candidates(row, col))
        random.shuffle(candidates)  # Randomize for variety
        
        for val in candidates:
            board.set(row, col, val)
            if self._solve_remaining(board):
                return True
            board.clear(row, col)
        
        return False
    
    def _remove_cells(self, solution: SudokuBoard, difficulty: Difficulty) -> SudokuBoard:
        """
        Remove cells from a complete solution to create a puzzle.
        
        Ensures the resulting puzzle has a unique solution.
        """
        puzzle = solution.copy()
        min_clues, max_clues = difficulty.clue_range
        
        # Calculate target number of cells to remove
        total_cells = self.size * self.size
        target_clues = random.randint(min_clues, max_clues)
        cells_to_remove = total_cells - target_clues
        
        # Get list of all filled cells
        filled_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if not puzzle.is_empty(i, j):
                    filled_cells.append((i, j))
        
        random.shuffle(filled_cells)
        
        removed = 0
        attempts = 0
        max_attempts = len(filled_cells)  # Try each cell at most once
        
        for row, col in filled_cells:
            if removed >= cells_to_remove:
                break
            
            if attempts >= max_attempts:
                break
            
            attempts += 1
            
            # Try removing this cell
            original_value = puzzle.get(row, col)
            puzzle.clear(row, col)
            
            # Check if puzzle still has unique solution
            if has_unique_solution(puzzle):
                removed += 1
            else:
                # Restore the cell
                puzzle.set(row, col, original_value)
        
        return puzzle
    
    @staticmethod
    def save_to_folder(puzzles: List[SudokuBoard], folder_path: str, prefix: str = "puzzle") -> None:
        """
        Save a list of puzzles to a folder as individual text files.
        
        Args:
            puzzles: List of SudokuBoard objects.
            folder_path: Directory to save the puzzles.
            prefix: Prefix for the filename (default: "puzzle").
        """
        import os
        os.makedirs(folder_path, exist_ok=True)
        
        for i, puzzle in enumerate(puzzles, 1):
            file_path = os.path.join(folder_path, f"{prefix}_{i}.txt")
            with open(file_path, "w") as f:
                f.write(puzzle.to_string())
                f.write("\n\nPretty format:\n")
                f.write(str(puzzle))
