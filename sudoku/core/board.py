"""Sudoku board representation with support for variable sizes."""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Set
import copy


class SudokuBoard:
    """
    Represents a Sudoku board of configurable size.
    
    Standard Sudoku is 9x9 with 3x3 boxes.
    Supports larger boards: 16x16 (4x4 boxes), 25x25 (5x5 boxes).
    """
    
    def __init__(self, size: int = 9, grid: Optional[np.ndarray] = None):
        """
        Initialize a Sudoku board.
        
        Args:
            size: Board size (9, 16, or 25). Must be a perfect square.
            grid: Optional initial grid. If None, creates empty board.
        """
        # Validate size is a perfect square
        box_size = int(np.sqrt(size))
        if box_size * box_size != size:
            raise ValueError(f"Size must be a perfect square, got {size}")
        
        self.size = size
        self.box_size = box_size
        
        if grid is not None:
            if grid.shape != (size, size):
                raise ValueError(f"Grid shape must be ({size}, {size})")
            self.grid = grid.copy().astype(np.int32)
        else:
            self.grid = np.zeros((size, size), dtype=np.int32)
    
    def copy(self) -> SudokuBoard:
        """Create a deep copy of the board."""
        new_board = SudokuBoard(self.size)
        new_board.grid = self.grid.copy()
        return new_board
    
    def get(self, row: int, col: int) -> int:
        """Get value at position (row, col). 0 means empty."""
        return self.grid[row, col]
    
    def set(self, row: int, col: int, value: int) -> None:
        """Set value at position (row, col). Use 0 to clear."""
        if value < 0 or value > self.size:
            raise ValueError(f"Value must be 0-{self.size}, got {value}")
        self.grid[row, col] = value
    
    def clear(self, row: int, col: int) -> None:
        """Clear the cell at position (row, col)."""
        self.grid[row, col] = 0
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if cell is empty (value is 0)."""
        return self.grid[row, col] == 0
    
    def get_row(self, row: int) -> np.ndarray:
        """Get all values in a row."""
        return self.grid[row, :]
    
    def get_col(self, col: int) -> np.ndarray:
        """Get all values in a column."""
        return self.grid[:, col]
    
    def get_box(self, row: int, col: int) -> np.ndarray:
        """Get all values in the box containing (row, col)."""
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        return self.grid[box_row:box_row + self.box_size,
                        box_col:box_col + self.box_size].flatten()
    
    def get_box_index(self, row: int, col: int) -> int:
        """Get the box index (0 to size-1) for a cell."""
        return (row // self.box_size) * self.box_size + (col // self.box_size)
    
    def get_candidates(self, row: int, col: int) -> Set[int]:
        """
        Get all valid candidate values for an empty cell.
        
        Returns:
            Set of valid values (1 to size) that can be placed at (row, col).
            Returns empty set if cell is not empty.
        """
        if not self.is_empty(row, col):
            return set()
        
        all_values = set(range(1, self.size + 1))
        
        # Remove values in same row
        used = set(self.get_row(row)) - {0}
        # Remove values in same column
        used |= set(self.get_col(col)) - {0}
        # Remove values in same box
        used |= set(self.get_box(row, col)) - {0}
        
        return all_values - used
    
    def get_peers(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """
        Get all peer cell positions (those in same row, column, or box).
        Args:
            row, col: Cell position.
        Returns:
            Set of (r, c) tuples, excluding (row, col) itself.
        """
        peers = set()
        # Row and Col peers
        for i in range(self.size):
            peers.add((row, i))
            peers.add((i, col))
            
        # Box peers
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        for i in range(self.box_size):
            for j in range(self.box_size):
                peers.add((box_row + i, box_col + j))
                
        peers.remove((row, col))
        return peers

    def is_valid_move(self, row: int, col: int, value: int) -> bool:
        """
        Check if placing value at (row, col) is valid.
        """
        if value == 0: return True
        
        # Check row
        if value in self.grid[row, :]:
            # If the value is already there, it's valid ONLY if it's at (row, col)
            # But normally we call this BEFORE setting. 
            # If we call it after setting, we need to be careful.
            # Assuming we call it BEFORE setting:
            return False
            
        # Check column
        if value in self.grid[:, col]:
            return False
            
        # Check box
        if value in self.get_box(row, col):
            return False
            
        return True
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get list of all empty cell positions."""
        empty = []
        for i in range(self.size):
            for j in range(self.size):
                if self.is_empty(i, j):
                    empty.append((i, j))
        return empty
    
    def count_empty(self) -> int:
        """Count the number of empty cells."""
        return int(np.sum(self.grid == 0))
    
    def count_filled(self) -> int:
        """Count the number of filled cells."""
        return int(np.sum(self.grid != 0))
    
    def is_complete(self) -> bool:
        """Check if all cells are filled."""
        return self.count_empty() == 0
    
    def is_valid(self) -> bool:
        """
        Check if the current board state is valid.
        Does not check if solution is complete, only if no conflicts exist.
        """
        # Check all rows
        for i in range(self.size):
            row = self.get_row(i)
            non_zero = row[row != 0]
            if len(non_zero) != len(set(non_zero)):
                return False
        
        # Check all columns
        for j in range(self.size):
            col = self.get_col(j)
            non_zero = col[col != 0]
            if len(non_zero) != len(set(non_zero)):
                return False
        
        # Check all boxes
        for box_row in range(0, self.size, self.box_size):
            for box_col in range(0, self.size, self.box_size):
                box = self.get_box(box_row, box_col)
                non_zero = box[box != 0]
                if len(non_zero) != len(set(non_zero)):
                    return False
        
        return True
    
    def is_solved(self) -> bool:
        """Check if the puzzle is completely and correctly solved."""
        return self.is_complete() and self.is_valid()
    
    def to_string(self) -> str:
        """
        Convert board to a compact string representation.
        Uses 0 for empty cells, 1-9 for standard, A-G for 16x16.
        """
        chars = []
        for i in range(self.size):
            for j in range(self.size):
                val = self.grid[i, j]
                if val == 0:
                    chars.append('0')
                elif val <= 9:
                    chars.append(str(val))
                else:
                    chars.append(chr(ord('A') + val - 10))
        return ''.join(chars)
    
    @classmethod
    def from_string(cls, s: str, size: int = 9) -> SudokuBoard:
        """
        Create a board from a string representation.
        
        Args:
            s: String of length size*size with values.
               0 or . for empty, 1-9 for values, A-G for 10-16.
            size: Board size.
        """
        if len(s) != size * size:
            raise ValueError(f"String length must be {size*size}, got {len(s)}")
        
        grid = np.zeros((size, size), dtype=np.int32)
        idx = 0
        for i in range(size):
            for j in range(size):
                c = s[idx]
                if c == '0' or c == '.':
                    grid[i, j] = 0
                elif c.isdigit():
                    grid[i, j] = int(c)
                else:
                    grid[i, j] = ord(c.upper()) - ord('A') + 10
                idx += 1
        
        return cls(size, grid)
    
    @classmethod
    def from_2d_list(cls, data: List[List[int]]) -> SudokuBoard:
        """Create a board from a 2D list."""
        arr = np.array(data, dtype=np.int32)
        size = arr.shape[0]
        return cls(size, arr)
    
    def __str__(self) -> str:
        """Pretty-print the board."""
        lines = []
        horizontal_sep = '+' + (('-' * (self.box_size * 2 + 1)) + '+') * self.box_size
        
        for i in range(self.size):
            if i % self.box_size == 0:
                lines.append(horizontal_sep)
            
            row_str = '|'
            for j in range(self.size):
                val = self.grid[i, j]
                if val == 0:
                    row_str += ' .'
                elif val <= 9:
                    row_str += f' {val}'
                else:
                    row_str += f' {chr(ord("A") + val - 10)}'
                
                if (j + 1) % self.box_size == 0:
                    row_str += ' |'
            
            lines.append(row_str)
        
        lines.append(horizontal_sep)
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return f"SudokuBoard(size={self.size}, filled={self.count_filled()})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SudokuBoard):
            return False
        return self.size == other.size and np.array_equal(self.grid, other.grid)
    
    def __hash__(self) -> int:
        return hash(self.to_string())
