"""Unit tests for Sudoku board and validation."""

import pytest
import numpy as np
from sudoku.core.board import SudokuBoard
from sudoku.core.validator import is_valid_placement, has_unique_solution


class TestSudokuBoard:
    """Tests for SudokuBoard class."""
    
    def test_create_empty_board(self):
        """Test creating an empty 9x9 board."""
        board = SudokuBoard()
        assert board.size == 9
        assert board.box_size == 3
        assert board.count_empty() == 81
        assert board.count_filled() == 0
    
    def test_create_16x16_board(self):
        """Test creating a 16x16 board."""
        board = SudokuBoard(size=16)
        assert board.size == 16
        assert board.box_size == 4
    
    def test_set_and_get(self):
        """Test setting and getting values."""
        board = SudokuBoard()
        board.set(0, 0, 5)
        assert board.get(0, 0) == 5
        assert not board.is_empty(0, 0)
        
        board.clear(0, 0)
        assert board.is_empty(0, 0)
    
    def test_get_candidates(self):
        """Test getting valid candidates for a cell."""
        board = SudokuBoard()
        board.set(0, 0, 5)
        board.set(0, 1, 3)
        
        # Cell (0, 2) should not have 5 or 3 as candidates
        candidates = board.get_candidates(0, 2)
        assert 5 not in candidates
        assert 3 not in candidates
        assert len(candidates) == 7  # 1-9 minus 5 and 3
    
    def test_is_valid(self):
        """Test board validation."""
        board = SudokuBoard()
        assert board.is_valid()  # Empty board is valid
        
        board.set(0, 0, 5)
        board.set(0, 1, 5)  # Duplicate in row
        assert not board.is_valid()
    
    def test_from_string(self):
        """Test creating board from string."""
        puzzle_str = "0" * 80 + "9"  # 80 zeros and a 9 at the end
        board = SudokuBoard.from_string(puzzle_str)
        assert board.get(8, 8) == 9
    
    def test_to_string(self):
        """Test converting board to string."""
        board = SudokuBoard()
        board.set(0, 0, 5)
        s = board.to_string()
        assert len(s) == 81
        assert s[0] == '5'
    
    def test_copy(self):
        """Test board copy."""
        board = SudokuBoard()
        board.set(4, 4, 7)
        copy = board.copy()
        
        assert copy.get(4, 4) == 7
        
        # Modify copy, original should be unchanged
        copy.set(4, 4, 8)
        assert board.get(4, 4) == 7


class TestValidator:
    """Tests for validation utilities."""
    
    def test_is_valid_placement(self):
        """Test placement validation."""
        board = SudokuBoard()
        board.set(0, 0, 5)
        
        # Can't place 5 in same row
        assert not is_valid_placement(board, 0, 5, 5)
        
        # Can't place 5 in same column
        assert not is_valid_placement(board, 5, 0, 5)
        
        # Can't place 5 in same box
        assert not is_valid_placement(board, 1, 1, 5)
        
        # Can place different value
        assert is_valid_placement(board, 0, 5, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
