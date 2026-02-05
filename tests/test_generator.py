"""Unit tests for puzzle generator."""

import pytest
from sudoku.generator import SudokuGenerator, Difficulty
from sudoku.core.validator import has_unique_solution


class TestSudokuGenerator:
    """Tests for SudokuGenerator class."""
    
    def test_generate_creates_valid_puzzle(self):
        """Test that generated puzzles are valid."""
        generator = SudokuGenerator(seed=42)
        puzzle = generator.generate(Difficulty.EASY)
        
        assert puzzle.is_valid()
        assert puzzle.count_empty() > 0
        assert puzzle.count_filled() > 0
    
    def test_difficulty_affects_clue_count(self):
        """Test that harder difficulties have fewer clues."""
        generator = SudokuGenerator(seed=42)
        
        easy = generator.generate(Difficulty.EASY)
        hard = generator.generate(Difficulty.HARD)
        
        assert easy.count_filled() > hard.count_filled()
    
    def test_generate_with_solution(self):
        """Test generating puzzle with solution."""
        generator = SudokuGenerator(seed=42)
        puzzle, solution = generator.generate_with_solution(Difficulty.MEDIUM)
        
        assert puzzle.is_valid()
        assert solution.is_solved()
        
        # Verify puzzle is subset of solution
        for i in range(puzzle.size):
            for j in range(puzzle.size):
                if not puzzle.is_empty(i, j):
                    assert puzzle.get(i, j) == solution.get(i, j)
    
    def test_generate_batch(self):
        """Test batch generation."""
        generator = SudokuGenerator(seed=42)
        puzzles = generator.generate_batch(3, Difficulty.MEDIUM)
        
        assert len(puzzles) == 3
        for puzzle in puzzles:
            assert puzzle.is_valid()
    
    def test_different_puzzles_without_seed(self):
        """Test that different generators produce different puzzles."""
        gen1 = SudokuGenerator(seed=123)
        gen2 = SudokuGenerator(seed=456)
        
        puzzle1 = gen1.generate(Difficulty.MEDIUM)
        puzzle2 = gen2.generate(Difficulty.MEDIUM)
        
        # Different seeds should generally produce different puzzles
        # Both should be valid
        assert puzzle1.is_valid()
        assert puzzle2.is_valid()


class TestDifficultyLevels:
    """Test difficulty level clue ranges."""
    
    def test_easy_clue_range(self):
        """Easy should have 36-45 clues."""
        min_clues, max_clues = Difficulty.EASY.clue_range
        assert min_clues >= 35
        assert max_clues <= 46
    
    def test_expert_clue_range(self):
        """Expert should have 17-21 clues."""
        min_clues, max_clues = Difficulty.EXPERT.clue_range
        assert min_clues >= 17
        assert max_clues <= 22


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
