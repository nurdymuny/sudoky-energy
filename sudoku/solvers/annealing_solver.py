"""Simulated Annealing solver for Sudoku (Energy-Based approach)."""

from __future__ import annotations
import random
import math
from typing import Optional, List, Tuple

from .base_solver import BaseSolver
from ..core.board import SudokuBoard


class AnnealingSolver(BaseSolver):
    """
    Simulated Annealing solver for Sudoku.
    
    Uses an energy-based approach where:
    - Energy = number of constraint violations
    - Goal is to minimize energy to 0
    
    The algorithm starts with a high temperature allowing exploration,
    then gradually cools to focus on exploitation.
    
    This is a probabilistic approach and may not always find a solution,
    but can be effective for hard puzzles where deterministic methods struggle.
    """
    
    name = "Simulated Annealing"
    
    def __init__(
        self,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.9999,
        min_temp: float = 0.0001,
        max_iterations: int = 500000,
        restarts: int = 5
    ):
        """
        Initialize the annealing solver.
        
        Args:
            initial_temp: Starting temperature.
            cooling_rate: Temperature multiplier each iteration (e.g., 0.9999).
            min_temp: Minimum temperature before stopping.
            max_iterations: Maximum iterations per restart.
            restarts: Number of restarts if solution not found.
        """
        super().__init__()
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.restarts = restarts
    
    def _solve(self, board: SudokuBoard) -> Optional[SudokuBoard]:
        """Solve using simulated annealing."""
        self.stats.iterations = 0
        self.stats.extra["restarts_used"] = 0
        
        # Store original fixed cells (clues)
        fixed_cells = set()
        for i in range(board.size):
            for j in range(board.size):
                if not board.is_empty(i, j):
                    fixed_cells.add((i, j))
        
        best_solution = None
        best_energy = float('inf')
        
        for restart in range(self.restarts):
            # Initialize with random valid assignment
            work_board = self._initialize_board(board, fixed_cells)
            
            temperature = self.initial_temp
            current_energy = self._calculate_energy(work_board)
            
            if current_energy == 0:
                return work_board
            
            for _ in range(self.max_iterations):
                self.stats.iterations += 1
                
                if temperature < self.min_temp:
                    break
                
                # Generate neighbor by swapping two cells in same box
                neighbor, swap = self._get_neighbor(work_board, fixed_cells)
                
                if neighbor is None:
                    continue
                
                new_energy = self._calculate_energy(neighbor)
                
                # Accept or reject based on energy and temperature
                delta = new_energy - current_energy
                
                if delta < 0:
                    # Better solution - always accept
                    work_board = neighbor
                    current_energy = new_energy
                elif temperature > 0:
                    # Worse solution - accept with probability
                    accept_prob = math.exp(-delta / temperature)
                    if random.random() < accept_prob:
                        work_board = neighbor
                        current_energy = new_energy
                
                # Cool down
                temperature *= self.cooling_rate
                
                # Check if solved
                if current_energy == 0:
                    return work_board
                
                # Track best solution
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = work_board.copy()
            
            self.stats.extra["restarts_used"] = restart + 1
            
            # If found solution, return it
            if current_energy == 0:
                return work_board
        
        # Return best found (may not be a valid solution)
        if best_solution is not None and best_solution.is_solved():
            return best_solution
        
        return None
    
    def _initialize_board(
        self, 
        board: SudokuBoard, 
        fixed_cells: set
    ) -> SudokuBoard:
        """
        Initialize the board with a random assignment.
        
        Each box gets digits 1-9 randomly distributed among empty cells.
        This ensures no conflicts within boxes (only row/col conflicts).
        """
        work_board = board.copy()
        size = board.size
        box_size = board.box_size
        
        # For each box, fill empty cells with remaining digits
        for box_row in range(0, size, box_size):
            for box_col in range(0, size, box_size):
                # Find what values are already in this box
                used_values = set()
                empty_in_box = []
                
                for i in range(box_size):
                    for j in range(box_size):
                        r, c = box_row + i, box_col + j
                        if (r, c) in fixed_cells:
                            used_values.add(board.get(r, c))
                        else:
                            empty_in_box.append((r, c))
                
                # Get remaining values and shuffle
                remaining = list(set(range(1, size + 1)) - used_values)
                random.shuffle(remaining)
                
                # Assign to empty cells
                for (r, c), val in zip(empty_in_box, remaining):
                    work_board.set(r, c, val)
        
        return work_board
    
    def _get_neighbor(
        self, 
        board: SudokuBoard, 
        fixed_cells: set
    ) -> Tuple[Optional[SudokuBoard], Optional[Tuple]]:
        """
        Generate a neighbor state by swapping two non-fixed cells in the same box.
        
        Returns:
            Tuple of (new_board, swap_info) or (None, None) if no valid swap.
        """
        size = board.size
        box_size = board.box_size
        
        # Pick a random box
        box_row = random.randint(0, box_size - 1) * box_size
        box_col = random.randint(0, box_size - 1) * box_size
        
        # Find non-fixed cells in this box
        non_fixed = []
        for i in range(box_size):
            for j in range(box_size):
                r, c = box_row + i, box_col + j
                if (r, c) not in fixed_cells:
                    non_fixed.append((r, c))
        
        if len(non_fixed) < 2:
            return None, None
        
        # Pick two random cells to swap
        idx1, idx2 = random.sample(range(len(non_fixed)), 2)
        cell1, cell2 = non_fixed[idx1], non_fixed[idx2]
        
        # Create new board with swapped values
        new_board = board.copy()
        val1, val2 = board.get(*cell1), board.get(*cell2)
        new_board.set(*cell1, val2)
        new_board.set(*cell2, val1)
        
        return new_board, (cell1, cell2)
    
    def _calculate_energy(self, board: SudokuBoard) -> int:
        """
        Calculate the energy (number of constraint violations).
        
        Counts duplicate values in rows and columns.
        (Box constraints are satisfied by construction.)
        """
        size = board.size
        energy = 0
        
        # Count row violations
        for r in range(size):
            values = [board.get(r, c) for c in range(size)]
            # Count duplicates
            seen = set()
            for v in values:
                if v in seen:
                    energy += 1
                seen.add(v)
        
        # Count column violations
        for c in range(size):
            values = [board.get(r, c) for r in range(size)]
            seen = set()
            for v in values:
                if v in seen:
                    energy += 1
                seen.add(v)
        
        return energy
