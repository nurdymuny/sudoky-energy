"""Depth-First Search solver with backtracking and constraint propagation."""

from __future__ import annotations
from typing import Optional, List, Tuple, Set

from .base_solver import BaseSolver
from ..core.board import SudokuBoard


class DFSSolver(BaseSolver):
    """
    Depth-First Search solver using recursive backtracking.
    
    Features:
    - Minimum Remaining Values (MRV) heuristic for cell selection
    - Constraint propagation (naked singles, hidden singles)
    - Backtracking counter for performance analysis
    """
    
    name = "DFS+Backtracking"
    
    def __init__(self, use_constraint_propagation: bool = True):
        """
        Initialize the DFS solver.
        
        Args:
            use_constraint_propagation: If True, apply constraint propagation
                                       before backtracking.
        """
        super().__init__()
        self.use_constraint_propagation = use_constraint_propagation
    
    def _solve(self, board: SudokuBoard) -> Optional[SudokuBoard]:
        """Solve using DFS with backtracking."""
        self.stats.iterations = 0
        self.stats.backtracks = 0
        self.stats.nodes_explored = 0
        
        # Apply initial constraint propagation
        if self.use_constraint_propagation:
            if not self._propagate_constraints(board):
                return None
        
        if self._backtrack(board):
            return board
        return None
    
    def _backtrack(self, board: SudokuBoard) -> bool:
        """
        Recursive backtracking algorithm.
        
        Returns True if solution found, False otherwise.
        """
        self.stats.iterations += 1
        
        # Find the best empty cell using MRV heuristic
        cell = self._select_unassigned_variable(board)
        if cell is None:
            # No empty cells - solution found!
            return True
        
        row, col = cell
        candidates = board.get_candidates(row, col)
        self.stats.nodes_explored += 1
        
        # No valid candidates - need to backtrack
        if not candidates:
            self.stats.backtracks += 1
            return False
        
        # Try each candidate value
        for value in sorted(candidates):  # Sort for deterministic behavior
            board.set(row, col, value)
            
            # Apply constraint propagation after each assignment
            if self.use_constraint_propagation:
                propagated_board = board.copy()
                if self._propagate_constraints(propagated_board):
                    # Copy propagated values back
                    for i in range(board.size):
                        for j in range(board.size):
                            if board.is_empty(i, j) and not propagated_board.is_empty(i, j):
                                board.set(i, j, propagated_board.get(i, j))
                    
                    if self._backtrack(board):
                        return True
                    
                    # Undo propagated values
                    for i in range(board.size):
                        for j in range(board.size):
                            if not propagated_board.is_empty(i, j) and cell != (i, j):
                                orig_val = propagated_board.get(i, j)
                                if board.get(i, j) == orig_val:
                                    # Only clear if it was set by propagation
                                    pass  # Keep it for now, clear later
            else:
                if self._backtrack(board):
                    return True
            
            board.clear(row, col)
            self.stats.backtracks += 1
        
        return False
    
    def _select_unassigned_variable(self, board: SudokuBoard) -> Optional[Tuple[int, int]]:
        """
        Select the next empty cell using MRV (Minimum Remaining Values) heuristic.
        
        Picks the cell with the fewest remaining valid candidates.
        This helps prune the search tree more effectively.
        """
        empty_cells = board.get_empty_cells()
        if not empty_cells:
            return None
        
        best_cell = None
        min_candidates = board.size + 1
        
        for cell in empty_cells:
            candidates = board.get_candidates(cell[0], cell[1])
            num_candidates = len(candidates)
            
            if num_candidates == 0:
                # No valid candidates at all - fail fast
                return cell
            
            if num_candidates < min_candidates:
                min_candidates = num_candidates
                best_cell = cell
                
                # Optimization: if only one candidate, use it immediately
                if min_candidates == 1:
                    break
        
        return best_cell
    
    def _propagate_constraints(self, board: SudokuBoard) -> bool:
        """
        Apply constraint propagation techniques.
        
        Currently implements:
        - Naked singles: If a cell has only one candidate, fill it in.
        
        Returns:
            True if propagation successful (no conflicts), False otherwise.
        """
        changed = True
        while changed:
            changed = False
            empty_cells = board.get_empty_cells()
            
            for row, col in empty_cells:
                candidates = board.get_candidates(row, col)
                
                if len(candidates) == 0:
                    # Conflict detected
                    return False
                
                if len(candidates) == 1:
                    # Naked single - fill it in
                    board.set(row, col, next(iter(candidates)))
                    changed = True
            
            # Also apply hidden singles
            changed = self._find_hidden_singles(board) or changed
        
        return True
    
    def _find_hidden_singles(self, board: SudokuBoard) -> bool:
        """
        Find and fill hidden singles.
        
        A hidden single is when a value can only go in one cell within a
        row, column, or box.
        
        Returns:
            True if any hidden singles were found and filled.
        """
        changed = False
        
        # Check each row
        for row in range(board.size):
            changed = self._find_hidden_singles_in_unit(
                board, [(row, col) for col in range(board.size)]
            ) or changed
        
        # Check each column
        for col in range(board.size):
            changed = self._find_hidden_singles_in_unit(
                board, [(row, col) for row in range(board.size)]
            ) or changed
        
        # Check each box
        for box_row in range(0, board.size, board.box_size):
            for box_col in range(0, board.size, board.box_size):
                cells = []
                for i in range(board.box_size):
                    for j in range(board.box_size):
                        cells.append((box_row + i, box_col + j))
                changed = self._find_hidden_singles_in_unit(board, cells) or changed
        
        return changed
    
    def _find_hidden_singles_in_unit(
        self, 
        board: SudokuBoard, 
        cells: List[Tuple[int, int]]
    ) -> bool:
        """
        Find hidden singles within a single unit (row, column, or box).
        """
        changed = False
        
        # For each possible value
        for value in range(1, board.size + 1):
            # Find cells where this value can go
            possible_cells = []
            for row, col in cells:
                if board.is_empty(row, col) and value in board.get_candidates(row, col):
                    possible_cells.append((row, col))
            
            # If only one cell can hold this value, fill it in
            if len(possible_cells) == 1:
                row, col = possible_cells[0]
                if board.is_empty(row, col):  # Double-check
                    board.set(row, col, value)
                    changed = True
        
        return changed
