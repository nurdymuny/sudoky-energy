"""Constraint Programming (CP) solver for Sudoku using Arc Consistency (AC-3)."""

from __future__ import annotations
import collections
from typing import Optional, List, Dict, Set, Tuple

from .base_solver import BaseSolver
from ..core.board import SudokuBoard


class CPSolver(BaseSolver):
    """
    Sudoku solver using Constraint Programming (CP) techniques.
    
    This solver uses:
    - Domain tracking: Each cell has a set of possible values.
    - AC-3 Algorithm: Enforces arc consistency to prune domains.
    - Logical Strategies: Naked Singles, Hidden Singles.
    - Hybrid Search: If logic stalls, falls back to backtracking on remaining domains.
    """
    
    name = "Constraint Programming"
    
    def __init__(self, use_backup_backtracking: bool = True):
        super().__init__()
        self.use_backup_backtracking = use_backup_backtracking
        self.domains: Dict[Tuple[int, int], Set[int]] = {}

    def _solve(self, board: SudokuBoard) -> Optional[SudokuBoard]:
        """Solve Sudoku using CP and propagation."""
        self.stats.iterations = 0
        
        # Initialize domains for all cells
        self.domains = self._initialize_domains(board)
        
        # 1. Initial Propagation (AC-3)
        if not self._enforce_arc_consistency(board):
            return None  # Inconsistent puzzle
            
        # 2. Main Logic Loop: Apply strategies until no more changes
        changed = True
        while changed:
            changed = False
            # Apply Naked Singles (cells with domain size 1)
            changed |= self._apply_naked_singles(board)
            # Re-propagate if changes were made
            if changed:
                if not self._enforce_arc_consistency(board):
                    return None
            
            # Apply Hidden Singles (number can only go in one place in row/col/box)
            changed |= self._apply_hidden_singles(board)
            
            # Re-propagate if changes were made
            if changed:
                if not self._enforce_arc_consistency(board):
                    return None
            
        # 3. Check if solved
        if board.is_solved():
            return board
            
        # 4. Backup: Backtracking with MRV (Fail-First) if logic alone didn't finish
        if self.use_backup_backtracking:
            return self._backtrack_on_domains(board)
            
        return None

    def _initialize_domains(self, board: SudokuBoard) -> Dict[Tuple[int, int], Set[int]]:
        """Initialize each cell's domain with 1-9 or its fixed value."""
        domains = {}
        for r in range(board.size):
            for c in range(board.size):
                val = board.get(r, c)
                if val != 0:
                    domains[(r, c)] = {val}
                else:
                    domains[(r, c)] = set(range(1, board.size + 1))
        return domains

    def _enforce_arc_consistency(self, board: SudokuBoard) -> bool:
        """
        Implementation of AC-3 algorithm.
        Prunes domains based on constraints between cells.
        """
        # Queue of all arcs (cell1, cell2) that are peers
        queue = collections.deque()
        for r in range(board.size):
            for c in range(board.size):
                for peer in board.get_peers(r, c):
                    queue.append(((r, c), peer))
                    
        while queue:
            (xi, xj) = queue.popleft()
            if self._revise(xi, xj):
                if not self.domains[xi]:
                    return False  # Domain empty -> Inconsistent
                
                # Domain of xi shrunk, add all neighbors except xj to queue
                for xk in board.get_peers(*xi):
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def _revise(self, xi: Tuple[int, int], xj: Tuple[int, int]) -> bool:
        """
        Revised function for AC-3.
        Returns True if domain of xi was modified.
        """
        revised = False
        # If xj has only one possible value, xi cannot have that value
        if len(self.domains[xj]) == 1:
            val = next(iter(self.domains[xj]))
            if val in self.domains[xi]:
                self.domains[xi].remove(val)
                revised = True
        return revised

    def _apply_naked_singles(self, board: SudokuBoard) -> bool:
        """Fill in cells that have only one possible value in their domain."""
        changed = False
        for (r, c), domain in self.domains.items():
            if board.get(r, c) == 0 and len(domain) == 1:
                val = next(iter(domain))
                board.set(r, c, val)
                changed = True
        return changed

    def _apply_hidden_singles(self, board: SudokuBoard) -> bool:
        """
        Check rows, columns, and boxes for values that can only go in one cell.
        """
        changed = False
        size = board.size
        
        # Check rows
        for r in range(size):
            changed |= self._find_hidden_in_unit(board, [(r, c) for c in range(size)])
            
        # Check columns
        for c in range(size):
            changed |= self._find_hidden_in_unit(board, [(r, c) for r in range(size)])
            
        # Check boxes
        box_size = board.box_size
        for box_r in range(0, size, box_size):
            for box_c in range(0, size, box_size):
                unit = []
                for i in range(box_size):
                    for j in range(box_size):
                        unit.append((box_r + i, box_c + j))
                changed |= self._find_hidden_in_unit(board, unit)
                
        return changed

    def _find_hidden_in_unit(self, board: SudokuBoard, unit: List[Tuple[int, int]]) -> bool:
        """Helper to find hidden singles in a given unit (row/col/box)."""
        changed = False
        # Count occurrences of each value in domains of empty cells in this unit
        counts = collections.defaultdict(list)
        for r, c in unit:
            if board.get(r, c) == 0:
                for val in self.domains[(r, c)]:
                    counts[val].append((r, c))
                    
        # Find values already placed in this unit to avoid duplicates
        placed_values = set()
        for r, c in unit:
            val = board.get(r, c)
            if val != 0:
                placed_values.add(val)

        # If a value appears only once, it's a hidden single
        for val, positions in counts.items():
            if len(positions) == 1:
                # CRITICAL FIX: If value is already in the unit, don't set it again
                if val in placed_values:
                    continue
                    
                r, c = positions[0]
                board.set(r, c, val)
                self.domains[(r, c)] = {val}
                changed = True
        return changed

    def _backtrack_on_domains(self, board: SudokuBoard) -> Optional[SudokuBoard]:
        """Hybrid backtracking that uses pruned domains and MRV heuristic."""
        self.stats.iterations += 1
        
        if board.is_solved():
            return board
            
        # Select empty cell with smallest domain (MRV)
        empty_cells = [(r, c) for r in range(board.size) for c in range(board.size) if board.get(r, c) == 0]
        if not empty_cells:
            return board if board.is_solved() else None
            
        r, c = min(empty_cells, key=lambda pos: len(self.domains[pos]))
        
        # Try values from domain
        original_domains = {pos: domain.copy() for pos, domain in self.domains.items()}
        
        for val in sorted(list(self.domains[(r, c)])):
            if board.is_valid_move(r, c, val):
                # Apply move
                board.set(r, c, val)
                self.domains[(r, c)] = {val}
                
                # Propagate
                if self._enforce_arc_consistency(board):
                    result = self._backtrack_on_domains(board)
                    if result:
                        return result
                
                # Backtrack
                board.set(r, c, 0)
                self.domains = {pos: domain.copy() for pos, domain in original_domains.items()}
                
        return None
