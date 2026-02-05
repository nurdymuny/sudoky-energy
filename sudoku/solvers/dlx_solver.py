"""Dancing Links (DLX) solver using Knuth's Algorithm X for Exact Cover."""

from __future__ import annotations
from typing import Optional, List

from .base_solver import BaseSolver
from ..core.board import SudokuBoard


class DLXNode:
    """A node in the Dancing Links structure."""
    __slots__ = ['left', 'right', 'up', 'down', 'column', 'row_id']
    
    def __init__(self, row_id: int = -1):
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column = None
        self.row_id = row_id


class ColumnNode(DLXNode):
    """Column header node with size counter."""
    __slots__ = ['size', 'col_id']
    
    def __init__(self, col_id: int = -1):
        super().__init__()
        self.size = 0
        self.col_id = col_id
        self.column = self


class DLXSolver(BaseSolver):
    """
    Dancing Links solver using Knuth's Algorithm X for Exact Cover.
    
    Sudoku can be formulated as an exact cover problem:
    - Each cell must have exactly one value (81 constraints)
    - Each row must have each digit exactly once (81 constraints)
    - Each column must have each digit exactly once (81 constraints)
    - Each box must have each digit exactly once (81 constraints)
    
    Total: 324 constraints, 729 possibilities (81 cells × 9 values)
    """
    
    name = "Dancing Links (DLX)"
    
    def __init__(self):
        super().__init__()
        self._header = None
        self._columns = []
        self._solution_rows = []
    
    def _solve(self, board: SudokuBoard) -> Optional[SudokuBoard]:
        """Solve using Dancing Links."""
        self.stats.iterations = 0
        self.stats.nodes_explored = 0
        self.stats.backtracks = 0
        
        # Build the DLX matrix
        self._build_matrix(board)
        
        # Solve using Algorithm X
        self._solution_rows = []
        if self._algorithm_x(0):
            # Decode solution
            return self._decode_solution(board)
        
        return None
    
    def _build_matrix(self, board: SudokuBoard) -> None:
        """Build the exact cover matrix for the given Sudoku puzzle."""
        size = board.size
        box_size = board.box_size
        
        # Number of constraints: 324 for 9x9
        num_constraints = 4 * size * size
        
        # Create header
        self._header = ColumnNode()
        
        # Create column headers
        self._columns = []
        prev = self._header
        for i in range(num_constraints):
            col = ColumnNode(col_id=i)
            
            # Link horizontally
            col.left = prev
            col.right = self._header
            prev.right = col
            self._header.left = col
            
            prev = col
            self._columns.append(col)
        
        # Create rows for each possibility
        row_id = 0
        for r in range(size):
            for c in range(size):
                for d in range(1, size + 1):
                    # Skip if cell is already filled with different value
                    if not board.is_empty(r, c):
                        if board.get(r, c) != d:
                            row_id += 1
                            continue
                    
                    # Skip if value violates existing constraints
                    if board.is_empty(r, c):
                        if d not in board.get_candidates(r, c):
                            row_id += 1
                            continue
                    
                    # This is a valid possibility - add row
                    box_idx = (r // box_size) * box_size + (c // box_size)
                    
                    # Calculate constraint column indices
                    cell_col = r * size + c
                    row_col = size * size + r * size + (d - 1)
                    col_col = 2 * size * size + c * size + (d - 1)
                    box_col = 3 * size * size + box_idx * size + (d - 1)
                    
                    # Create nodes for this row
                    constraint_indices = [cell_col, row_col, col_col, box_col]
                    nodes = []
                    
                    for col_idx in constraint_indices:
                        node = DLXNode(row_id=row_id)
                        col = self._columns[col_idx]
                        node.column = col
                        
                        # Link vertically (insert above column header)
                        node.up = col.up
                        node.down = col
                        col.up.down = node
                        col.up = node
                        col.size += 1
                        
                        nodes.append(node)
                    
                    # Link horizontally (circular)
                    for i in range(len(nodes)):
                        nodes[i].right = nodes[(i + 1) % len(nodes)]
                        nodes[i].left = nodes[(i - 1) % len(nodes)]
                    
                    row_id += 1
    
    def _algorithm_x(self, depth: int) -> bool:
        """Knuth's Algorithm X with Dancing Links (iterative with stack)."""
        # Use explicit stack to avoid recursion limit
        # Stack entries: (phase, col, row_node)
        # phase 0: enter, phase 1: next row
        
        stack = []
        
        while True:
            self.stats.iterations += 1
            
            # Check if matrix is empty (success)
            if self._header.right is self._header:
                return True
            
            # Choose column with minimum size (MRV heuristic)
            chosen_col = None
            min_size = float('inf')
            
            c = self._header.right
            while c is not self._header:
                if c.size < min_size:
                    min_size = c.size
                    chosen_col = c
                c = c.right
            
            if min_size == 0:
                # Dead end - backtrack
                self.stats.backtracks += 1
                
                while stack:
                    phase, col, row_node = stack.pop()
                    
                    # Uncover from this row
                    j = row_node.left
                    while j is not row_node:
                        self._uncover(j.column)
                        j = j.left
                    self._solution_rows.pop()
                    
                    # Try next row
                    next_row = row_node.down
                    if next_row is not col:
                        # There's another row to try
                        self._solution_rows.append(next_row.row_id)
                        j = next_row.right
                        while j is not next_row:
                            self._cover(j.column)
                            j = j.right
                        stack.append((1, col, next_row))
                        break
                    else:
                        # No more rows, uncover this column
                        self._uncover(col)
                else:
                    # Stack empty, no solution
                    return False
                continue
            
            # Cover chosen column
            self._cover(chosen_col)
            self.stats.nodes_explored += 1
            
            # Try first row in this column
            row = chosen_col.down
            if row is chosen_col:
                # No rows in this column, backtrack
                self._uncover(chosen_col)
                
                while stack:
                    phase, col, row_node = stack.pop()
                    
                    j = row_node.left
                    while j is not row_node:
                        self._uncover(j.column)
                        j = j.left
                    self._solution_rows.pop()
                    
                    next_row = row_node.down
                    if next_row is not col:
                        self._solution_rows.append(next_row.row_id)
                        j = next_row.right
                        while j is not next_row:
                            self._cover(j.column)
                            j = j.right
                        stack.append((1, col, next_row))
                        break
                    else:
                        self._uncover(col)
                else:
                    return False
                continue
            
            # Add this row to solution
            self._solution_rows.append(row.row_id)
            
            # Cover all other columns satisfied by this row
            j = row.right
            while j is not row:
                self._cover(j.column)
                j = j.right
            
            # Push to stack
            stack.append((0, chosen_col, row))
    
    def _cover(self, col: ColumnNode) -> None:
        """Cover a column (remove it and all rows using it)."""
        col.right.left = col.left
        col.left.right = col.right
        
        row = col.down
        while row is not col:
            j = row.right
            while j is not row:
                j.down.up = j.up
                j.up.down = j.down
                j.column.size -= 1
                j = j.right
            row = row.down
    
    def _uncover(self, col: ColumnNode) -> None:
        """Uncover a column (restore it and all rows using it)."""
        row = col.up
        while row is not col:
            j = row.left
            while j is not row:
                j.column.size += 1
                j.down.up = j
                j.up.down = j
                j = j.left
            row = row.up
        
        col.right.left = col
        col.left.right = col
    
    def _decode_solution(self, original_board: SudokuBoard) -> SudokuBoard:
        """Decode the solution rows back to a Sudoku board."""
        size = original_board.size
        solution = original_board.copy()
        
        for row_id in self._solution_rows:
            # Decode: row_id = r * size * size + c * size + (d - 1)
            d = (row_id % size) + 1
            c = (row_id // size) % size
            r = row_id // (size * size)
            
            solution.set(r, c, d)
        
        return solution
