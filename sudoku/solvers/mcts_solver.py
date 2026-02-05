"""Monte Carlo Tree Search solver for Sudoku."""

from __future__ import annotations
import random
import math
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from .base_solver import BaseSolver
from ..core.board import SudokuBoard


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    board: SudokuBoard
    parent: Optional['MCTSNode'] = None
    children: Dict[Tuple[int, int, int], 'MCTSNode'] = None  # (row, col, value) -> child
    visits: int = 0
    wins: float = 0.0
    untried_moves: List[Tuple[int, int, int]] = None  # (row, col, value)
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.untried_moves is None:
            self.untried_moves = self._get_all_moves()
    
    def _get_all_moves(self) -> List[Tuple[int, int, int]]:
        """Get all possible moves from this state."""
        moves = []
        empty_cells = self.board.get_empty_cells()
        
        # Prioritize cells with fewer candidates
        cell_candidates = []
        for row, col in empty_cells:
            candidates = self.board.get_candidates(row, col)
            if candidates:
                cell_candidates.append((len(candidates), row, col, candidates))
        
        cell_candidates.sort()  # Sort by number of candidates
        
        for _, row, col, candidates in cell_candidates:
            for val in candidates:
                moves.append((row, col, val))
        
        return moves
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (solved or no moves)."""
        return self.board.is_solved() or (
            not self.untried_moves and not self.children
        )
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0
    
    def ucb1(self, exploration: float = 1.41) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration_term
    
    def best_child(self, exploration: float = 1.41) -> 'MCTSNode':
        """Select best child using UCB1."""
        return max(self.children.values(), key=lambda c: c.ucb1(exploration))
    
    def expand(self) -> 'MCTSNode':
        """Expand by trying an untried move."""
        if not self.untried_moves:
            return None
        
        # Pick a random untried move
        move = self.untried_moves.pop(random.randint(0, len(self.untried_moves) - 1))
        row, col, val = move
        
        # Create new board state
        new_board = self.board.copy()
        new_board.set(row, col, val)
        
        # Create child node
        child = MCTSNode(board=new_board, parent=self)
        self.children[move] = child
        
        return child
    
    def update(self, result: float) -> None:
        """Update node statistics with simulation result."""
        self.visits += 1
        self.wins += result


class MCTSSolver(BaseSolver):
    """
    Monte Carlo Tree Search solver for Sudoku.
    
    Uses UCT (Upper Confidence Bound for Trees) for selection
    and random playouts for simulation.
    
    Note: MCTS is not ideal for Sudoku due to the combinatorial nature,
    but is included for comparison purposes. It works better for
    game-playing scenarios with uncertainty.
    """
    
    name = "MCTS"
    
    def __init__(
        self, 
        max_iterations: int = 10000,
        exploration: float = 1.41,
        max_playout_depth: int = 50
    ):
        """
        Initialize MCTS solver.
        
        Args:
            max_iterations: Maximum MCTS iterations.
            exploration: UCB1 exploration parameter (sqrt(2) ≈ 1.41 is common).
            max_playout_depth: Maximum depth for random playouts.
        """
        super().__init__()
        self.max_iterations = max_iterations
        self.exploration = exploration
        self.max_playout_depth = max_playout_depth
    
    def _solve(self, board: SudokuBoard) -> Optional[SudokuBoard]:
        """Solve using MCTS."""
        self.stats.iterations = 0
        self.stats.nodes_explored = 0
        
        # Create root node
        root = MCTSNode(board=board)
        
        # Check if already solved
        if board.is_solved():
            return board
        
        best_solution = None
        
        for iteration in range(self.max_iterations):
            self.stats.iterations += 1
            
            # Selection: traverse tree using UCB1
            node = self._select(root)
            
            # Expansion: expand if not terminal and not fully explored
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
                if node is None:
                    continue
                self.stats.nodes_explored += 1
            
            # Simulation: random playout
            result, solution = self._simulate(node)
            
            if solution is not None and solution.is_solved():
                best_solution = solution
                break
            
            # Backpropagation: update statistics
            self._backpropagate(node, result)
        
        # If we found a solution during search, return it
        if best_solution is not None:
            return best_solution
        
        # Otherwise, try to reconstruct best path
        return self._get_best_solution(root)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCB1."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            if not node.children:
                return node
            node = node.best_child(self.exploration)
        return node
    
    def _simulate(self, node: MCTSNode) -> Tuple[float, Optional[SudokuBoard]]:
        """
        Perform a random playout from the node.
        
        Returns:
            Tuple of (reward, solution if found).
            Reward is 1.0 for solved, 0.0-1.0 based on progress.
        """
        board = node.board.copy()
        
        # Quick check - already solved?
        if board.is_solved():
            return 1.0, board
        
        # Random playout
        for _ in range(self.max_playout_depth):
            empty_cells = board.get_empty_cells()
            if not empty_cells:
                break
            
            # Find cell with fewest candidates (greedy heuristic)
            best_cell = None
            min_candidates = board.size + 1
            candidates_list = []
            
            for cell in empty_cells:
                candidates = board.get_candidates(cell[0], cell[1])
                if not candidates:
                    # Dead end
                    filled_ratio = board.count_filled() / (board.size * board.size)
                    return filled_ratio * 0.5, None
                
                if len(candidates) < min_candidates:
                    min_candidates = len(candidates)
                    best_cell = cell
                    candidates_list = list(candidates)
            
            if best_cell is None:
                break
            
            # Make random choice
            row, col = best_cell
            value = random.choice(candidates_list)
            board.set(row, col, value)
            
            if board.is_solved():
                return 1.0, board
        
        # Calculate reward based on how complete the board is
        filled_ratio = board.count_filled() / (board.size * board.size)
        return filled_ratio, None
    
    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """Backpropagate the result up the tree."""
        while node is not None:
            node.update(result)
            node = node.parent
    
    def _get_best_solution(self, root: MCTSNode) -> Optional[SudokuBoard]:
        """
        Try to find a solution by following the best path from root.
        """
        node = root
        
        while node.children:
            # Pick child with highest win rate
            best_child = None
            best_rate = -1
            
            for child in node.children.values():
                if child.visits > 0:
                    rate = child.wins / child.visits
                    if rate > best_rate:
                        best_rate = rate
                        best_child = child
            
            if best_child is None:
                break
            
            node = best_child
            
            if node.board.is_solved():
                return node.board
        
        # Couldn't find complete solution through best path
        return None
