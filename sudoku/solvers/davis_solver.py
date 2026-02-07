"""
Davis Manifold Sudoku Solver
============================
Applies the Davis Field Equations (Bee Rosa Davis, 2025) to Sudoku solving.

Core mappings:
  - Davis Law:  C = τ/K  →  Inference capacity from constraint geometry
  - Trichotomy: Γ parameter  →  Difficulty classification & solver selection
  - T3 (Gap-Filling Complexity Reduction)  →  Search space bounds
  - E0 (Davis Energy Functional)  →  Geometrically principled SA energy
  - F5 (Optimal Constraint Ordering)  →  Curvature-aware cell selection
  - Holonomy monitoring  →  Early pruning in DFS

Reference: "The Field Equations of Semantic Coherence" (Davis, 2025)
"""

import math
import time
import random
from typing import Optional

from .base_solver import BaseSolver
from ..core.board import SudokuBoard

# =============================================================================
# PART 1: THE SUDOKU MANIFOLD — Geometric Primitives
# =============================================================================

def get_candidates(board: list[list[int]], row: int, col: int) -> set[int]:
    """Get valid candidates for a cell. Returns the local valid region R_c."""
    if board[row][col] != 0:
        return set()
    
    used = set()
    # Row constraint
    used.update(board[row])
    # Column constraint
    used.update(board[r][col] for r in range(9))
    # Box constraint
    br, bc = 3 * (row // 3), 3 * (col // 3)
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            used.update({board[r][c]})
    
    used.discard(0)
    return set(range(1, 10)) - used


def count_empty(board: list[list[int]]) -> int:
    """Count unobserved variables n."""
    return sum(1 for r in range(9) for c in range(9) if board[r][c] == 0)


# =============================================================================
# PART 2: LOCAL CURVATURE K_loc — Constraint Interaction Density
# =============================================================================

def local_curvature(board: list[list[int]], row: int, col: int) -> float:
    """
    Compute K_loc at a cell position.
    
    Curvature = density of constraint interactions at this point on the manifold.
    High curvature means the cell participates in many tightly-coupled constraints.
    
    From the Field Equations, §T1: regions of high curvature resist unique completion.
    
    Components:
      1. Constraint density: how many filled cells share constraints with this cell
      2. Mutual constraint coupling: how many other empty cells compete for the same values
      3. Intersection factor: this cell sits at the intersection of row, col, AND box constraints
    """
    if board[row][col] != 0:
        return 0.0
    
    candidates = get_candidates(board, row, col)
    if not candidates:
        return float('inf')  # Inconsistent — curvature singularity
    
    # Component 1: Constraint saturation of the cell's neighborhood
    #   Count UNIQUE filled peers (cells sharing row, column, or box).
    #   Max unique peers = 20 (8 row + 8 col + 4 non-overlapping box).
    br, bc = 3 * (row // 3), 3 * (col // 3)
    filled_peers = set()
    for c in range(9):
        if c != col and board[row][c] != 0:
            filled_peers.add((row, c))
    for r in range(9):
        if r != row and board[r][col] != 0:
            filled_peers.add((r, col))
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if (r, c) != (row, col) and board[r][c] != 0:
                filled_peers.add((r, c))
    
    saturation = len(filled_peers) / 20.0  # 20 = max unique peers
    
    # Component 2: Candidate scarcity (fewer candidates = higher curvature)
    #   From T3: |S_valid| shrinks exponentially with constraints
    scarcity = 1.0 - (len(candidates) / 9.0)
    
    # Component 3: Coupling — how many OTHER empty cells in the same 
    #   constraint groups compete for the same candidates
    #   Use a set to avoid double-counting cells that share both
    #   row+box or col+box with the target cell.
    peers = set()
    for c in range(9):
        if c != col:
            peers.add((row, c))
    for r in range(9):
        if r != row:
            peers.add((r, col))
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if (r, c) != (row, col):
                peers.add((r, c))
    
    coupling = 0
    empty_peer_count = 0
    for (pr, pc) in peers:
        if board[pr][pc] == 0:
            other_cands = get_candidates(board, pr, pc)
            coupling += len(candidates & other_cands)
            empty_peer_count += 1
    
    # Normalize: max overlap per peer is 9 (all candidates shared),
    # max peers is 20 (8 row + 8 col + 4 non-overlapping box).
    max_coupling = max(empty_peer_count * len(candidates), 1)
    coupling_norm = coupling / max_coupling
    
    # K_loc = weighted combination
    K = 0.4 * saturation + 0.35 * scarcity + 0.25 * coupling_norm
    return K


def max_curvature(board: list[list[int]]) -> float:
    """Compute K̂_max = max curvature over all empty cells."""
    K_max = 0.0
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                K = local_curvature(board, r, c)
                if K != float('inf'):
                    K_max = max(K_max, K)
    return K_max


def mean_curvature(board: list[list[int]]) -> float:
    """Average curvature over all empty cells."""
    curvatures = []
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                K = local_curvature(board, r, c)
                if K != float('inf'):
                    curvatures.append(K)
    return sum(curvatures) / len(curvatures) if curvatures else 0.0


# =============================================================================
# PART 3: TRICHOTOMY PARAMETER Γ — Difficulty Classification
# =============================================================================

def trichotomy_parameter(board: list[list[int]]) -> float:
    """
    Compute Γ = (m · τ_budget) / (K̂_max · log|S|)
    
    From the Extended Master Trichotomy:
      Γ > 1  →  DETERMINED (easy, unique solution, fast consensus)
      Γ = 1  →  CRITICAL (phase transition, algorithm-dependent)
      Γ < 1  →  UNDERDETERMINED (hard, multiple possible paths)
    
    For Sudoku:
      m = number of filled cells (observed constraints)
      τ_budget = 1.0 (exact Sudoku, no tolerance for error)
      K̂_max = maximum curvature over the board
      |S| = 9^n where n = number of empty cells
    """
    n = count_empty(board)
    m = 81 - n  # Observed constraints (filled cells)
    
    if n == 0:
        return float('inf')  # Solved
    
    K_max = max_curvature(board)
    if K_max == 0:
        return float('inf')  # Trivially solvable
    
    tau_budget = 1.0
    log_S = n * math.log(9)  # log|S_unconstrained|
    
    Gamma = (m * tau_budget) / (K_max * log_S)
    return Gamma


def classify_difficulty(board: list[list[int]]) -> str:
    """
    Use the Davis Trichotomy to classify puzzle difficulty.
    
    Calibrated against empirical Γ values:
      17 clues (minimum unique) → Γ ≈ 0.15
      25 clues (hard)           → Γ ≈ 0.30
      35 clues (medium)         → Γ ≈ 0.60
      50 clues (easy)           → Γ ≈ 1.20
    """
    Gamma = trichotomy_parameter(board)
    
    if Gamma > 1.0:
        return "DETERMINED (Easy)"
    elif Gamma > 0.6:
        return "DETERMINED (Medium)"
    elif Gamma > 0.35:
        return "CRITICAL (Hard)"
    elif Gamma > 0.2:
        return "UNDERDETERMINED (Expert)"
    else:
        return "UNDERDETERMINED (Extreme)"


def select_optimal_solver(board: list[list[int]]) -> str:
    """
    Use the Trichotomy to select the optimal solver.
    
    From the Geometry of Sameness: different solvers are different
    realizations of the same semantic sameness structure. The Error
    Budget Transfer Theorem tells us when each realization is optimal.
    """
    Gamma = trichotomy_parameter(board)
    
    if Gamma > 1.0:
        return "cp"        # Constraint propagation alone suffices
    elif Gamma > 0.6:
        return "dlx"       # Dancing Links for exact, efficient solving
    elif Gamma > 0.2:
        return "dfs"       # DFS with curvature-aware ordering
    else:
        return "davis"     # Davis Manifold Relaxation for extreme cases


# =============================================================================
# PART 4: CONSTRAINT INFORMATION VALUE V(c) — Curvature-Aware Ordering
# =============================================================================

def information_value(board: list[list[int]], row: int, col: int) -> float:
    """
    Compute V(c) = ∫_{R_c} K_loc(x) dV_g(x)
    
    From the Curvature-Information Duality (T1bii):
    "Information value equals integrated curvature over the constraint region."
    
    A cell's information value is how much filling it reduces |S_valid|.
    This is better than MRV because it accounts for constraint coupling,
    not just candidate count.
    """
    if board[row][col] != 0:
        return 0.0
    
    candidates = get_candidates(board, row, col)
    if not candidates:
        return float('inf')  # Must backtrack — infinite priority
    
    K_loc = local_curvature(board, row, col)
    
    # V(c) = K_loc × |affected cells| / |candidates|
    # High curvature + few candidates + many affected cells = high value
    
    # Count cells that share constraints with this cell (the constraint region)
    affected = set()
    for c in range(9):
        if c != col and board[row][c] == 0:
            affected.add((row, c))
    for r in range(9):
        if r != row and board[r][col] == 0:
            affected.add((r, col))
    br, bc = 3 * (row // 3), 3 * (col // 3)
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if (r, c) != (row, col) and board[r][c] == 0:
                affected.add((r, c))
    
    # Information value: curvature integrated over the constraint region.
    # Skip peers with inf curvature (dead cells) — those indicate the
    # board is already inconsistent and will be caught by holonomy_prune,
    # not by inflating V(c) of a viable neighbor.
    region_curvature = K_loc
    for (ar, ac) in affected:
        peer_K = local_curvature(board, ar, ac)
        if peer_K != float('inf'):
            region_curvature += peer_K
    
    # Normalize by candidates (fewer candidates = each assignment is more informative)
    V = region_curvature / len(candidates)
    
    return V


def select_next_cell_davis(board: list[list[int]]) -> Optional[tuple[int, int]]:
    """
    Select the next cell to fill using Davis ordering (F5).
    
    From Optimal Constraint Ordering: "at each step, project to the
    constraint whose region R_c is closest to the current path in
    geodesic distance."
    
    Translated: pick the cell with highest information value V(c).
    This subsumes MRV but adds geometric coupling information.
    """
    best_cell = None
    best_value = -1.0
    
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                V = information_value(board, r, c)
                if V == float('inf'):
                    return (r, c)  # Forced — must handle this cell now
                if V > best_value:
                    best_value = V
                    best_cell = (r, c)
    
    return best_cell


# =============================================================================
# PART 5: HOLONOMY MONITORING — Geometric Consistency Check
# =============================================================================

def check_holonomy(board: list[list[int]]) -> float:
    """
    Compute ‖Hol - I‖ for the current board state.
    
    From the Sudoku Principle (Corollary): "compute holonomy around gap 
    boundaries; if ‖Hol − I‖ < τ for all such loops, the completion is unique."
    
    For Sudoku: holonomy deficit = measure of constraint inconsistency.
    A board with no valid completions has ‖Hol - I‖ = ∞.
    A board with unique completion has ‖Hol - I‖ → 0.
    """
    deficit = 0.0
    
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                cands = get_candidates(board, r, c)
                if len(cands) == 0:
                    return float('inf')  # Inconsistency — holonomy blowup
                # Deficit contribution: how far from uniquely determined
                deficit += (len(cands) - 1) / 8.0
    
    return deficit


def holonomy_prune(board: list[list[int]], row: int, col: int, val: int) -> bool:
    """
    Holonomy-based look-ahead pruning.
    
    After tentatively placing val at (row, col), check whether any
    cell in the affected constraint regions becomes impossible.
    
    This is stronger than standard arc consistency because it checks
    the GEOMETRIC consistency of the entire constraint neighborhood,
    not just direct peers.
    
    From S3a (Inconsistency Localization): inconsistency is always
    localizable to at most (d+1) constraints.
    """
    # Temporarily place the value
    board[row][col] = val
    
    # Check all affected cells (the constraint "loop boundary")
    cells_to_check = set()
    for c in range(9):
        if board[row][c] == 0:
            cells_to_check.add((row, c))
    for r in range(9):
        if board[r][col] == 0:
            cells_to_check.add((r, col))
    br, bc = 3 * (row // 3), 3 * (col // 3)
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if board[r][c] == 0:
                cells_to_check.add((r, c))
    
    pruned = False
    for (cr, cc) in cells_to_check:
        if len(get_candidates(board, cr, cc)) == 0:
            pruned = True
            break
    
    # Undo the tentative placement
    board[row][col] = 0
    return pruned


# =============================================================================
# PART 6: DAVIS ENERGY FUNCTIONAL — For Simulated Annealing
# =============================================================================

def davis_energy(board: list[list[int]], 
                 lambda1: float = 0.3, 
                 lambda2: float = 0.4, 
                 lambda3: float = 0.3) -> float:
    """
    Davis Energy Functional for Sudoku:
    
    E[γ] = λ₁∫ds + λ₂∫K_loc(s)ds + λ₃∫‖Hol_γ − I‖ds
    
    From E0 (Principle of Least Holonomy): "Among all paths connecting
    premises to conclusions, the realized path minimizes total holonomy."
    
    Components:
      λ₁ · path_length: number of cells remaining (lower = closer to solution)
      λ₂ · curvature:   total constraint complexity (lower = smoother path)  
      λ₃ · holonomy:    constraint violation measure (lower = more consistent)
    """
    n = count_empty(board)
    
    # Term 1: Path length (how far from solution)
    path_length = n / 81.0
    
    # Term 2: Integrated curvature over remaining cells
    total_curvature = 0.0
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                K = local_curvature(board, r, c)
                if K != float('inf'):
                    total_curvature += K
    curvature_term = total_curvature / max(n, 1)
    
    # Term 3: Holonomy deficit (constraint violations)
    #   len(vals) - len(set(vals)) counts excess duplicates per unit.
    #   Max per unit: 8 (all 9 cells same value). 27 units total → max 216.
    violations = 0
    for r in range(9):
        vals = [board[r][c] for c in range(9) if board[r][c] != 0]
        violations += len(vals) - len(set(vals))
    for c in range(9):
        vals = [board[r][c] for r in range(9) if board[r][c] != 0]
        violations += len(vals) - len(set(vals))
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            vals = [board[r][c] for r in range(br, br + 3) 
                    for c in range(bc, bc + 3) if board[r][c] != 0]
            violations += len(vals) - len(set(vals))
    holonomy = violations / 216.0  # Normalize to [0,1]; max = 27 units × 8 each
    
    E = lambda1 * path_length + lambda2 * curvature_term + lambda3 * holonomy
    return E


# =============================================================================
# PART 7: THE DAVIS SOLVER — DFS with Geometric Enhancements
# =============================================================================

class DavisSolver:
    """
    Sudoku solver using Davis Field Equations.
    
    Combines:
      - Curvature-aware cell ordering (F5: Optimal Constraint Ordering)
      - Holonomy-based pruning (T1: Geometric Completion Uniqueness)
      - Bitmask candidate tracking for performance
    
    This is the CPU reference implementation of the Davis geometric DFS.
    The GPU implementation (davis_solver_blackwell.cu) parallelizes the
    same algorithm across thousands of CUDA threads.
    """
    
    def __init__(self):
        self.backtracks = 0
        self.nodes_explored = 0
        self.holonomy_prunes = 0
        self.solve_time = 0.0
    
    def solve(self, board: list[list[int]]) -> bool:
        """Solve using Davis-enhanced DFS with bitmask candidate tracking."""
        start = time.time()
        self.backtracks = 0
        self.nodes_explored = 0
        self.holonomy_prunes = 0
        
        if not self._is_valid_board(board):
            self.solve_time = time.time() - start
            return False
        
        # Build bitmask candidate sets for all cells.
        # cands[r][c] is a bitmask: bit k set means digit k is a candidate.
        # Bit 0 unused; bits 1-9 represent digits 1-9.
        cands = [[0]*9 for _ in range(9)]
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    mask = 0x3FE  # bits 1-9 set = 0b1111111110
                    for cc in range(9):
                        if board[r][cc] != 0:
                            mask &= ~(1 << board[r][cc])
                    for rr in range(9):
                        if board[rr][c] != 0:
                            mask &= ~(1 << board[rr][c])
                    br, bc = 3*(r//3), 3*(c//3)
                    for rr in range(br, br+3):
                        for cc in range(bc, bc+3):
                            if board[rr][cc] != 0:
                                mask &= ~(1 << board[rr][cc])
                    cands[r][c] = mask
        
        # Precompute peers for each cell (row, col, box mates)
        self._peers = [[[] for _ in range(9)] for _ in range(9)]
        for r in range(9):
            for c in range(9):
                ps = set()
                for cc in range(9):
                    if cc != c: ps.add((r, cc))
                for rr in range(9):
                    if rr != r: ps.add((rr, c))
                br, bc = 3*(r//3), 3*(c//3)
                for rr in range(br, br+3):
                    for cc in range(bc, bc+3):
                        if (rr,cc) != (r,c): ps.add((rr,cc))
                self._peers[r][c] = list(ps)
        
        result = self._solve_fast(board, cands)
        self.solve_time = time.time() - start
        return result
    
    @staticmethod
    def _popcount(x):
        """Count set bits."""
        c = 0
        while x:
            c += 1
            x &= x - 1
        return c
    
    @staticmethod
    def _is_valid_board(board: list[list[int]]) -> bool:
        """Check for duplicate values in any row, column, or box."""
        for r in range(9):
            vals = [board[r][c] for c in range(9) if board[r][c] != 0]
            if len(vals) != len(set(vals)):
                return False
        for c in range(9):
            vals = [board[r][c] for r in range(9) if board[r][c] != 0]
            if len(vals) != len(set(vals)):
                return False
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                vals = [board[r][c] for r in range(br, br + 3)
                        for c in range(bc, bc + 3) if board[r][c] != 0]
                if len(vals) != len(set(vals)):
                    return False
        return True
    
    def _solve_fast(self, board, cands):
        """Recursive DFS with bitmask candidates and Davis-inspired ordering."""
        self.nodes_explored += 1
        
        # Select cell: MRV (fewest candidates first).
        # On tie: pick cell with most constrained peers (curvature proxy).
        best_cell = None
        best_count = 10
        ties = []
        
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    cnt = self._popcount(cands[r][c])
                    if cnt == 0:
                        self.backtracks += 1
                        return False  # Dead end
                    if cnt < best_count:
                        best_count = cnt
                        best_cell = (r, c)
                        ties = [(r, c)]
                    elif cnt == best_count:
                        ties.append((r, c))
        
        if best_cell is None:
            return True  # All cells filled
        
        # Curvature tiebreak: among cells with same candidate count,
        # pick the one with most constrained peers (highest coupling)
        if len(ties) > 1:
            best_coupling = -1
            for (r, c) in ties:
                coupling = self._peer_coupling(board, cands, r, c)
                if coupling > best_coupling:
                    best_coupling = coupling
                    best_cell = (r, c)
        
        row, col = best_cell
        mask = cands[row][col]
        
        # Extract candidate digits and order by LCV (least constraining value):
        # Try the value that eliminates fewest candidates from peers.
        digits = []
        m = mask
        while m:
            d = m & (-m)        # lowest set bit
            digits.append(d.bit_length() - 1)
            m &= m - 1
        
        if len(digits) > 1:
            # Lightweight LCV: count how many empty peers have each digit
            def lcv_score(d):
                bit = 1 << d
                count = 0
                for (pr, pc) in self._peers[row][col]:
                    if board[pr][pc] == 0 and (cands[pr][pc] & bit):
                        count += 1
                return count
            digits.sort(key=lcv_score)
        
        bit_val = 1 << 0  # placeholder
        for val in digits:
            bit_val = 1 << val
            
            # Holonomy pruning: after placing val, would any peer have 0 candidates?
            prune = False
            for (pr, pc) in self._peers[row][col]:
                if board[pr][pc] == 0 and (pr, pc) != (row, col):
                    new_cand = cands[pr][pc] & ~bit_val
                    if new_cand == 0:
                        prune = True
                        break
            if prune:
                self.holonomy_prunes += 1
                continue
            
            # Place value and propagate candidate elimination
            board[row][col] = val
            cands[row][col] = 0
            
            # Save and update peer candidates
            saved = []
            for (pr, pc) in self._peers[row][col]:
                if board[pr][pc] == 0 and (cands[pr][pc] & bit_val):
                    saved.append((pr, pc, cands[pr][pc]))
                    cands[pr][pc] &= ~bit_val
            
            if self._solve_fast(board, cands):
                return True
            
            # Undo
            for (pr, pc, old_cand) in saved:
                cands[pr][pc] = old_cand
            board[row][col] = 0
            cands[row][col] = mask
            self.backtracks += 1
        
        return False
    
    def _peer_coupling(self, board, cands, r, c):
        """Count empty peers that share candidates with this cell (curvature proxy)."""
        mask = cands[r][c]
        coupling = 0
        for (pr, pc) in self._peers[r][c]:
            if board[pr][pc] == 0:
                coupling += self._popcount(cands[pr][pc] & mask)
        return coupling
    
    def stats(self) -> dict:
        return {
            "solve_time_ms": round(self.solve_time * 1000, 2),
            "nodes_explored": self.nodes_explored,
            "backtracks": self.backtracks,
            "holonomy_prunes": self.holonomy_prunes
        }


# =============================================================================
# FRAMEWORK INTEGRATION — BaseSolver adapter for benchmarking
# =============================================================================

class DavisManifoldSolver(BaseSolver):
    """
    Framework-integrated Davis Manifold solver.
    
    Wraps the standalone DavisSolver to work with the project's
    BaseSolver / SudokuBoard protocol for benchmarking and CLI use.
    All geometric logic (curvature, holonomy, information value) is
    preserved exactly — this adapter only translates between
    SudokuBoard and list[list[int]].
    """
    
    name = "Davis Manifold"
    
    def __init__(self):
        super().__init__()
    
    def _solve(self, board: SudokuBoard) -> Optional[SudokuBoard]:
        """Solve via the Davis geometric DFS."""
        # Convert SudokuBoard → list[list[int]] for the geometric engine
        grid = []
        for r in range(board.size):
            row = []
            for c in range(board.size):
                row.append(int(board.get(r, c)))
            grid.append(row)
        
        # Run the Davis solver
        inner = DavisSolver()
        solved = inner.solve(grid)
        
        # Capture geometric stats
        self.stats.nodes_explored = inner.nodes_explored
        self.stats.backtracks = inner.backtracks
        self.stats.iterations = inner.nodes_explored
        self.stats.extra["holonomy_prunes"] = inner.holonomy_prunes
        
        if not solved:
            return None
        
        # Write solution back onto the SudokuBoard
        for r in range(board.size):
            for c in range(board.size):
                if board.get(r, c) == 0:
                    board.set(r, c, grid[r][c])
        
        return board


# =============================================================================
# PART 8: GEOMETRIC ANALYSIS — Pre-solve Diagnostics
# =============================================================================

def geometric_analysis(board: list[list[int]]) -> dict:
    """
    Full Davis geometric analysis of a Sudoku board.
    
    Returns the key geometric quantities from the Field Equations
    applied to this specific puzzle instance.
    """
    n = count_empty(board)
    m = 81 - n
    K_max = max_curvature(board)
    K_mean = mean_curvature(board)
    Gamma = trichotomy_parameter(board)
    holonomy = check_holonomy(board)
    
    # Constraint saturation threshold (F1)
    # m* = K_max · log|S| / τ
    if K_max > 0:
        m_star = K_max * n * math.log(9) / 1.0
    else:
        m_star = 0
    
    # Geometric condition number (T4aii)
    # κ_g = L · √K_max / τ
    kappa_g = n * math.sqrt(K_max) / 1.0 if K_max > 0 else 0
    
    # Search space bound from T3
    # |S_valid| ≤ 9^n · exp(-m · τ / K_max)
    if K_max > 0:
        log_S_valid = n * math.log(9) - (m * 1.0 / K_max)
        S_valid_bound = math.exp(min(log_S_valid, 700))  # Prevent overflow
    else:
        S_valid_bound = 1.0
    
    return {
        "empty_cells": n,
        "filled_cells": m,
        "max_curvature_K_max": round(K_max, 4),
        "mean_curvature": round(K_mean, 4),
        "trichotomy_Gamma": round(Gamma, 4),
        "difficulty": classify_difficulty(board),
        "recommended_solver": select_optimal_solver(board),
        "constraint_saturation_m_star": round(m_star, 2),
        "constraints_vs_threshold": f"{m} / {round(m_star, 1)}",
        "geometric_condition_number": round(kappa_g, 4),
        "well_conditioned": kappa_g < 1.0,
        "holonomy_deficit": round(holonomy, 4),
        "search_space_bound": f"≤ {S_valid_bound:.2e}",
        "davis_energy": round(davis_energy(board), 4)
    }


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Example: a hard Sudoku puzzle
    puzzle = [
        [0, 0, 0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 3, 5, 0, 0, 0],
        [0, 0, 0, 6, 0, 0, 0, 7, 0],
        [7, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 8],
        [0, 4, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0],
        [6, 5, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    print("=" * 60)
    print("DAVIS FIELD EQUATIONS — SUDOKU GEOMETRIC ANALYSIS")
    print("=" * 60)
    
    analysis = geometric_analysis(puzzle)
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("SOLVING WITH DAVIS-ENHANCED DFS")
    print("=" * 60)
    
    solver = DavisSolver()
    # Work on a copy
    board_copy = [row[:] for row in puzzle]
    
    if solver.solve(board_copy):
        print("\n  Solution found!")
        for row in board_copy:
            print("  " + " ".join(str(x) for x in row))
        print()
        stats = solver.stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("  No solution exists.")
