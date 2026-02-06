
import collections
from sudoku.core.board import SudokuBoard
from sudoku.solvers.cp_solver import CPSolver

def reproduce():
    # Create empty board
    board = SudokuBoard(9)
    cp = CPSolver()
    cp.domains = cp._initialize_domains(board)
    
    # Setup scenario:
    # Cell (0,0) and (1,0) are empty.
    # Both have 4 in their domains.
    # We pretend Row 0 logic determines (0,0) must be 4.
    # Then we run Hidden Singles checking Column 0.
    
    r1, c1 = 0, 0
    r2, c2 = 1, 0
    val = 4
    
    # Manually set domains to simulate state
    cp.domains[(r1, c1)] = {val} # constrained to 4
    cp.domains[(r2, c2)] = {val, 5} # can be 4 or 5
    
    # Simulate Row Check setting (0,0) = 4
    # This is what happens inside _apply_hidden_singles loop
    board.set(r1, c1, val)
    # Note: we do NOT remove 4 from (1,0) domain yet (simulating no AC-3)
    
    print(f"Cell {r1,c1} is set to {board.get(r1,c1)}")
    print(f"Cell {r2,c2} is {board.get(r2,c2)} with domain {cp.domains[(r2,c2)]}")
    
    # Now run _find_hidden_in_unit for Column 0
    # Column 0 unit: (0,0), (1,0), ... (8,0)
    col_unit = [(r, 0) for r in range(9)]
    
    print("Running _find_hidden_in_unit on Column 0...")
    cp._find_hidden_in_unit(board, col_unit)
    
    print(f"Cell {r2,c2} value after check: {board.get(r2,c2)}")
    
    if board.get(r2, c2) == val:
        print("BUG REPRODUCED: Solver set duplicate value in column!")
    else:
        print("No bug triggered.")

if __name__ == "__main__":
    reproduce()
