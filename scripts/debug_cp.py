
from sudoku.core.board import SudokuBoard
from sudoku.generator import SudokuGenerator, Difficulty
from sudoku.solvers import CPSolver, DLXSolver

def debug_cp():
    # Generate a hard puzzle
    gen = SudokuGenerator(seed=42)
    puzzles = gen.generate_batch(1, Difficulty.HARD)
    puzzle = puzzles[0]
    
    print("Puzzle:", flush=True)
    print(puzzle, flush=True)
    
    # Run CP Solver
    cp = CPSolver()
    
    # Manually run parts of _solve to trace
    print("\nTracing CP Solver:", flush=True)
    cp.stats.iterations = 0
    cp.domains = cp._initialize_domains(puzzle)
    print(f"Domains initialized. Size: {len(cp.domains)}", flush=True)
    
    # Initial AC-3
    print("Running Initial AC-3...", flush=True)
    consistent = cp._enforce_arc_consistency(puzzle)
    print(f"Initial AC-3 Consistent: {consistent}", flush=True)
    
    if not consistent:
        print("FAILED AT INITIAL AC-3", flush=True)
        # Find empty domain
        for pos, dom in cp.domains.items():
            if not dom:
                print(f"Empty domain at {pos}", flush=True)
        return

    # One loop of Naked Singles
    print("Running Naked Singles...", flush=True)
    changed = cp._apply_naked_singles(puzzle)
    print(f"Naked Singles Changed: {changed}", flush=True)
    
    if changed:
        consistent = cp._enforce_arc_consistency(puzzle)
        print(f"Re-propagate AC-3 Consistent: {consistent}", flush=True)

    # Backtracking test
    print("Entering Backtracking...", flush=True)
    result = cp._backtrack_on_domains(puzzle)
    print(f"Backtrack count: {cp.stats.iterations}", flush=True)
    print(f"Backtrack Result: {result is not None}", flush=True)

if __name__ == "__main__":
    debug_cp()
