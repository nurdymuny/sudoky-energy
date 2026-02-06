
import os
import json
import glob
from typing import List, Dict, Any

from sudoku.core.board import SudokuBoard
from sudoku.solvers import CPSolver
from sudoku.benchmark.benchmark import Benchmark, BenchmarkResult
from sudoku.benchmark.visualizer import Visualizer  # Note: Class name is Visualizer, not BenchmarkVisualizer

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.json")
PUZZLES_DIR = os.path.join(RESULTS_DIR, "puzzles")

def load_puzzles() -> Dict[str, List[SudokuBoard]]:
    puzzles = {}
    if not os.path.exists(PUZZLES_DIR):
        print(f"Puzzles directory {PUZZLES_DIR} not found!")
        return {}
    
    for difficulty_level in ["easy", "medium", "hard", "expert"]:
        diff_dir = os.path.join(PUZZLES_DIR, difficulty_level)
        if not os.path.isdir(diff_dir):
            continue
            
        puzzles[difficulty_level] = []
        files = glob.glob(os.path.join(diff_dir, "*.txt"))
        files.sort()
        
        for fpath in files:
            with open(fpath, "r") as f:
                content = f.read().strip()
                if not content:
                    continue
                puzzle_str = content.split('\n')[0].strip()
                if puzzle_str:
                    board = SudokuBoard.from_string(puzzle_str)
                    puzzles[difficulty_level].append(board)
    
    return puzzles

def update_benchmark():
    # 1. Load existing results
    old_results = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                old_results = json.load(f)
        except Exception as e:
            print(f"Error loading {RESULTS_FILE}: {e}")
            return
    else:
        print(f"{RESULTS_FILE} not found. Running fresh?")

    # 2. Filter out old CPSolver results
    filtered_results = [r for r in old_results if r["algorithm"] != "CPSolver"]
    print(f"Loaded {len(old_results)} results, kept {len(filtered_results)} after removing old CPSolver data.")
    
    # 3. Load puzzles
    puzzles = load_puzzles()
    total_puzzles = sum(len(p) for p in puzzles.values())
    print(f"Loaded {total_puzzles} puzzles from disk.")
    
    if total_puzzles == 0:
        print("No puzzles found. Aborting.")
        return

    # 4. Initialize Benchmark with only CPSolver
    benchmark = Benchmark(solvers={"CPSolver": CPSolver()})
    
    # Manually inject puzzles
    benchmark.puzzles = puzzles
    
    # 5. Run Benchmark
    print("Running CPSolver benchmark...")
    new_results_objs = benchmark.run()
    new_results = [r.to_dict() for r in new_results_objs]
    
    # 6. Merge
    final_results = filtered_results + new_results
    
    # 7. Save merged results
    with open(RESULTS_FILE, "w") as f:
        json.dump(final_results, f, indent=2)
        
    print(f"Merged results saved to {RESULTS_FILE}")

    # 8. Reconstruct BenchmarkResult objects for summary/plots
    all_result_objs = []
    for r in final_results:
        kwargs = {
            k: r[k] for k in ["puzzle_id", "difficulty", "algorithm", "solved", 
                              "time_seconds", "memory_bytes", "iterations", 
                              "backtracks", "nodes_explored", "extra"]
            if k in r
        }
        all_result_objs.append(BenchmarkResult(**kwargs))
    
    # 9. Generate Summary
    # We can use a temporary Benchmark instance to help with summary structure, 
    # but Benchmark.get_summary() relies on self.solvers and self.results.
    summary_bench = Benchmark()
    summary_bench.results = all_result_objs
    
    unique_solvers = set(r["algorithm"] for r in final_results)
    summary_bench.solvers = {name: None for name in unique_solvers} 
    
    from sudoku.generator import Difficulty
    unique_diffs = set(r["difficulty"] for r in final_results)
    summary_bench.difficulties = [d for d in Difficulty if d.value in unique_diffs]
    
    summary = summary_bench.get_summary()
    
    summary_file = os.path.join(RESULTS_DIR, "benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # 10. Update plots
    try:
        print("Updating plots...")
        # Add CPSolver color
        Visualizer.COLORS["CPSolver"] = "#f39c12"  # Orange
        
        viz = Visualizer(all_result_objs, RESULTS_DIR)
        viz.generate_all()
        # Also generate markdown summary table
        viz.generate_summary_table()
        print("Plots and summary table updated.")
    except Exception as e:
        print(f"Failed to update plots: {e}")

if __name__ == "__main__":
    update_benchmark()
