"""Command-line interface for the Sudoku solver system."""

import argparse
import sys
import json
from typing import Optional

from .generator import SudokuGenerator, Difficulty
from .solvers import DFSSolver, MCTSSolver, DLXSolver, AnnealingSolver
from .benchmark import Benchmark
from .benchmark.visualizer import Visualizer
from .benchmark.tuner import Tuner
from .core.board import SudokuBoard


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Sudoku Puzzle Generator & Multi-Algorithm Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 medium difficulty puzzles
  python -m sudoku.cli generate --count 5 --difficulty medium

  # Solve a puzzle with DFS
  python -m sudoku.cli solve --algorithm dfs --puzzle "0030206..."

  # Run full benchmark
  python -m sudoku.cli benchmark --puzzles 10 --output results/
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate Sudoku puzzles")
    gen_parser.add_argument(
        "--count", "-n", type=int, default=5,
        help="Number of puzzles to generate (default: 5)"
    )
    gen_parser.add_argument(
        "--difficulty", "-d", 
        choices=["easy", "medium", "hard", "expert", "all"],
        default="medium",
        help="Difficulty level (default: medium)"
    )
    gen_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file for puzzles (JSON format)"
    )
    gen_parser.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Random seed for reproducibility"
    )
    
    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a Sudoku puzzle")
    solve_parser.add_argument(
        "--algorithm", "-a",
        choices=["dfs", "mcts", "dlx", "annealing", "cp", "all"],
        default="dlx",
        help="Solving algorithm to use (default: dlx)"
    )
    solve_parser.add_argument(
        "--puzzle", "-p", type=str, required=True,
        help="Puzzle string (81 chars, 0 for empty cells)"
    )
    solve_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed solving statistics"
    )
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run solver benchmarks")
    bench_parser.add_argument(
        "--puzzles", "-n", type=int, default=10,
        help="Puzzles per difficulty (default: 10)"
    )
    bench_parser.add_argument(
        "--difficulty", "-d",
        choices=["easy", "medium", "hard", "expert", "all"],
        default="all",
        help="Difficulty to benchmark (default: all)"
    )
    bench_parser.add_argument(
        "--output", "-o", type=str, default="results",
        help="Output directory for results (default: results)"
    )
    bench_parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    bench_parser.add_argument(
        "--no-charts", action="store_true",
        help="Skip chart generation"
    )
    
    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Tune Simulated Annealing hyperparameters")
    tune_parser.add_argument(
        "--puzzles", "-n", type=int, default=10,
        help="Number of puzzles to use for tuning (default: 10)"
    )
    tune_parser.add_argument(
        "--difficulty", "-d",
        choices=["easy", "medium", "hard", "expert", "all"],
        default="hard",
        help="Difficulty to use for tuning (default: hard)"
    )
    tune_parser.add_argument(
        "--output", "-o", type=str, default="results/tuning",
        help="Output directory for tuning results (default: results/tuning)"
    )
    tune_parser.add_argument(
        "--quick", action="store_true",
        help="Run a quicker, smaller grid search"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "solve":
        cmd_solve(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "tune":
        cmd_tune(args)


def cmd_generate(args):
    """Handle the generate command."""
    generator = SudokuGenerator(seed=args.seed)
    
    if args.difficulty == "all":
        difficulties = list(Difficulty)
    else:
        difficulties = [Difficulty(args.difficulty)]
    
    all_puzzles = []
    
    for difficulty in difficulties:
        print(f"\nGenerating {args.count} {difficulty.value} puzzles...")
        puzzles = generator.generate_batch(args.count, difficulty)
        
        for i, puzzle in enumerate(puzzles, 1):
            puzzle_data = {
                "difficulty": difficulty.value,
                "index": i,
                "puzzle": puzzle.to_string(),
                "clues": puzzle.count_filled()
            }
            all_puzzles.append(puzzle_data)
            
            print(f"\n--- {difficulty.value.capitalize()} Puzzle {i} ({puzzle.count_filled()} clues) ---")
            print(puzzle)
    
    # If no output file is specified, save to a default 'puzzles' folder
    # but still create the all_puzzles list for potential JSON output
    if not args.output:
        import os
        base_dir = "puzzles"
        os.makedirs(base_dir, exist_ok=True)
        
        for difficulty in difficulties:
            diff_puzzles = [p for p in all_puzzles if p["difficulty"] == difficulty.value]
            # Convert back to SudokuBoard for save_to_folder
            board_puzzles = [SudokuBoard.from_string(p["puzzle"]) for p in diff_puzzles]
            
            diff_dir = os.path.join(base_dir, difficulty.value)
            SudokuGenerator.save_to_folder(board_puzzles, diff_dir, prefix=f"puzzle_{difficulty.value}")
        
        print(f"\nPuzzles also saved individually in the '{base_dir}/' directory")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_puzzles, f, indent=2)
        print(f"\nAll puzzles saved to {args.output}")
    
    print(f"\nTotal puzzles generated: {len(all_puzzles)}")


def cmd_solve(args):
    """Handle the solve command."""
    # Parse puzzle
    try:
        board = SudokuBoard.from_string(args.puzzle)
    except Exception as e:
        print(f"Error parsing puzzle: {e}")
        sys.exit(1)
    
    print("Input puzzle:")
    print(board)
    print()
    
    # Get solvers
    solvers = {}
    if args.algorithm == "all":
        solvers = {
            "DFS": DFSSolver(),
            "MCTS": MCTSSolver(max_iterations=5000),
            "DLX": DLXSolver(),
            "Annealing": AnnealingSolver(),
            "CP": CPSolver()
        }
    else:
        solver_map = {
            "dfs": ("DFS", DFSSolver()),
            "mcts": ("MCTS", MCTSSolver(max_iterations=5000)),
            "dlx": ("DLX", DLXSolver()),
            "annealing": ("Annealing", AnnealingSolver()),
            "cp": ("CP", CPSolver())
        }
        name, solver = solver_map[args.algorithm]
        solvers = {name: solver}
    
    # Solve with each algorithm
    for name, solver in solvers.items():
        print(f"Solving with {name}...")
        solution, stats = solver.solve(board)
        
        if stats.solved:
            print(f"✓ Solved in {stats.time_seconds:.4f}s")
            if args.verbose:
                print(f"  Iterations: {stats.iterations:,}")
                print(f"  Backtracks: {stats.backtracks:,}")
                print(f"  Memory: {stats.memory_bytes / 1024:.2f} KB")
            print(solution)
        else:
            print(f"✗ Failed to solve")
            if args.verbose:
                print(f"  Time: {stats.time_seconds:.4f}s")
                print(f"  Iterations: {stats.iterations:,}")
        print()


def cmd_benchmark(args):
    """Handle the benchmark command."""
    # Parse difficulties
    if args.difficulty == "all":
        difficulties = list(Difficulty)
    else:
        difficulties = [Difficulty(args.difficulty)]
    
    print("=" * 60)
    print("SUDOKU SOLVER BENCHMARK")
    print("=" * 60)
    print(f"Puzzles per difficulty: {args.puzzles}")
    print(f"Difficulties: {[d.value for d in difficulties]}")
    
    # Initialize benchmark to get solvers list
    benchmark = Benchmark(
        puzzles_per_difficulty=args.puzzles,
        difficulties=difficulties,
        seed=args.seed
    )
    
    print(f"Algorithms: {', '.join(benchmark.solvers.keys())}")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    # Run benchmark
    results = benchmark.run()
    
    # Print summary
    summary = benchmark.get_summary()
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\nBy Algorithm:")
    print("-" * 50)
    for algo, stats in summary["results_by_algorithm"].items():
        print(f"\n{algo}:")
        print(f"  Accuracy: {stats['accuracy']:.1f}% ({stats['total_solved']}/{stats['total_tested']})")
        print(f"  Avg Time: {stats['avg_time_seconds']:.4f}s")
        print(f"  Avg Memory: {stats['avg_memory_mb']:.2f} MB")
    
    # Save results
    benchmark.save_results(args.output)
    
    # Generate charts
    if not args.no_charts:
        print("\nGenerating charts...")
        visualizer = Visualizer(results, args.output)
        charts = visualizer.generate_all()
        visualizer.generate_summary_table()
        print(f"Charts saved to {args.output}/")
        for chart in charts:
            print(f"  - {chart.split('/')[-1]}")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")


def cmd_tune(args):
    """Handle the tune command."""
    # Parse difficulties
    if args.difficulty == "all":
        difficulties = list(Difficulty)
    else:
        difficulties = [Difficulty(args.difficulty)]
        
    print("=" * 60)
    print("SIMULATED ANNEALING HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Puzzles: {args.puzzles}")
    print(f"Difficulties: {[d.value for d in difficulties]}")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    tuner = Tuner(
        puzzles_count=args.puzzles,
        difficulties=difficulties,
        output_dir=args.output
    )
    
    if args.quick:
        results = tuner.tune_annealing(
            initial_temps=[1.0],
            cooling_rates=[0.9999, 0.99995],
            max_iterations_list=[100000],
            restarts_list=[3]
        )
    else:
        # Full grid search for all difficulties
        results = tuner.tune_annealing()
        
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    
    for diff in [d.value for d in difficulties]:
        best = tuner.get_best_params(diff)
        acc = max(r.accuracy for r in results[diff])
        print(f"\nDifficulty: {diff}")
        print(f"  Best Accuracy: {acc:.1f}%")
        print(f"  Best Parameters: {best}")
    
    print("\nVisualization and optimal parameters saved to:", args.output)
    print("=" * 60)


if __name__ == "__main__":
    main()
