"""Hyperparameter tuning for Sudoku solvers."""

from __future__ import annotations
import time
import os
import json
import itertools
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.board import SudokuBoard
from ..generator import SudokuGenerator, Difficulty
from ..solvers.annealing_solver import AnnealingSolver


@dataclass
class TuningResult:
    """Result of a single hyperparameter configuration."""
    params: Dict[str, Any]
    accuracy: float
    avg_time: float
    avg_iterations: float
    avg_energy: float  # Remaining energy if not solved


class Tuner:
    """
    Hyperparameter tuner for Sudoku solvers.
    
    Currently focuses on Simulated Annealing but can be extended.
    """
    
    def __init__(
        self,
        puzzles_count: int = 10,
        difficulties: List[Difficulty] = [Difficulty.HARD],
        output_dir: str = "results/tuning",
        seed: Optional[int] = 42
    ):
        self.puzzles_count = puzzles_count
        self.difficulties = difficulties
        self.output_dir = output_dir
        self.seed = seed
        self.results: Dict[str, List[TuningResult]] = {}
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate test puzzles for each difficulty
        generator = SudokuGenerator(seed=self.seed)
        self.test_puzzles = {}
        for diff in self.difficulties:
            self.test_puzzles[diff.value] = generator.generate_batch(self.puzzles_count, diff)
            print(f"Generated {self.puzzles_count} {diff.value} puzzles for tuning.")

    def tune_annealing(
        self,
        initial_temps: List[float] = [1.0],
        cooling_rates: List[float] = [0.9999, 0.99995, 0.99999],
        max_iterations_list: List[int] = [100000, 300000],
        restarts_list: List[int] = [3, 5]
    ) -> Dict[str, List[TuningResult]]:
        """
        Perform grid search for Simulated Annealing parameters across all difficulties.
        """
        param_grid = list(itertools.product(
            initial_temps, 
            cooling_rates, 
            max_iterations_list, 
            restarts_list
        ))
        
        self.results = {}
        
        for diff_name, puzzles in self.test_puzzles.items():
            print(f"\nTuning for difficulty: {diff_name}")
            diff_results = []
            
            for init_temp, cool_rate, max_iter, restarts in tqdm(param_grid, desc=f"Tuning {diff_name}"):
                params = {
                    "initial_temp": init_temp,
                    "cooling_rate": cool_rate,
                    "max_iterations": max_iter,
                    "restarts": restarts
                }
                
                solved_count = 0
                total_time = 0.0
                total_iters = 0
                
                solver = AnnealingSolver(**params)
                
                for puzzle in puzzles:
                    solution, stats = solver.solve(puzzle)
                    if stats.solved:
                        solved_count += 1
                    total_time += stats.time_seconds
                    total_iters += stats.iterations
                
                res = TuningResult(
                    params=params,
                    accuracy=(solved_count / self.puzzles_count) * 100,
                    avg_time=total_time / self.puzzles_count,
                    avg_iterations=total_iters / self.puzzles_count,
                    avg_energy=0.0
                )
                diff_results.append(res)
            
            self.results[diff_name] = diff_results
            self._generate_plots(diff_name, diff_results)
            
        # Save categorical results
        self._save_results()
        self._save_optimal_params()
        
        return self.results

    def _save_optimal_params(self):
        """Save the best parameters found for each difficulty."""
        optimal_params = {}
        for diff_name, results in self.results.items():
            best = max(results, key=lambda x: x.accuracy)
            optimal_params[diff_name] = {
                "params": best.params,
                "accuracy": best.accuracy,
                "avg_time": best.avg_time
            }
            
        output_file = os.path.join(self.output_dir, "optimal_parameters.json")
        with open(output_file, "w") as f:
            json.dump(optimal_params, f, indent=2)
        print(f"\nOptimal parameters saved to {output_file}")

    def _save_results(self):
        """Save tuning results to JSON."""
        output_file = os.path.join(self.output_dir, "tuning_results.json")
        data = {}
        for diff, results in self.results.items():
            data[diff] = [
                {
                    "params": r.params,
                    "accuracy": r.accuracy,
                    "avg_time": r.avg_time,
                    "avg_iterations": r.avg_iterations
                }
                for r in results
            ]
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
            
    def _generate_plots(self, diff_name: str, results: List[TuningResult]):
        """Generate heatmap plots for parameter sensitivity for a specific difficulty."""
        try:
            import pandas as pd
            df = pd.DataFrame([
                {**r.params, "accuracy": r.accuracy}
                for r in results
            ])
            
            # If we have enough unique values for a heatmap
            if len(df['cooling_rate'].unique()) > 1 and len(df['restarts'].unique()) > 1:
                plt.figure(figsize=(10, 8))
                pivot = df.pivot_table(
                    index='cooling_rate', 
                    columns='restarts', 
                    values='accuracy',
                    aggfunc='mean'
                )
                sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.1f')
                plt.title(f'Annealing Accuracy ({diff_name}): Cooling Rate vs Restarts')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'accuracy_heatmap_{diff_name}.png'))
                plt.close()
                
            # Plot Accuracy vs Max Iterations
            if len(df['max_iterations'].unique()) > 1:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='max_iterations', y='accuracy', data=df)
                plt.title(f'Accuracy vs Max Iterations ({diff_name})')
                plt.savefig(os.path.join(self.output_dir, f'accuracy_vs_iters_{diff_name}.png'))
                plt.close()

        except Exception as e:
            print(f"Skipping plots for {diff_name} due to: {e}")
        
        # Print best for this difficulty
        best = max(results, key=lambda x: x.accuracy)
        print(f"  Best for {diff_name}: {best.accuracy}% with {best.params}")

    def get_best_params(self, difficulty: Optional[str] = None) -> Dict[str, Any]:
        """Return the best parameters found (optionally for a specific difficulty)."""
        if not self.results:
            return {}
            
        if difficulty and difficulty in self.results:
            return max(self.results[difficulty], key=lambda x: x.accuracy).params
            
        # Overall best across all difficulties
        all_res = []
        for r_list in self.results.values():
            all_res.extend(r_list)
        if not all_res: return {}
        return max(all_res, key=lambda x: x.accuracy).params
