"""Visualization utilities for benchmark results."""

from __future__ import annotations
import os
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

from .benchmark import BenchmarkResult


class Visualizer:
    """
    Visualization generator for Sudoku solver benchmark results.
    
    Creates charts comparing algorithm performance across various metrics.
    """
    
    # Color palette for algorithms
    COLORS = {
        "DFS": "#2ecc71",       # Green
        "MCTS": "#3498db",      # Blue
        "DLX": "#9b59b6",       # Purple
        "Annealing": "#e74c3c", # Red
        "CP": "#f39c12"         # Orange
    }
    
    def __init__(self, results: List[BenchmarkResult], output_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            results: List of benchmark results.
            output_dir: Directory to save generated charts.
        """
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def generate_all(self) -> List[str]:
        """
        Generate all charts.
        
        Returns:
            List of paths to generated chart files.
        """
        charts = []
        
        charts.append(self.plot_time_comparison())
        charts.append(self.plot_time_by_difficulty())
        charts.append(self.plot_accuracy_comparison())
        charts.append(self.plot_memory_comparison())
        charts.append(self.plot_time_distribution())
        charts.append(self.plot_iterations_comparison())
        
        return charts
    
    def plot_time_comparison(self) -> str:
        """Create bar chart comparing average solve times."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by algorithm
        algorithms = sorted(set(r.algorithm for r in self.results))
        avg_times = []
        colors = []
        
        for algo in algorithms:
            times = [r.time_seconds for r in self.results if r.algorithm == algo]
            avg_times.append(np.mean(times))
            colors.append(self.COLORS.get(algo, "#95a5a6"))
        
        bars = ax.bar(algorithms, avg_times, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, time in zip(bars, avg_times):
            height = bar.get_height()
            ax.annotate(f'{time:.4f}s',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Average Time (seconds)', fontsize=12)
        ax.set_title('Average Solve Time by Algorithm', fontsize=14, fontweight='bold')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "time_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_time_by_difficulty(self) -> str:
        """Create grouped bar chart of times by difficulty and algorithm."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        algorithms = sorted(set(r.algorithm for r in self.results))
        difficulties = sorted(set(r.difficulty for r in self.results))
        
        x = np.arange(len(difficulties))
        width = 0.8 / len(algorithms)
        
        for i, algo in enumerate(algorithms):
            times = []
            for diff in difficulties:
                algo_diff_times = [
                    r.time_seconds for r in self.results 
                    if r.algorithm == algo and r.difficulty == diff
                ]
                times.append(np.mean(algo_diff_times) if algo_diff_times else 0)
            
            offset = (i - len(algorithms) / 2 + 0.5) * width
            bars = ax.bar(x + offset, times, width, 
                         label=algo, 
                         color=self.COLORS.get(algo, "#95a5a6"),
                         edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Difficulty', fontsize=12)
        ax.set_ylabel('Average Time (seconds)', fontsize=12)
        ax.set_title('Solve Time by Difficulty and Algorithm', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in difficulties])
        ax.legend(title='Algorithm')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "time_by_difficulty.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_accuracy_comparison(self) -> str:
        """Create grouped bar chart comparing solve accuracy by difficulty."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        algorithms = sorted(set(r.algorithm for r in self.results))
        difficulties = sorted(set(r.difficulty for r in self.results))
        
        x = np.arange(len(difficulties))
        width = 0.8 / len(algorithms)
        
        for i, algo in enumerate(algorithms):
            accuracies = []
            for diff in difficulties:
                algo_diff_results = [
                    r for r in self.results 
                    if r.algorithm == algo and r.difficulty == diff
                ]
                solved = sum(1 for r in algo_diff_results if r.solved)
                accuracy = (solved / len(algo_diff_results)) * 100 if algo_diff_results else 0
                accuracies.append(accuracy)
            
            offset = (i - len(algorithms) / 2 + 0.5) * width
            bars = ax.bar(x + offset, accuracies, width, 
                         label=algo, 
                         color=self.COLORS.get(algo, "#95a5a6"),
                         edgecolor='black', linewidth=0.5)
            
            # Add percentage labels for small sets (optional, can be cluttered)
            if len(difficulties) <= 4 and len(algorithms) <= 4:
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{acc:.0f}%',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Difficulty', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Solve Accuracy by Difficulty and Algorithm', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in difficulties])
        ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 115)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "accuracy_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_memory_comparison(self) -> str:
        """Create bar chart comparing memory usage."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = sorted(set(r.algorithm for r in self.results))
        avg_memory = []
        colors = []
        
        for algo in algorithms:
            memory = [r.memory_bytes / (1024 * 1024) for r in self.results if r.algorithm == algo]
            avg_memory.append(np.mean(memory))
            colors.append(self.COLORS.get(algo, "#95a5a6"))
        
        bars = ax.bar(algorithms, avg_memory, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, mem in zip(bars, avg_memory):
            height = bar.get_height()
            ax.annotate(f'{mem:.2f} MB',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Average Memory (MB)', fontsize=12)
        ax.set_title('Memory Usage by Algorithm', fontsize=14, fontweight='bold')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "memory_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_time_distribution(self) -> str:
        """Create box plot showing time distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        algorithms = sorted(set(r.algorithm for r in self.results))
        data = []
        labels = []
        
        for algo in algorithms:
            times = [r.time_seconds for r in self.results if r.algorithm == algo]
            data.append(times)
            labels.append(algo)
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color boxes
        for patch, algo in zip(bp['boxes'], algorithms):
            patch.set_facecolor(self.COLORS.get(algo, "#95a5a6"))
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Solve Time Distribution by Algorithm', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "time_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_iterations_comparison(self) -> str:
        """Create grouped bar chart comparing average iterations by difficulty."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        algorithms = sorted(set(r.algorithm for r in self.results))
        difficulties = sorted(set(r.difficulty for r in self.results))
        
        x = np.arange(len(difficulties))
        width = 0.8 / len(algorithms)
        
        for i, algo in enumerate(algorithms):
            avg_iters = []
            for diff in difficulties:
                algo_diff_results = [
                    r for r in self.results 
                    if r.algorithm == algo and r.difficulty == diff
                ]
                iters = [r.iterations for r in algo_diff_results]
                avg_iters.append(np.mean(iters) if iters else 0)
            
            offset = (i - len(algorithms) / 2 + 0.5) * width
            ax.bar(x + offset, avg_iters, width, 
                   label=algo, 
                   color=self.COLORS.get(algo, "#95a5a6"),
                   edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Difficulty', fontsize=12)
        ax.set_ylabel('Average Iterations (Log Scale)', fontsize=12)
        ax.set_title('Average Iterations by Difficulty and Algorithm', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in difficulties])
        ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Use log scale for iterations as they can vary by several orders of magnitude
        ax.set_yscale('log')
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "iterations_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def generate_summary_table(self) -> str:
        """Generate a markdown summary table."""
        algorithms = sorted(set(r.algorithm for r in self.results))
        
        lines = [
            "# Benchmark Summary\n",
            "| Algorithm | Accuracy | Avg Time | Avg Memory | Avg Iterations |",
            "|-----------|----------|----------|------------|----------------|"
        ]
        
        for algo in algorithms:
            algo_results = [r for r in self.results if r.algorithm == algo]
            
            solved = sum(1 for r in algo_results if r.solved)
            accuracy = (solved / len(algo_results)) * 100 if algo_results else 0
            
            avg_time = np.mean([r.time_seconds for r in algo_results])
            avg_memory = np.mean([r.memory_bytes / (1024 * 1024) for r in algo_results])
            avg_iters = np.mean([r.iterations for r in algo_results])
            
            lines.append(
                f"| {algo} | {accuracy:.1f}% | {avg_time:.4f}s | {avg_memory:.2f} MB | {int(avg_iters):,} |"
            )
        
        content = "\n".join(lines)
        
        path = os.path.join(self.output_dir, "benchmark_summary.md")
        with open(path, "w") as f:
            f.write(content)
        
        return path
