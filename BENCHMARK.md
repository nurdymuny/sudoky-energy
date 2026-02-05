# Sudoku Solver Benchmark Results

This document summarizes the performance findings for various Sudoku solving algorithms tested across different difficulty levels.

## Executive Summary

The benchmark evaluated four different Sudoku solving algorithms against a set of 20 puzzles (5 for each difficulty level: Easy, Medium, Hard, and Expert).

| Algorithm | Accuracy | Avg Time (s) | Max Time (s) | Avg Memory (MB) |
|-----------|----------|--------------|--------------|-----------------|
| **DLX** (Dancing Links) | 100.0% | 0.0203 | 0.0468 | 0.11 |
| **DFS** (Backtracking) | 100.0% | 0.1001 | 0.4020 | 0.01 |
| **Annealing** | 65.0% | 12.6977 | 25.6068 | 0.01 |
| **MCTS** | 50.0% | 52.0940 | 123.5635 | 35.50 |

---

## Detailed Performance Analysis

### 1. Algorithm Accuracy by Difficulty
| Difficulty | DLX | DFS | Annealing | MCTS |
|------------|-----|-----|-----------|------|
| **Easy** | 100% | 100% | 100% | 100% |
| **Medium** | 100% | 100% | 100% | 100% |
| **Hard** | 100% | 100% | 20% | 0% |
| **Expert** | 100% | 100% | 40% | 0% |

### 2. Average Execution Time (Seconds)
| Difficulty | DLX | DFS | Annealing | MCTS |
|------------|-----|-----|-----------|------|
| **Easy** | 0.0158 | 0.0268 | 2.3207 | 0.0542 |
| **Medium**| 0.0227 | 0.0425 | 2.8906 | 1.0005 |
| **Hard** | 0.0209 | 0.1887 | 22.7202 | 107.4459 |
| **Expert** | 0.0217 | 0.1424 | 22.8592 | 99.8754 |

---

## Key Findings

### 🏆 Top Performer: DLX (Dancing Links)
- **DLX** is the clear winner in both speed and robustness. 
- It maintains a consistent sub-30ms solving time regardless of puzzle difficulty.
- 100% accuracy across all tested puzzles.

### 🥈 Runner Up: DFS (Depth-First Search / Backtracking)
- **DFS** is highly efficient and 100% accurate.
- While significantly slower than DLX on harder puzzles (approx. 5-10x slower), it remains well under 1 second for all puzzles.
- Lowest memory footprint of all algorithms.

### ⚠️ Stochastic Algorithms: Annealing & MCTS
- **Simulated Annealing** performs well on Easy and Medium puzzles but its heuristic nature causes it to struggle significantly with Hard and Expert levels, where the search space is more constrained.
- **MCTS** (Monte Carlo Tree Search) lacks the efficiency required for Sudoku. While it solves Easy/Medium puzzles reasonably, it fails to find solutions for any Hard or Expert puzzles within the time limits. It also consumes significantly more memory (35.5 MB) due to tree expansion.

## Visualizations
The following plots are available in the `results/` directory for further visual analysis:
- `time_comparison.png`: Overall execution time comparisons.
- `accuracy_comparison.png`: Success rates across algorithms.
- `memory_comparison.png`: Memory usage distribution.
- `time_by_difficulty.png`: Performance scaling with difficulty.

---
*Generated based on benchmark results as of 2026-02-05.*
