# Sudoku Puzzle Generator & Multi-Algorithm Solver

A comprehensive Sudoku puzzle generator and solver system featuring four distinct solving algorithms with comparative benchmarking.

## Features

- **Puzzle Generator**: Create Sudoku puzzles with 4 difficulty levels (Easy, Medium, Hard, Expert)
- **Five Solving Algorithms**:
  - Depth-First Search with Backtracking
  - Constraint Programming (CP) Solver
  - Dancing Links (Knuth's Algorithm X)
  - Simulated Annealing
  - Monte Carlo Tree Search (MCTS)
- **Benchmarking**: Performance comparison with time, memory, and accuracy metrics
- **Visualization**: Charts and comparison matrix

## Installation

```bash
cd /home/robomotic/DevOps/github/sudoku
pip install -e .
```

## Usage

### Generate Puzzles

```bash
# Generate 5 medium difficulty puzzles
python -m sudoku.cli generate --count 5 --difficulty medium

# Generate puzzles for all difficulties
python -m sudoku.cli generate --count 10 --difficulty all
```

### Solve Puzzles

```bash
# Solve with specific algorithm
python -m sudoku.cli solve --algorithm dfs --puzzle "puzzle_string"
python -m sudoku.cli solve --algorithm mcts --puzzle "puzzle_string"
python -m sudoku.cli solve --algorithm dlx --puzzle "puzzle_string"
python -m sudoku.cli solve --algorithm cp --puzzle "puzzle_string"
python -m sudoku.cli solve --algorithm annealing --puzzle "puzzle_string"
```

### Run Benchmarks

```bash
# Full benchmark suite
python -m sudoku.cli benchmark --puzzles 20 --output results/

# Benchmark specific difficulty
python -m sudoku.cli benchmark --puzzles 10 --difficulty hard --output results/
```

## Algorithms Overview

| Algorithm | Best For | Time Complexity | Guaranteed Solution |
|-----------|----------|-----------------|---------------------|
| DFS+Backtracking | All puzzles | O(9^n) | Yes |
| CP Solver | Logic-heavy puzzles | Mixed | Yes |
| Dancing Links | Exact solving | O(branches) | Yes |
| Simulated Annealing | Heuristic Search | O(iterations) | No |
| MCTS | Exploration | Configurable | No |

## Project Structure

```
sudoku/
├── sudoku/
│   ├── core/          # Board representation & validation
│   ├── generator/     # Puzzle generation
│   ├── solvers/       # Solving algorithms
│   └── benchmark/     # Benchmarking & visualization
├── docs/              # Documentation
├── results/           # Benchmark outputs
└── tests/             # Unit tests
```

## License

MIT License
