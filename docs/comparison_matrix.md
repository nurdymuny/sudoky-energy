# Sudoku Solving Algorithms: Comparison Matrix

A comprehensive comparison of the four solving algorithms implemented in this project.

## Algorithm Overview

| Feature | DFS + Backtracking | MCTS | Dancing Links (DLX) | Simulated Annealing |
|---------|-------------------|------|---------------------|---------------------|
| **Type** | Deterministic | Probabilistic | Deterministic | Probabilistic |
| **Approach** | Exhaustive Search | Tree Search + Monte Carlo | Exact Cover | Energy Minimization |
| **Guaranteed Solution** | ✓ Yes | ✗ No | ✓ Yes | ✗ No |
| **Time Complexity** | O(9^n) worst case | O(iterations × depth) | O(branch factor) | O(iterations × restarts) |
| **Space Complexity** | O(n²) | O(nodes × n²) | O(constraints × possibilities) | O(n²) |

---

## Detailed Analysis

### 1. Depth-First Search with Backtracking

**How it works:**
- Recursively tries values in empty cells
- Uses Minimum Remaining Values (MRV) heuristic to pick cells wisely
- Constraint propagation (naked singles, hidden singles) prunes search space

**Pros:**
- ✓ Simple to understand and implement
- ✓ Guaranteed to find solution if one exists
- ✓ Very efficient with constraint propagation
- ✓ Low memory usage

**Cons:**
- ✗ Exponential worst-case complexity
- ✗ Performance depends heavily on puzzle structure
- ✗ Not easily parallelizable

**Best for:** General purpose solving, especially with optimizations

---

### 2. Monte Carlo Tree Search (MCTS)

**How it works:**
- Builds a search tree using UCB1 selection policy
- Random playouts estimate value of partial solutions
- Balances exploration vs exploitation

**Pros:**
- ✓ Anytime algorithm (can return best-so-far solution)
- ✓ Naturally parallelizable
- ✓ Can handle uncertainty and partial information
- ✓ Works well for game-playing scenarios

**Cons:**
- ✗ Not ideal for constraint satisfaction problems
- ✗ No guarantee of finding solution
- ✗ Slower than deterministic methods for Sudoku
- ✗ Randomness makes debugging difficult

**Best for:** Problems with stochastic elements or when anytime behavior is needed

---

### 3. Dancing Links (Algorithm X)

**How it works:**
- Formulates Sudoku as an exact cover problem
- Uses circular doubly-linked lists for efficient backtracking
- Covers/uncovers columns in O(1) operations

**Pros:**
- ✓ Optimal for exact cover problems
- ✓ Very efficient backtracking via pointer manipulation
- ✓ Guaranteed to find all solutions
- ✓ Scales well to larger puzzles

**Cons:**
- ✗ Complex implementation
- ✗ Higher memory usage for the data structure
- ✗ Overkill for simple puzzles

**Best for:** Finding all solutions, larger Sudoku variants (16×16, 25×25)

---

### 4. Simulated Annealing

**How it works:**
- Starts with random valid assignment per box
- Swaps cells to minimize constraint violations (energy)
- Temperature scheduling allows escaping local minima

**Pros:**
- ✓ Can escape local optima
- ✓ Simple energy function
- ✓ Good for highly constrained problems
- ✓ Easily parallelizable (multiple restarts)

**Cons:**
- ✗ No guarantee of finding solution
- ✗ Requires parameter tuning (temperature, cooling rate)
- ✗ May need multiple restarts
- ✗ Can be slow for easy puzzles

**Best for:** Hard puzzles where deterministic methods struggle

---

## Scalability to Larger Boards

| Board Size | DFS | MCTS | DLX | Annealing |
|------------|-----|------|-----|-----------|
| 9×9 | ⭐⭐⭐⭐ Fast | ⭐⭐ Slow | ⭐⭐⭐⭐⭐ Very Fast | ⭐⭐⭐ Moderate |
| 16×16 | ⭐⭐ Slow | ⭐ Very Slow | ⭐⭐⭐⭐ Fast | ⭐⭐⭐ Moderate |
| 25×25 | ⭐ Very Slow | ✗ Impractical | ⭐⭐⭐ Moderate | ⭐⭐ Slow |

### Scalability Notes:

- **DFS**: Complexity grows exponentially with board size. Constraint propagation helps but has limits.
- **MCTS**: Search space explodes; random playouts become increasingly unlikely to succeed.
- **DLX**: Best scalability. The exact cover matrix grows as O(n⁴) but operations remain efficient.
- **Annealing**: Swap neighborhood grows as O(n²), but energy landscape becomes more complex.

---

## Recommendation Summary

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Standard 9×9 puzzles | DFS or DLX |
| Speed is critical | DLX |
| Need all solutions | DLX |
| Very hard puzzles | Annealing or DLX |
| Larger boards (16×16+) | DLX |
| Learning/educational | DFS |
| Parallelization needed | Annealing (restarts) or MCTS |

---

## Performance Expectations (9×9 Sudoku)

| Algorithm | Easy | Medium | Hard | Expert |
|-----------|------|--------|------|--------|
| DFS | ~0.001s | ~0.005s | ~0.02s | ~0.1s |
| MCTS | ~0.5s | ~1s | ~2s | Often fails |
| DLX | ~0.0005s | ~0.001s | ~0.005s | ~0.01s |
| Annealing | ~0.1s | ~0.3s | ~0.5s | May need restarts |

*Note: Actual times depend on puzzle structure and system performance.*
