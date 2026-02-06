
from sudoku.generator import Difficulty
from sudoku.benchmark.tuner import Tuner

tuner = Tuner(
    puzzles_count=1,
    difficulties=[Difficulty.EASY],
    output_dir="results/tuning"
)

# Run with parameters that trigger heatmap and boxplot
tuner.tune_annealing(
    initial_temps=[1.0],
    cooling_rates=[0.999, 0.9999],
    max_iterations_list=[50000, 100000],
    restarts_list=[1, 2]
)
print("Tuning visualization generation script complete.")
