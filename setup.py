from setuptools import setup, find_packages

setup(
    name="sudoku",
    version="1.0.0",
    description="Sudoku Puzzle Generator & Multi-Algorithm Solver",
    author="robomotic",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "sudoku=sudoku.cli:main",
        ],
    },
)
