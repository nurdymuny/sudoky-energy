# Davis Manifold Sudoku Solver — Build Configuration
# ==================================================
# "The Field Equations of Semantic Coherence" (B. R. Davis, 2025)
#
# Targets:
#   make blackwell   — Blackwell (sm_120), optimal for RTX 5070/5090
#   make blackwell100 — Blackwell B200/B100 (sm_100)
#   make hopper      — Hopper H100/H200 (sm_90), full features
#   make ampere      — Ampere A100/A10 (sm_80), compatible
#   make all         — Build all architectures
#   make gpu         — Alias for blackwell
#   make test        — Run CPU solver tests
#   make bench       — Run GPU benchmark (10K puzzles)
#   make clean       — Remove build artifacts

NVCC        := nvcc
CXX         := g++

# Detect OS for platform-specific commands
ifeq ($(OS),Windows_NT)
    PYTHON  := python
    MKDIR   = if not exist $(1) mkdir $(1)
    RM_RF   = if exist $(1) rmdir /s /q $(1)
    RM_F    = del /f /q $(1) 2>nul || true
    BIN_EXT := .exe
    LIB_EXT := .dll
else
    PYTHON  := python3
    MKDIR   = mkdir -p $(1)
    RM_RF   = rm -rf $(1)
    RM_F    = rm -f $(1)
    BIN_EXT :=
    LIB_EXT := .so
endif

# Common NVCC flags
NVCC_BASE   := -O3 --use_fast_math -Xcompiler -fPIC -std=c++17
NVCC_SHARED := -shared -Xcompiler -fPIC

# Blackwell RTX 50-series (sm_120): Consumer Blackwell
# Requires CUDA Toolkit 12.8+ and --allow-unsupported-compiler on MSVC
SM_BLACKWELL120 := -arch=sm_120 -gencode=arch=compute_120,code=sm_120

# Blackwell (sm_100): Thread block clusters, TMA, FP8 tensor cores
# Requires CUDA Toolkit 12.8+
SM_BLACKWELL := -arch=sm_100 -gencode=arch=compute_100,code=sm_100

# Hopper (sm_90): Thread block clusters, TMA, FP8
# Requires CUDA Toolkit 12.0+
SM_HOPPER   := -arch=sm_90 -gencode=arch=compute_90,code=sm_90

# Ampere (sm_80): Basic cooperative groups
# Requires CUDA Toolkit 11.0+
SM_AMPERE   := -arch=sm_80 -gencode=arch=compute_80,code=sm_80

# Multi-arch (fat binary, covers Ampere through Blackwell)
SM_MULTI    := -gencode=arch=compute_80,code=sm_80 \
               -gencode=arch=compute_90,code=sm_90 \
               -gencode=arch=compute_100,code=sm_100 \
               -gencode=arch=compute_120,code=sm_120

# Source files
CUDA_SRC    := davis_solver_blackwell.cu
CPU_SRC     := davis_solver.py
GPU_WRAPPER := davis_solver_gpu.py

# Output
BUILD_DIR   := build
LIB_NAME    := libdavis_solver$(LIB_EXT)
BIN_NAME    := davis_solver

# ============================================================
# Build targets
# ============================================================

.PHONY: all blackwell blackwell100 hopper ampere gpu test bench clean

# Default: Blackwell RTX 50-series (sm_120)
gpu: blackwell

blackwell: $(BUILD_DIR)
	$(NVCC) $(NVCC_BASE) $(SM_BLACKWELL120) \
		-o $(BUILD_DIR)/$(BIN_NAME)_blackwell$(BIN_EXT) $(CUDA_SRC)
	$(NVCC) $(NVCC_BASE) $(NVCC_SHARED) $(SM_BLACKWELL120) \
		-o $(BUILD_DIR)/$(LIB_NAME) $(CUDA_SRC)
	@echo ""
	@echo "Built for Blackwell RTX 50-series (sm_120)"
	@echo "  Binary: $(BUILD_DIR)/$(BIN_NAME)_blackwell$(BIN_EXT)"
	@echo "  Library: $(BUILD_DIR)/$(LIB_NAME)"

blackwell100: $(BUILD_DIR)
	$(NVCC) $(NVCC_BASE) $(SM_BLACKWELL) \
		-o $(BUILD_DIR)/$(BIN_NAME)_blackwell100$(BIN_EXT) $(CUDA_SRC)
	$(NVCC) $(NVCC_BASE) $(NVCC_SHARED) $(SM_BLACKWELL) \
		-o $(BUILD_DIR)/$(LIB_NAME) $(CUDA_SRC)
	@echo ""
	@echo "Built for Blackwell B200/B100 (sm_100)"

hopper: $(BUILD_DIR)
	$(NVCC) $(NVCC_BASE) $(SM_HOPPER) \
		-o $(BUILD_DIR)/$(BIN_NAME)_hopper$(BIN_EXT) $(CUDA_SRC)
	$(NVCC) $(NVCC_BASE) $(NVCC_SHARED) $(SM_HOPPER) \
		-o $(BUILD_DIR)/$(LIB_NAME) $(CUDA_SRC)
	@echo ""
	@echo "Built for Hopper (sm_90)"

ampere: $(BUILD_DIR)
	$(NVCC) $(NVCC_BASE) $(SM_AMPERE) \
		-o $(BUILD_DIR)/$(BIN_NAME)_ampere$(BIN_EXT) $(CUDA_SRC)
	$(NVCC) $(NVCC_BASE) $(NVCC_SHARED) $(SM_AMPERE) \
		-o $(BUILD_DIR)/$(LIB_NAME) $(CUDA_SRC)
	@echo ""
	@echo "Built for Ampere (sm_80)"

all: $(BUILD_DIR)
	$(NVCC) $(NVCC_BASE) $(SM_MULTI) \
		-o $(BUILD_DIR)/$(BIN_NAME)$(BIN_EXT) $(CUDA_SRC)
	$(NVCC) $(NVCC_BASE) $(NVCC_SHARED) $(SM_MULTI) \
		-o $(BUILD_DIR)/$(LIB_NAME) $(CUDA_SRC)
	@echo ""
	@echo "Built fat binary (sm_80 + sm_90 + sm_100 + sm_120)"

$(BUILD_DIR):
	$(call MKDIR,$(BUILD_DIR))

# ============================================================
# Test & Benchmark
# ============================================================

test:
	@echo "Running CPU solver tests..."
	$(PYTHON) -c "\
from davis_solver import *; \
import math; \
errors = []; \
b=[[0]*9 for _ in range(9)]; b[0][0]=5; b[0][1]=3; b[1][0]=6; \
assert get_candidates(b,0,2)=={1,2,4,7,8,9}, 'T1'; \
print('T1  get_candidates:    PASS'); \
solved=[[(i*3+i//3+j)%9+1 for j in range(9)] for i in range(9)]; \
E=davis_energy(solved); assert E==0.0, f'T2: {E}'; \
print('T2  energy(solved)=0:  PASS'); \
s=DavisSolver(); bc=[[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]; \
assert s.solve(bc), 'T3'; \
assert all(sorted(bc[r])==list(range(1,10)) for r in range(9)), 'T3v'; \
st=s.stats(); \
print(f'T3  solver correct:    PASS ({st[\"solve_time_ms\"]}ms, {st[\"nodes_explored\"]} nodes)'); \
print(); print('All CPU tests passed.')"

bench: blackwell
	@echo "Running GPU benchmark..."
	$(BUILD_DIR)/$(BIN_NAME)_blackwell 10000

bench-single: blackwell
	$(BUILD_DIR)/$(BIN_NAME)_blackwell

# ============================================================
# Python interface
# ============================================================

bench-python: blackwell
	@echo "Running Python GPU benchmark..."
	$(PYTHON) $(GPU_WRAPPER) 10000

# ============================================================
# Clean
# ============================================================

clean:
	$(call RM_RF,$(BUILD_DIR))
	$(call RM_F,*.o)
	$(call RM_F,*.so)
	$(call RM_F,*.dll)
	$(call RM_F,*.exe)
	$(call RM_F,*.exp)
	$(call RM_F,*.lib)
	$(call RM_F,*.pdb)
	@echo "Cleaned."

# ============================================================
# Help
# ============================================================

help:
	@echo "Davis Manifold Sudoku Solver — Build Targets"
	@echo ""
	@echo "  make blackwell    — Build for Blackwell RTX 50-series (sm_120)"
	@echo "  make blackwell100 — Build for Blackwell B200/B100 (sm_100)"
	@echo "  make hopper       — Build for Hopper H100/H200 (sm_90)"
	@echo "  make ampere       — Build for Ampere A100/A10 (sm_80)"
	@echo "  make all          — Fat binary for all architectures"
	@echo "  make gpu          — Alias for blackwell (default)"
	@echo "  make test         — Run CPU solver unit tests"
	@echo "  make bench        — GPU benchmark (10K puzzles)"
	@echo "  make clean        — Remove build artifacts"
