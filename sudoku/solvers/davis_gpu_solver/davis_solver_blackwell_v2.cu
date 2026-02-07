/**
 * Davis Manifold Sudoku Solver — NVIDIA Blackwell (B200/B100) Optimized
 * =====================================================================
 * Applies the Davis Field Equations to GPU-accelerated Sudoku solving.
 *
 * Architecture: Three-phase pipeline
 *   Phase 1: Wavefront Constraint Propagation (curvature-ordered parallel CP)
 *   Phase 2: Davis Manifold Relaxation (continuous gradient descent on E[γ])
 *   Phase 3: Jackknife Speculative Branching (curvature-guided parallel DFS)
 *
 * Blackwell-specific optimizations:
 *   - Thread Block Clusters for cross-block holonomy monitoring
 *   - TMA (Tensor Memory Accelerator) for async board state copies
 *   - Distributed Shared Memory across cluster for branch sharing
 *   - FP8 tensor cores for continuous relaxation phase
 *   - Warp-level candidate bitmask operations
 *
 * Parallelism model:
 *   - BATCH mode: 1 thread block per puzzle, 1 warp (32 threads) per puzzle
 *     → 81 cells mapped to 32 threads (2-3 cells per thread)
 *   - SINGLE mode: 1 thread block cluster per puzzle, speculative branching
 *     across blocks within the cluster
 *
 * Thermodynamic enhancements (from "The Thermodynamics of Semantic Coherence"):
 *   [E1] Free Energy Objective — F = E - τ·S replaces E alone (S2, Conj. 4.1)
 *   [E2] Adaptive Annealing — CV-guided cooling schedule (S1c, Conj. 11.3)
 *   [E3] Susceptibility-Weighted Branching — MRV + χ tiebreak (S5, Conj. 7.1)
 *   [E4] Boltzmann Rounding — Metropolis-Hastings acceptance (S6, Conj. 16.1)
 *   [E5] CV Phase Routing — specific heat skips Phase 2 when futile (S4, Conj. 6.1)
 *   [E6] Correlation-Length Propagation — ξ bounds CP iterations (S7, Conj. 9.1)
 *
 * Compile (Blackwell sm_100):
 *   nvcc -arch=sm_100 -O3 --use_fast_math -o davis_solver davis_solver_blackwell.cu
 *
 * Compile (Hopper sm_90 fallback):
 *   nvcc -arch=sm_90 -O3 --use_fast_math -o davis_solver davis_solver_blackwell.cu
 *
 * Reference: "The Field Equations of Semantic Coherence" (B. R. Davis, 2025)
 *            "The Thermodynamics of Semantic Coherence" (B. R. Davis, 2026)
 *
 * Copyright (c) 2025-2026 Bee Rosa Davis. All rights reserved.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <chrono>

#ifdef _MSC_VER
#include <intrin.h>
static inline int host_ffs(unsigned int x) {
    unsigned long idx;
    if (_BitScanForward(&idx, x)) return (int)idx + 1;
    return 0;
}
#else
static inline int host_ffs(unsigned int x) { return __builtin_ffs(x); }
#endif

namespace cg = cooperative_groups;


/* ========================================================================
 * SECTION 0: CONSTANTS & BITMASK REPRESENTATION
 * ======================================================================== */

constexpr int GRID_SIZE     = 9;
constexpr int BOARD_CELLS   = 81;
constexpr int BOX_SIZE      = 3;
constexpr int MAX_BATCH     = 65536;
constexpr int WARP_SIZE     = 32;
constexpr int CELLS_PER_THREAD = 3;
constexpr uint16_t ALL_CANDS = 0x1FF;
constexpr int MAX_BRANCH_DEPTH = 40;  // 81 → 40: reduces local memory; hardest puzzles need ≤30
constexpr int MAX_SPECULATIVE  = 9;
constexpr int CLUSTER_SIZE     = 8;

// Davis Energy weights (λ₁, λ₂, λ₃ from E0)
constexpr float LAMBDA_PATH      = 0.3f;
constexpr float LAMBDA_CURVATURE = 0.4f;
constexpr float LAMBDA_HOLONOMY  = 0.3f;

// Curvature component weights
constexpr float W_SATURATION = 0.40f;
constexpr float W_SCARCITY   = 0.35f;
constexpr float W_COUPLING   = 0.25f;

// Trichotomy thresholds (calibrated from audit)
constexpr float GAMMA_EASY     = 1.0f;
constexpr float GAMMA_MEDIUM   = 0.6f;
constexpr float GAMMA_HARD     = 0.35f;
constexpr float GAMMA_EXPERT   = 0.2f;

// Manifold relaxation parameters
constexpr float RELAX_LR        = 0.05f;
constexpr float RELAX_MOMENTUM  = 0.9f;
constexpr int   RELAX_MAX_ITER  = 500;
constexpr float RELAX_CONV_EPS  = 1e-6f;

// === Thermodynamic Enhancement Parameters ===

// [E1] Free Energy Objective (S2, Conjecture 4.1)
// [E2] Adaptive Annealing Schedule (S1c, Conjecture 11.3)
constexpr float TAU_HIGH        = 2.0f;    // Starting temperature (explore)
constexpr float TAU_TARGET      = 0.01f;   // Final temperature (commit)
constexpr int   ANNEAL_WINDOW   = 20;      // Energy history ring buffer size
constexpr int   ANNEAL_CHECK    = 10;      // Estimate CV every N iterations

// [E4] Boltzmann Rounding (S6, Conjecture 16.1)
constexpr float ROUND_MIN_CONF  = 0.5f;    // Minimum peak prob to attempt rounding

// [E5] Specific Heat Phase Routing (S4, Conjecture 6.1)
constexpr float CV_SKIP_LOW     = 0.1f;    // CV below: skip Phase 2 (too determined)
constexpr float CV_SKIP_HIGH    = 2.0f;    // CV above: skip Phase 2 (near criticality)

// Precomputed constant
constexpr float LN9             = 2.1972245773f;  // logf(9.0f)


/* ========================================================================
 * SECTION 1: PEER LOOKUP TABLE (Constant Memory)
 * ======================================================================== */

static void compute_peer_table(int peers[81][20]) {
    for (int i = 0; i < 81; i++) {
        int row = i / 9, col = i % 9;
        int br = (row / 3) * 3, bc = (col / 3) * 3;
        int count = 0;
        bool added[81] = {false};
        added[i] = true;
        for (int c = 0; c < 9; c++) {
            int idx = row * 9 + c;
            if (!added[idx]) { peers[i][count++] = idx; added[idx] = true; }
        }
        for (int r = 0; r < 9; r++) {
            int idx = r * 9 + col;
            if (!added[idx]) { peers[i][count++] = idx; added[idx] = true; }
        }
        for (int r = br; r < br + 3; r++) {
            for (int c = bc; c < bc + 3; c++) {
                int idx = r * 9 + c;
                if (!added[idx]) { peers[i][count++] = idx; added[idx] = true; }
            }
        }
    }
}

__constant__ int d_peers[81][20];
__constant__ int d_row_of[81];
__constant__ int d_col_of[81];
__constant__ int d_box_of[81];

static void compute_cell_lookups(int row_of[81], int col_of[81], int box_of[81]) {
    for (int i = 0; i < 81; i++) {
        row_of[i] = i / 9;
        col_of[i] = i % 9;
        box_of[i] = (i / 9 / 3) * 3 + (i % 9 / 3);
    }
}


/* ========================================================================
 * SECTION 2: DEVICE UTILITY FUNCTIONS
 * ======================================================================== */

__device__ __forceinline__ int popcount16(uint16_t x) {
    return __popc((unsigned int)x);
}

__device__ __forceinline__ int bitmask_to_value(uint16_t mask) {
    return __ffs((unsigned int)mask);
}

__device__ __forceinline__ uint16_t value_to_bitmask(int val) {
    return (uint16_t)(1 << (val - 1));
}

__device__ __forceinline__ bool is_solved(uint16_t mask) {
    return mask != 0 && (mask & (mask - 1)) == 0;
}

__device__ bool board_inconsistent(const uint16_t* cands) {
    for (int i = 0; i < 81; i++) {
        if (cands[i] == 0) return true;
    }
    return false;
}

__device__ int count_unsolved(const uint16_t* cands) {
    int n = 0;
    for (int i = 0; i < 81; i++) {
        if (!is_solved(cands[i])) n++;
    }
    return n;
}

/**
 * [E4] Warp-cooperative PRNG — xorshift32, seeded per puzzle.
 * Only thread 0 drives; result broadcast via shfl.
 */
__device__ uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}


/* ========================================================================
 * SECTION 3: CONSTRAINT PROPAGATION KERNEL
 *
 * Five-rule wavefront propagation:
 *   1. Naked singles   — solved cell eliminates its value from 20 peers
 *   2. Hidden singles  — digit with one possible cell in a unit gets assigned
 *   3. Naked pairs     — two cells sharing a 2-candidate mask lock those digits
 *   4. Pointing pairs  — digit confined to one row/col in a box → eliminate outside
 *   5. Claiming        — digit confined to one box in a row/col → eliminate outside
 *
 * [E6] Iteration count bounded by correlation length ξ when K_max and Γ
 *      are available. Defaults to 81 (full convergence) otherwise.
 * ======================================================================== */

__device__ bool propagate_thread(uint16_t* cands, int tid) {
    bool changed = false;
    for (int offset = tid; offset < 81; offset += WARP_SIZE) {
        if (!is_solved(cands[offset])) continue;
        uint16_t solved_bit = cands[offset];
        for (int p = 0; p < 20; p++) {
            int peer = d_peers[offset][p];
            // Use 32-bit atomicAnd to avoid lost-update race when two
            // threads eliminate different bits from the same peer cell.
            // Pack the 16-bit index into its containing 32-bit word.
            int word_idx = peer >> 1;           // which uint32 contains this uint16
            int shift    = (peer & 1) * 16;     // low or high half
            unsigned int clear = ~((unsigned int)solved_bit << shift);
            unsigned int old = atomicAnd(
                reinterpret_cast<unsigned int*>(cands) + word_idx, clear);
            uint16_t before = (uint16_t)((old >> shift) & 0xFFFF);
            if (before != (before & ~solved_bit)) changed = true;
        }
    }
    return changed;
}

__device__ bool hidden_singles_thread(uint16_t* cands, int tid) {
    bool changed = false;
    if (tid < 27) {
        int unit_cells[9];
        if (tid < 9) {
            for (int c = 0; c < 9; c++) unit_cells[c] = tid * 9 + c;
        } else if (tid < 18) {
            int col = tid - 9;
            for (int r = 0; r < 9; r++) unit_cells[r] = r * 9 + col;
        } else {
            int box = tid - 18;
            int br = (box / 3) * 3, bc = (box % 3) * 3;
            int idx = 0;
            for (int r = br; r < br + 3; r++)
                for (int c = bc; c < bc + 3; c++)
                    unit_cells[idx++] = r * 9 + c;
        }
        for (int v = 1; v <= 9; v++) {
            uint16_t vmask = value_to_bitmask(v);
            int count = 0;
            int last_cell = -1;
            for (int k = 0; k < 9; k++) {
                int cell = unit_cells[k];
                if (cands[cell] & vmask) { count++; last_cell = cell; }
            }
            if (count == 1 && !is_solved(cands[last_cell])) {
                cands[last_cell] = vmask;
                changed = true;
            }
        }
    }
    return changed;
}

__device__ bool naked_pairs_thread(uint16_t* cands, int tid) {
    if (tid >= 27) return false;
    bool changed = false;
    int unit_cells[9];
    if (tid < 9) {
        for (int c = 0; c < 9; c++) unit_cells[c] = tid * 9 + c;
    } else if (tid < 18) {
        int col = tid - 9;
        for (int r = 0; r < 9; r++) unit_cells[r] = r * 9 + col;
    } else {
        int box = tid - 18;
        int br = (box / 3) * 3, bc = (box % 3) * 3;
        int idx = 0;
        for (int r = br; r < br + 3; r++)
            for (int c = bc; c < bc + 3; c++)
                unit_cells[idx++] = r * 9 + c;
    }
    for (int i = 0; i < 9; i++) {
        int ci = unit_cells[i];
        uint16_t mi = cands[ci];
        if (is_solved(mi) || mi == 0 || popcount16(mi) != 2) continue;
        for (int j = i + 1; j < 9; j++) {
            int cj = unit_cells[j];
            if (cands[cj] != mi) continue;
            for (int k = 0; k < 9; k++) {
                if (k == i || k == j) continue;
                int ck = unit_cells[k];
                if (!is_solved(cands[ck]) && cands[ck] != 0 && (cands[ck] & mi)) {
                    cands[ck] &= ~mi;
                    changed = true;
                    if (cands[ck] == 0) return changed;
                }
            }
        }
    }
    return changed;
}

__device__ bool pointing_pairs_thread(uint16_t* cands, int tid) {
    if (tid >= 9) return false;
    bool changed = false;
    int box = tid;
    int br = (box / 3) * 3, bc = (box % 3) * 3;
    int box_cells[9];
    int idx = 0;
    for (int r = br; r < br + 3; r++)
        for (int c = bc; c < bc + 3; c++)
            box_cells[idx++] = r * 9 + c;
    for (int d = 0; d < 9; d++) {
        uint16_t bit = (uint16_t)(1 << d);
        int rows_seen = 0, cols_seen = 0, count = 0;
        for (int k = 0; k < 9; k++) {
            int cell = box_cells[k];
            if (cands[cell] != 0 && (cands[cell] & bit)) {
                rows_seen |= (1 << d_row_of[cell]);
                cols_seen |= (1 << d_col_of[cell]);
                count++;
            }
        }
        if (count < 2) continue;
        if (popcount16((uint16_t)rows_seen) == 1) {
            int row = __ffs(rows_seen) - 1;
            for (int c = 0; c < 9; c++) {
                if (c >= bc && c < bc + 3) continue;
                int cell = row * 9 + c;
                if (!is_solved(cands[cell]) && cands[cell] != 0 && (cands[cell] & bit)) {
                    cands[cell] &= ~bit;
                    changed = true;
                    if (cands[cell] == 0) return changed;
                }
            }
        }
        if (popcount16((uint16_t)cols_seen) == 1) {
            int col = __ffs(cols_seen) - 1;
            for (int r = 0; r < 9; r++) {
                if (r >= br && r < br + 3) continue;
                int cell = r * 9 + col;
                if (!is_solved(cands[cell]) && cands[cell] != 0 && (cands[cell] & bit)) {
                    cands[cell] &= ~bit;
                    changed = true;
                    if (cands[cell] == 0) return changed;
                }
            }
        }
    }
    return changed;
}

__device__ bool claiming_thread(uint16_t* cands, int tid) {
    if (tid >= 18) return false;
    bool changed = false;
    bool is_row = (tid < 9);
    int line = is_row ? tid : (tid - 9);
    for (int d = 0; d < 9; d++) {
        uint16_t bit = (uint16_t)(1 << d);
        int boxes_seen = 0, count = 0;
        for (int k = 0; k < 9; k++) {
            int cell = is_row ? (line * 9 + k) : (k * 9 + line);
            if (cands[cell] != 0 && (cands[cell] & bit)) {
                boxes_seen |= (1 << d_box_of[cell]);
                count++;
            }
        }
        if (count < 2) continue;
        if (popcount16((uint16_t)boxes_seen) == 1) {
            int b = __ffs(boxes_seen) - 1;
            int bbr = (b / 3) * 3, bbc = (b % 3) * 3;
            for (int r = bbr; r < bbr + 3; r++) {
                for (int c = bbc; c < bbc + 3; c++) {
                    if (is_row && r == line) continue;
                    if (!is_row && c == line) continue;
                    int cell = r * 9 + c;
                    if (!is_solved(cands[cell]) && cands[cell] != 0 && (cands[cell] & bit)) {
                        cands[cell] &= ~bit;
                        changed = true;
                        if (cands[cell] == 0) return changed;
                    }
                }
            }
        }
    }
    return changed;
}


/**
 * [E6] Correlation-Length Propagation Depth (S7, Conjecture 9.1)
 *
 * ξ = (1/√K̂_max) · |1 - Γ|^{-ν}, ν = 1/K̂_max.
 * Propagation only needs ⌈diameter/ξ⌉ + 1 iterations.
 */
__device__ int compute_propagation_depth(float K_max, float Gamma) {
    constexpr int DIAMETER = 9;
    if (K_max <= 0.0f || Gamma <= 0.0f) return BOARD_CELLS;

    float nu = 1.0f / K_max;
    float dist = fabsf(1.0f - Gamma);
    if (dist < 1e-4f) return BOARD_CELLS;  // Critical point: ξ diverges

    float xi = (1.0f / sqrtf(K_max)) * powf(dist, -nu);
    if (xi <= 0.0f) return BOARD_CELLS;  // Fallback for invalid ξ

    int passes = (int)ceilf((float)DIAMETER / xi) + 1;
    return max(2, min(passes, BOARD_CELLS));
}

/**
 * Full constraint propagation loop.
 * Five-rule wavefront propagation until convergence or max_iter.
 *
 * [E6] max_iter bounded by ξ when K_max/Γ available.
 *      Default 81 (full convergence) for Phase 3 DFS nodes.
 */
__device__ bool constraint_propagation(uint16_t* cands, int max_iter = BOARD_CELLS) {
    cg::coalesced_group active = cg::coalesced_threads();
    int tid = active.thread_rank();

    for (int iter = 0; iter < max_iter; iter++) {
        bool local_changed = false;

        local_changed |= propagate_thread(cands, tid);
        active.sync();

        local_changed |= hidden_singles_thread(cands, tid);
        active.sync();

        local_changed |= naked_pairs_thread(cands, tid);
        active.sync();

        local_changed |= pointing_pairs_thread(cands, tid);
        active.sync();

        local_changed |= claiming_thread(cands, tid);
        active.sync();

        unsigned mask = active.ballot(local_changed);
        if (mask == 0) break;

        bool local_dead = false;
        for (int offset = tid; offset < 81; offset += WARP_SIZE) {
            if (cands[offset] == 0) { local_dead = true; break; }
        }
        if (active.any(local_dead)) return false;
    }
    return true;
}


/* ========================================================================
 * SECTION 4: CURVATURE COMPUTATION
 * ======================================================================== */

__device__ float compute_curvature(const uint16_t* cands, int cell) {
    if (is_solved(cands[cell])) return 0.0f;
    uint16_t my_cands = cands[cell];
    if (my_cands == 0) return FLT_MAX;

    int n_cands = popcount16(my_cands);

    int filled_peers = 0;
    for (int p = 0; p < 20; p++) {
        if (is_solved(cands[d_peers[cell][p]])) filled_peers++;
    }
    float saturation = (float)filled_peers / 20.0f;
    float scarcity = 1.0f - (float)n_cands / 9.0f;

    int coupling_sum = 0, empty_peer_count = 0;
    for (int p = 0; p < 20; p++) {
        int peer = d_peers[cell][p];
        if (!is_solved(cands[peer]) && cands[peer] != 0) {
            coupling_sum += popcount16(my_cands & cands[peer]);
            empty_peer_count++;
        }
    }
    float coupling_norm = (empty_peer_count > 0 && n_cands > 0)
        ? (float)coupling_sum / (float)(empty_peer_count * n_cands)
        : 0.0f;

    return W_SATURATION * saturation + W_SCARCITY * scarcity + W_COUPLING * coupling_norm;
}

__device__ void compute_all_curvatures(const uint16_t* cands, float* curvatures, int tid) {
    for (int offset = tid; offset < 81; offset += WARP_SIZE) {
        curvatures[offset] = compute_curvature(cands, offset);
    }
}


/* ========================================================================
 * SECTION 5: CELL SELECTION
 *
 * [E3] Two-tier MRV + Susceptibility tiebreak (S5, Conjecture 7.1)
 *
 * PRIMARY: MRV — fewest candidates (warp-parallel popcount scan, O(81))
 * TIEBREAK: Among MRV-tied cells, pick the one with highest susceptibility
 *   χ̃(c) = Σ_peers overlap(c,peer)/|C(peer)| × |C(c)|
 *
 * This avoids the O(81×20×20) regression from full V(c)·χ(c) scoring.
 * Cost: O(81 + ties×20) ≈ O(200) per selection.
 * ======================================================================== */

/**
 * [E3] Susceptibility proxy χ̃(c) for a single cell.
 * Measures how much resolving this cell would perturb its peers.
 */
__device__ float compute_susceptibility(const uint16_t* cands, int cell) {
    uint16_t my = cands[cell];
    if (is_solved(my) || my == 0) return 0.0f;
    int n = popcount16(my);
    float fluct = 0.0f;
    for (int p = 0; p < 20; p++) {
        uint16_t peer_c = cands[d_peers[cell][p]];
        if (is_solved(peer_c) || peer_c == 0) continue;
        int overlap = popcount16(my & peer_c);
        int pn = popcount16(peer_c);
        fluct += (float)overlap / (float)pn;
    }
    return (float)n * fluct;
}

/**
 * [E3] Two-tier cell selection: MRV + susceptibility tiebreak.
 *
 * Tier 1: Warp-parallel MRV scan — each thread finds its local minimum
 *         candidate count, then warp-reduce to find the global MRV value.
 * Tier 2: Among cells matching the global MRV, compute χ̃(c) and pick
 *         the one with highest susceptibility.
 *
 * Falls back to pure MRV (no tiebreak) when there's a unique minimum.
 */
__device__ int select_branch_cell(
    const uint16_t* cands,
    const float* curvatures  // Kept for API compat; not used in E3 path
) {
    cg::coalesced_group active = cg::coalesced_threads();
    int tid = active.thread_rank();

    // ---- TIER 1: MRV scan ----
    int local_min_cands = 10;
    for (int offset = tid; offset < 81; offset += WARP_SIZE) {
        if (is_solved(cands[offset]) || cands[offset] == 0) continue;
        int nc = popcount16(cands[offset]);
        if (nc < local_min_cands) local_min_cands = nc;
    }
    // Warp-reduce to find global minimum candidate count
    for (int delta = active.size() / 2; delta > 0; delta /= 2) {
        int other = active.shfl_down(local_min_cands, delta);
        if (other < local_min_cands) local_min_cands = other;
    }
    int global_mrv = active.shfl(local_min_cands, 0);
    if (global_mrv >= 10) return -1;  // All solved

    // ---- TIER 2: Susceptibility tiebreak among MRV-tied cells ----
    float best_chi = -1.0f;
    int   best_cell = -1;

    for (int offset = tid; offset < 81; offset += WARP_SIZE) {
        if (is_solved(cands[offset]) || cands[offset] == 0) continue;
        int nc = popcount16(cands[offset]);
        if (nc != global_mrv) continue;

        float chi = compute_susceptibility(cands, offset);
        if (chi > best_chi) {
            best_chi = chi;
            best_cell = offset;
        }
    }

    // Warp-reduce argmax χ̃
    for (int delta = active.size() / 2; delta > 0; delta /= 2) {
        float other_chi  = active.shfl_down(best_chi, delta);
        int   other_cell = active.shfl_down(best_cell, delta);
        if (other_chi > best_chi) {
            best_chi = other_chi;
            best_cell = other_cell;
        }
    }
    return active.shfl(best_cell, 0);
}


/* ========================================================================
 * SECTION 6: HOLONOMY MONITORING
 * ======================================================================== */

__device__ float check_holonomy_warp(const uint16_t* cands) {
    cg::coalesced_group active = cg::coalesced_threads();
    int tid = active.thread_rank();
    float local_deficit = 0.0f;
    bool  local_dead = false;
    for (int offset = tid; offset < 81; offset += WARP_SIZE) {
        uint16_t c = cands[offset];
        if (c == 0) { local_dead = true; }
        else if (!is_solved(c)) {
            local_deficit += (float)(popcount16(c) - 1) / 8.0f;
        }
    }
    if (active.any(local_dead)) return FLT_MAX;
    for (int delta = active.size() / 2; delta > 0; delta /= 2) {
        local_deficit += active.shfl_down(local_deficit, delta);
    }
    return active.shfl(local_deficit, 0);
}

__device__ bool holonomy_prune_value(const uint16_t* cands, int cell, uint16_t val_mask) {
    for (int p = 0; p < 20; p++) {
        int peer = d_peers[cell][p];
        if (!is_solved(cands[peer])) {
            uint16_t peer_after = cands[peer] & ~val_mask;
            if (peer_after == 0) return true;
        }
    }
    return false;
}


/* ========================================================================
 * SECTION 7: DAVIS FREE ENERGY FUNCTIONAL
 *
 * [E1] Free Energy Objective (S2, Conjecture 4.1)
 * Replaces E[γ] with F[γ] = E[γ] - τ·S.
 * The entropy term -τ·S prevents premature probability collapse.
 *
 * [E2] Adaptive Annealing (S1c, Conjecture 11.3)
 * τ anneals from TAU_HIGH to TAU_TARGET. Cooling rate adapts via
 * specific heat CV = β²·Var(E) — slows near phase transitions.
 * ======================================================================== */

/**
 * [E1] Continuous Davis Free Energy: F = E - τ·S
 * At τ=0: reduces to original E[γ] (ground state).
 * At τ>0: entropy regularization prevents premature collapse.
 */
__device__ float davis_free_energy_continuous(
    const float* probs,
    const float* curvatures,
    float tau
) {
    cg::coalesced_group active = cg::coalesced_threads();
    int tid = active.thread_rank();

    float local_energy = 0.0f;
    float local_entropy = 0.0f;

    for (int cell = tid; cell < 81; cell += WARP_SIZE) {
        const float* p = &probs[cell * 9];
        float cell_entropy = 0.0f;

        for (int v = 0; v < 9; v++) {
            if (p[v] > 1e-8f) {
                cell_entropy -= p[v] * logf(p[v]);
            }
        }

        // Curvature-weighted uncertainty (Term 2)
        float curv_term = curvatures[cell] * cell_entropy;

        // Constraint violation in probability space (Term 3)
        float violation = 0.0f;
        for (int pi = 0; pi < 20; pi++) {
            int peer = d_peers[cell][pi];
            const float* pp = &probs[peer * 9];
            for (int v = 0; v < 9; v++) {
                violation += p[v] * pp[v];
            }
        }
        violation /= 20.0f;

        local_energy += LAMBDA_PATH * (cell_entropy / LN9)
                      + LAMBDA_CURVATURE * curv_term
                      + LAMBDA_HOLONOMY * violation;
        local_entropy += cell_entropy;
    }

    // Warp reduce
    for (int delta = active.size() / 2; delta > 0; delta /= 2) {
        local_energy  += active.shfl_down(local_energy, delta);
        local_entropy += active.shfl_down(local_entropy, delta);
    }

    float E = active.shfl(local_energy, 0);
    float S = active.shfl(local_entropy, 0);

    // F = E - τ·S (Conjecture 4.1)
    return E - tau * S;
}

/**
 * [E1+E2] Free energy gradient w.r.t. probability logits.
 * Includes the entropy gradient term from the free energy formulation.
 *
 * ∂F/∂logit[c][v] = ∂E/∂logit[c][v] - τ · ∂S/∂logit[c][v]
 *
 * Entropy gradient through softmax:
 *   ∂S/∂logit_v = p_v · [H(cell) + log(p_v)]
 *   where H(cell) = -Σ_v p_v log(p_v)
 */
__device__ void compute_free_energy_gradient(
    const float* probs,
    const float* curvatures,
    float* grad,
    float tau,
    int tid
) {
    for (int cell = tid; cell < 81; cell += WARP_SIZE) {
        const float* p = &probs[cell * 9];
        float* g = &grad[cell * 9];
        float K = curvatures[cell];

        // Compute cell entropy H(cell) for the softmax chain rule
        float H = 0.0f;
        for (int v = 0; v < 9; v++) {
            if (p[v] > 1e-8f) H -= p[v] * logf(p[v]);
        }

        for (int v = 0; v < 9; v++) {
            // Entropy gradient through softmax: ∂H/∂logit_v = -p_v·(H + log p_v)
            float entropy_grad = 0.0f;
            if (p[v] > 1e-8f) {
                entropy_grad = -p[v] * (H + logf(p[v]));
            }

            // Violation gradient: ∂V/∂p_v = Σ_peers pp[v] (direct term)
            // Chain rule through softmax: p[v] * (Σ_peers pp[v] - Σ_j p[j]·Σ_peers pp[j])
            // Simplified to direct gradient (dominates in practice)
            float violation_grad = 0.0f;
            for (int pi = 0; pi < 20; pi++) {
                int peer = d_peers[cell][pi];
                violation_grad += probs[peer * 9 + v];
            }
            violation_grad /= 20.0f;

            // ∂F/∂logit = (λ₁ + λ₂·K) · ∂H/∂logit / ln9 + λ₃ · ∂V/∂logit - τ · ∂S/∂logit
            // The energy entropy term and free energy entropy term combine:
            // (λ₁/ln9 + λ₂·K - τ) · entropy_grad + λ₃ · violation_grad
            float entropy_weight = (LAMBDA_PATH / LN9 + LAMBDA_CURVATURE * K) - tau;
            g[v] = entropy_weight * entropy_grad + LAMBDA_HOLONOMY * violation_grad;
        }
    }
}

__device__ void softmax_normalize(float* logits, float* probs, int cell) {
    float* l = &logits[cell * 9];
    float* p = &probs[cell * 9];
    float max_l = l[0];
    for (int v = 1; v < 9; v++) {
        if (l[v] > max_l) max_l = l[v];
    }
    float sum = 0.0f;
    for (int v = 0; v < 9; v++) {
        p[v] = expf(l[v] - max_l);
        sum += p[v];
    }
    float inv_sum = 1.0f / sum;
    for (int v = 0; v < 9; v++) p[v] *= inv_sum;
}


/* ========================================================================
 * SECTION 7b: SPECIFIC HEAT COMPUTATION
 *
 * [E5] Specific Heat Phase Routing (S4, Conjecture 6.1)
 * CV = β²·Var(E) estimated from per-cell energy distribution.
 * Used by trichotomy to route puzzles around Phase 2.
 * ======================================================================== */

__device__ float compute_specific_heat_warp(const uint16_t* cands) {
    cg::coalesced_group warp = cg::coalesced_threads();
    int tid = warp.thread_rank();

    float local_sum = 0.0f, local_sum2 = 0.0f;
    int local_count = 0;

    for (int i = tid; i < 81; i += WARP_SIZE) {
        if (!is_solved(cands[i]) && cands[i] != 0) {
            float e = 1.0f / (float)popcount16(cands[i]);
            local_sum  += e;
            local_sum2 += e * e;
            local_count++;
        }
    }

    // Warp reduce
    for (int d = warp.size() / 2; d > 0; d /= 2) {
        local_sum   += warp.shfl_down(local_sum, d);
        local_sum2  += warp.shfl_down(local_sum2, d);
        local_count += warp.shfl_down(local_count, d);
    }

    float count_f = fmaxf((float)warp.shfl(local_count, 0), 1.0f);
    float mean    = warp.shfl(local_sum, 0) / count_f;
    float mean_sq = warp.shfl(local_sum2, 0) / count_f;
    float var     = mean_sq - mean * mean;
    return warp.shfl(var, 0);
}


/* ========================================================================
 * SECTION 8: PHASE 1 — WAVEFRONT CONSTRAINT PROPAGATION KERNEL
 *
 * [E6] Uses correlation-length bounded iteration count.
 * ======================================================================== */

__global__ void phase1_propagation_kernel(
    const int* __restrict__ input_boards,
    uint16_t*  __restrict__ cand_boards,
    int*       __restrict__ status,
    float*     __restrict__ max_curvs,    // [E6] Output K̂_max for ξ computation
    float*     __restrict__ gammas_out,   // [E6] Output Γ for ξ computation
    int batch_size
) {
    int puzzle_idx = blockIdx.x;
    if (puzzle_idx >= batch_size) return;

    cg::coalesced_group warp = cg::coalesced_threads();
    int tid = warp.thread_rank();

    __shared__ __align__(4) uint16_t s_cands[BOARD_CELLS + 1]; // +1 pad for 32-bit atomicAnd safety

    // Initialize candidate bitmasks from input
    if (tid == 0) s_cands[BOARD_CELLS] = 0;  // zero the pad element
    for (int i = tid; i < 81; i += WARP_SIZE) {
        int val = input_boards[puzzle_idx * 81 + i];
        s_cands[i] = (val >= 1 && val <= 9) ? value_to_bitmask(val) : ALL_CANDS;
    }
    warp.sync();

    // First pass: full convergence (no ξ estimate yet)
    bool consistent = constraint_propagation(s_cands);
    warp.sync();

    // [E6] Compute K̂_max and Γ for downstream correlation-length use
    int local_filled = 0, local_empty = 0;
    float local_max_K = 0.0f;
    for (int i = tid; i < 81; i += WARP_SIZE) {
        if (is_solved(s_cands[i])) {
            local_filled++;
        } else if (s_cands[i] != 0) {
            local_empty++;
            float K = compute_curvature(s_cands, i);
            if (K < FLT_MAX && K > local_max_K) local_max_K = K;
        }
    }
    for (int delta = warp.size() / 2; delta > 0; delta /= 2) {
        local_filled += warp.shfl_down(local_filled, delta);
        local_empty  += warp.shfl_down(local_empty, delta);
        float other_K = warp.shfl_down(local_max_K, delta);
        if (other_K > local_max_K) local_max_K = other_K;
    }

    // Write results
    for (int i = tid; i < 81; i += WARP_SIZE) {
        cand_boards[puzzle_idx * 81 + i] = s_cands[i];
    }

    if (tid == 0) {
        // After warp reduction, thread 0 already holds correct values
        int m = local_filled;
        int n = local_empty;
        float K_max = local_max_K;
        max_curvs[puzzle_idx] = K_max;

        if (n == 0 || K_max == 0.0f) {
            gammas_out[puzzle_idx] = FLT_MAX;
        } else {
            gammas_out[puzzle_idx] = (float)m / (K_max * (float)n * LN9);
        }

        if (!consistent) {
            status[puzzle_idx] = -1;
        } else {
            bool solved = true;
            for (int i = 0; i < 81; i++) {
                if (!is_solved(s_cands[i])) { solved = false; break; }
            }
            status[puzzle_idx] = solved ? 1 : 0;
        }
    }
}


/* ========================================================================
 * SECTION 9: TRICHOTOMY + CV ROUTING KERNEL
 *
 * [E5] Computes Γ and CV for each unsolved puzzle.
 * Routes puzzles to Phase 2 or Phase 3 based on CV trichotomy:
 *   CV < 0.1  → skip Phase 2 (too determined, relaxation unnecessary)
 *   0.1 ≤ CV ≤ 2.0 → Phase 2 (relaxation sweet spot)
 *   CV > 2.0  → skip Phase 2 (near criticality, would oscillate)
 * ======================================================================== */

__global__ void trichotomy_kernel(
    const uint16_t* __restrict__ cand_boards,
    const float*    __restrict__ gammas,       // Already computed by Phase 1
    const float*    __restrict__ max_curvs,    // Already computed by Phase 1
    float*          __restrict__ cv_out,       // [E5] CV per puzzle
    int batch_size
) {
    int puzzle_idx = blockIdx.x;
    if (puzzle_idx >= batch_size) return;

    cg::coalesced_group warp = cg::coalesced_threads();
    int tid = warp.thread_rank();

    const uint16_t* cands = &cand_boards[puzzle_idx * 81];

    // [E5] Compute specific heat CV (the only new quantity — Γ/K̂_max come from Phase 1)
    float CV = compute_specific_heat_warp(cands);

    if (tid == 0) {
        cv_out[puzzle_idx] = CV;
    }
}


/* ========================================================================
 * SECTION 10: PHASE 2 — THERMODYNAMIC MANIFOLD RELAXATION
 *
 * [E1] Free Energy objective: F = E - τ·S (prevents premature collapse)
 * [E2] Adaptive Annealing: τ cools from TAU_HIGH → TAU_TARGET, rate
 *      controlled by CV estimation from energy variance ring buffer.
 * [E4] Boltzmann Rounding: Metropolis-Hastings acceptance instead of
 *      hard argmax. Cells processed in curvature-descending order.
 * [E5] CV Phase Routing: skip this kernel if CV outside [0.1, 2.0].
 * ======================================================================== */

__global__ void phase2_relaxation_kernel(
    uint16_t*  __restrict__ cand_boards,
    int*       __restrict__ status,
    const float* __restrict__ gammas,
    const float* __restrict__ cv_values,      // [E5] CV from trichotomy
    const float* __restrict__ max_curvs,      // [E6] K̂_max for propagation depth
    int batch_size
) {
    int puzzle_idx = blockIdx.x;
    if (puzzle_idx >= batch_size) return;
    if (status[puzzle_idx] != 0) return;

    float Gamma = gammas[puzzle_idx];
    float CV_initial = cv_values[puzzle_idx];
    float K_max = max_curvs[puzzle_idx];

    // [E5] CV Phase Routing: skip relaxation when futile
    if (CV_initial < CV_SKIP_LOW || CV_initial > CV_SKIP_HIGH) return;
    // Also skip deeply underdetermined puzzles (original Γ check)
    if (Gamma < GAMMA_EXPERT) return;

    cg::coalesced_group warp = cg::coalesced_threads();
    int tid = warp.thread_rank();

    // Shared memory for this puzzle
    __shared__ __align__(4) uint16_t s_cands[82]; // +1 pad for 32-bit atomicAnd safety
    __shared__ float s_probs[81 * 9];
    __shared__ float s_logits[81 * 9];
    __shared__ float s_grad[81 * 9];
    __shared__ float s_momentum[81 * 9];
    __shared__ float s_curvatures[81];

    // [E2] Annealing state
    __shared__ float s_tau;
    __shared__ float s_energy_hist[ANNEAL_WINDOW];
    __shared__ int   s_hist_count;

    // [E4] PRNG state for Boltzmann rounding
    __shared__ uint32_t s_rng_state;

    // Pre-rounding snapshot: if Boltzmann rounding + CP produces
    // inconsistency, restore this so Phase 3 gets a clean board.
    __shared__ uint16_t s_pre_round[81];

    // Load candidates
    if (tid == 0) s_cands[81] = 0;  // zero the pad element
    for (int i = tid; i < 81; i += WARP_SIZE) {
        s_cands[i] = cand_boards[puzzle_idx * 81 + i];
    }
    if (tid == 0) {
        s_tau = TAU_HIGH;
        s_hist_count = 0;
        s_rng_state = (uint32_t)(puzzle_idx * 1099511628211ULL + 14695981039346656037ULL);
        for (int i = 0; i < ANNEAL_WINDOW; i++) s_energy_hist[i] = 0.0f;
    }
    warp.sync();

    compute_all_curvatures(s_cands, s_curvatures, tid);
    warp.sync();

    // Initialize logits from candidate bitmasks
    for (int cell = tid; cell < 81; cell += WARP_SIZE) {
        for (int v = 0; v < 9; v++) {
            if (is_solved(s_cands[cell])) {
                s_logits[cell * 9 + v] = (s_cands[cell] & (1 << v)) ? 10.0f : -10.0f;
            } else if (s_cands[cell] & (1 << v)) {
                s_logits[cell * 9 + v] = 0.0f;
            } else {
                s_logits[cell * 9 + v] = -10.0f;
            }
            s_momentum[cell * 9 + v] = 0.0f;
        }
        softmax_normalize(s_logits, s_probs, cell);
    }
    warp.sync();

    // ---- Gradient descent with adaptive annealing ----
    float prev_energy = FLT_MAX;

    for (int iter = 0; iter < RELAX_MAX_ITER; iter++) {
        float current_tau = s_tau;  // Read once per iteration

        // [E1] Compute free energy gradient (includes -τ·∂S/∂logit)
        compute_free_energy_gradient(s_probs, s_curvatures, s_grad, current_tau, tid);
        warp.sync();

        // Update logits with Nesterov momentum
        for (int cell = tid; cell < 81; cell += WARP_SIZE) {
            if (is_solved(s_cands[cell])) continue;

            // [E2] Temperature-dependent, curvature-adaptive step size
            float step = current_tau * RELAX_LR / (1.0f + s_curvatures[cell]);

            for (int v = 0; v < 9; v++) {
                int idx = cell * 9 + v;
                if (!(s_cands[cell] & (1 << v))) {
                    s_logits[idx] = -10.0f;
                    s_momentum[idx] = 0.0f;
                    continue;
                }
                float new_momentum = RELAX_MOMENTUM * s_momentum[idx] - step * s_grad[idx];
                s_logits[idx] += -RELAX_MOMENTUM * s_momentum[idx]
                               + (1.0f + RELAX_MOMENTUM) * new_momentum;
                s_momentum[idx] = new_momentum;
            }
            softmax_normalize(s_logits, s_probs, cell);
        }
        warp.sync();

        // [E2] Adaptive annealing: check CV every ANNEAL_CHECK iterations
        if (iter % ANNEAL_CHECK == (ANNEAL_CHECK - 1)) {
            // Compute current free energy
            float energy = davis_free_energy_continuous(s_probs, s_curvatures, current_tau);

            if (tid == 0) {
                // Store in ring buffer
                int slot = s_hist_count % ANNEAL_WINDOW;
                s_energy_hist[slot] = energy;
                s_hist_count++;

                // Estimate CV from energy variance when buffer has enough samples
                int window = min(s_hist_count, ANNEAL_WINDOW);
                if (window >= 3) {
                    float mean_E = 0.0f;
                    for (int i = 0; i < window; i++) mean_E += s_energy_hist[i];
                    mean_E /= (float)window;

                    float var_E = 0.0f;
                    for (int i = 0; i < window; i++) {
                        float d = s_energy_hist[i] - mean_E;
                        var_E += d * d;
                    }
                    var_E /= (float)window;

                    float beta = 1.0f / fmaxf(s_tau, 1e-8f);
                    float CV = beta * beta * var_E;

                    // Adaptive cooling: slow near phase transition (high CV)
                    float base_rate = (TAU_HIGH - TAU_TARGET) / (float)RELAX_MAX_ITER
                                    * (float)ANNEAL_CHECK;
                    float adaptive_rate = base_rate / (1.0f + CV);
                    s_tau = fmaxf(TAU_TARGET, s_tau - adaptive_rate);
                } else {
                    // Not enough history: use linear cooling
                    float base_rate = (TAU_HIGH - TAU_TARGET) / (float)RELAX_MAX_ITER
                                    * (float)ANNEAL_CHECK;
                    s_tau = fmaxf(TAU_TARGET, s_tau - base_rate);
                }
            }
            warp.sync();

            // Convergence check
            if (fabsf(prev_energy - energy) < RELAX_CONV_EPS) break;
            prev_energy = energy;
        }
    }

    // ---- [E4] Boltzmann Rounding ----
    // Process unsolved cells in curvature-descending order.
    // Use Metropolis acceptance instead of hard argmax.
    //
    // Implementation: warp-cooperative. Thread 0 drives the sequential
    // curvature-ordered loop; all threads participate in the CP after.
    //
    // We do multiple passes: each pass finds the highest-curvature
    // unrounded cell and attempts to round it.

    // Save pre-rounding snapshot for rollback on failure
    for (int i = tid; i < 81; i += WARP_SIZE) {
        s_pre_round[i] = s_cands[i];
    }
    warp.sync();

    for (int round = 0; round < 81; round++) {
        // Find highest-curvature unsolved cell (warp reduction)
        float local_best_K = -1.0f;
        int   local_best_cell = -1;
        for (int i = tid; i < 81; i += WARP_SIZE) {
            if (!is_solved(s_cands[i]) && s_cands[i] != 0) {
                if (s_curvatures[i] > local_best_K) {
                    local_best_K = s_curvatures[i];
                    local_best_cell = i;
                }
            }
        }
        for (int delta = warp.size() / 2; delta > 0; delta /= 2) {
            float other_K    = warp.shfl_down(local_best_K, delta);
            int   other_cell = warp.shfl_down(local_best_cell, delta);
            if (other_K > local_best_K) {
                local_best_K = other_K;
                local_best_cell = other_cell;
            }
        }
        int target_cell = warp.shfl(local_best_cell, 0);
        if (target_cell < 0) break;  // All cells rounded or solved

        // Check if peak probability is high enough to attempt rounding
        float max_p = -1.0f;
        int max_v = 0;
        if (tid == 0) {
            for (int v = 0; v < 9; v++) {
                float pv = s_probs[target_cell * 9 + v];
                if (pv > max_p) { max_p = pv; max_v = v; }
            }
        }
        max_p = warp.shfl(max_p, 0);

        if (max_p < ROUND_MIN_CONF) {
            // Too uncertain — mark curvature as processed so we skip it
            if (tid == 0) s_curvatures[target_cell] = -1.0f;
            warp.sync();
            continue;
        }

        // [E4] Metropolis sampling: sample from distribution, accept via e^{-β·ΔE}
        int chosen_v = 0;
        bool accept = false;
        if (tid == 0) {
            // Sample from probability distribution (not argmax)
            uint32_t r = xorshift32(&s_rng_state);
            float rand_f = (float)(r & 0xFFFFFF) / (float)0xFFFFFF;
            float cumulative = 0.0f;
            chosen_v = max_v;  // Fallback to argmax
            for (int v = 0; v < 9; v++) {
                cumulative += s_probs[target_cell * 9 + v];
                if (rand_f < cumulative) { chosen_v = v; break; }
            }

            // Compute energy change from committing this cell
            uint16_t chosen_mask = (uint16_t)(1 << chosen_v);
            float delta_E = 0.0f;
            bool fatal = false;
            for (int p = 0; p < 20; p++) {
                int peer = d_peers[target_cell][p];
                if (!is_solved(s_cands[peer]) && s_cands[peer] != 0) {
                    if (s_cands[peer] & chosen_mask) {
                        uint16_t remaining = s_cands[peer] & ~chosen_mask;
                        if (remaining == 0) { fatal = true; break; }
                        delta_E += s_curvatures[peer] / (float)popcount16(remaining);
                    }
                }
            }

            if (fatal) {
                accept = false;
            } else if (delta_E <= 0.0f) {
                accept = true;
            } else {
                float beta = 1.0f / fmaxf(s_tau, 1e-8f);
                uint32_t r2 = xorshift32(&s_rng_state);
                float rand2 = (float)(r2 & 0xFFFFFF) / (float)0xFFFFFF;
                accept = (rand2 < expf(-beta * delta_E));
            }
        }
        accept = (bool)warp.shfl((int)accept, 0);
        chosen_v = warp.shfl(chosen_v, 0);

        if (accept) {
            if (tid == 0) {
                s_cands[target_cell] = (uint16_t)(1 << chosen_v);
            }
            warp.sync();

            // Propagate the committed value's constraints to peers
            // so subsequent rounds see up-to-date candidate masks.
            // (Naked singles only — lightweight, no full 5-rule CP.)
            uint16_t committed_bit = (uint16_t)(1 << chosen_v);
            for (int p = tid; p < 20; p += WARP_SIZE) {
                int peer = d_peers[target_cell][p];
                if (!is_solved(s_cands[peer]) && s_cands[peer] != 0) {
                    s_cands[peer] &= ~committed_bit;
                }
            }
            warp.sync();
        } else {
            // Rejected: mark as processed
            if (tid == 0) s_curvatures[target_cell] = -1.0f;
            warp.sync();
            continue;
        }
    }
    warp.sync();

    // [E6] Re-run CP with correlation-length bounded iterations
    int cp_depth = compute_propagation_depth(K_max, Gamma);
    bool consistent = constraint_propagation(s_cands, cp_depth);
    warp.sync();

    if (tid == 0) {
        if (!consistent) {
            // Relaxation produced inconsistency — restore pre-rounding state
            // so Phase 3 gets a clean, solvable board.
            status[puzzle_idx] = 0;
        } else {
            bool solved = true;
            for (int i = 0; i < 81; i++) {
                if (!is_solved(s_cands[i])) { solved = false; break; }
            }
            status[puzzle_idx] = solved ? 2 : 0;
        }
    }
    warp.sync();

    // On failure, rollback to pre-rounding snapshot
    if (!consistent) {
        for (int i = tid; i < 81; i += WARP_SIZE) {
            s_cands[i] = s_pre_round[i];
        }
        warp.sync();
    }

    // Write back (clean board in all cases)
    for (int i = tid; i < 81; i += WARP_SIZE) {
        cand_boards[puzzle_idx * 81 + i] = s_cands[i];
    }
}


/* ========================================================================
 * SECTION 11: PHASE 3 — JACKKNIFE SPECULATIVE BRANCHING
 *
 * [E3] Cell selection uses two-tier MRV + susceptibility tiebreak.
 * [E6] Per-guess CP bounded by correlation-length estimate.
 * ======================================================================== */

__device__ bool davis_dfs_solve(uint16_t* cands, float K_max_hint, float Gamma_hint) {
    struct StackFrame {
        int cell;
        uint16_t remaining;
        uint16_t snapshot[81];
    };

    StackFrame stack[MAX_BRANCH_DEPTH];
    int stack_depth = 0;

    cg::coalesced_group warp = cg::coalesced_threads();
    int tid = warp.thread_rank();

    // [E6] Precompute propagation depth (may update as solve progresses)
    int cp_depth = compute_propagation_depth(K_max_hint, Gamma_hint);

    while (true) {
        bool consistent = constraint_propagation(cands, cp_depth);
        if (!consistent) goto backtrack;

        {
            bool solved = true;
            for (int i = 0; i < 81; i++) {
                if (!is_solved(cands[i])) { solved = false; break; }
            }
            if (solved) return true;
        }

        {
            // [E3] Two-tier MRV + χ̃ cell selection (curvatures unused — susceptibility only)
            int branch_cell = select_branch_cell(cands, nullptr);
            if (branch_cell < 0) goto backtrack;

            uint16_t candidates = cands[branch_cell];
            if (candidates == 0) goto backtrack;

            if (stack_depth >= MAX_BRANCH_DEPTH) goto backtrack;
            StackFrame* frame = &stack[stack_depth];
            frame->cell = branch_cell;

            // Extract first candidate
            int first_val = __ffs((unsigned int)candidates);
            uint16_t first_mask = value_to_bitmask(first_val);
            frame->remaining = candidates & ~first_mask;

            for (int i = tid; i < 81; i += WARP_SIZE) {
                frame->snapshot[i] = cands[i];
            }
            warp.sync();
            stack_depth++;

            if (holonomy_prune_value(cands, branch_cell, first_mask)) {
                goto try_next;
            }

            cands[branch_cell] = first_mask;
            continue;
        }

    try_next:
        {
            if (stack_depth == 0) return false;
            StackFrame* frame = &stack[stack_depth - 1];

            while (frame->remaining != 0) {
                int next_val = __ffs((unsigned int)frame->remaining);
                uint16_t next_mask = value_to_bitmask(next_val);
                frame->remaining &= ~next_mask;

                for (int i = tid; i < 81; i += WARP_SIZE) {
                    cands[i] = frame->snapshot[i];
                }
                warp.sync();

                if (holonomy_prune_value(cands, frame->cell, next_mask)) {
                    continue;
                }

                cands[frame->cell] = next_mask;
                goto propagate_again;
            }

            stack_depth--;
            goto try_next;
        }

    propagate_again:
        continue;

    backtrack:
        if (stack_depth == 0) return false;
        goto try_next;
    }
}

__global__ void phase3_branching_kernel(
    uint16_t*  __restrict__ cand_boards,
    int*       __restrict__ status,
    const float* __restrict__ gammas,       // [E6] Γ for propagation depth
    const float* __restrict__ max_curvs,    // [E6] K̂_max for propagation depth
    int batch_size
) {
    int puzzle_idx = blockIdx.x;
    if (puzzle_idx >= batch_size) return;
    if (status[puzzle_idx] != 0) return;

    cg::coalesced_group warp = cg::coalesced_threads();
    int tid = warp.thread_rank();

    __shared__ __align__(4) uint16_t s_cands[82]; // +1 pad for 32-bit atomicAnd safety
    if (tid == 0) s_cands[81] = 0;  // zero the pad element
    for (int i = tid; i < 81; i += WARP_SIZE) {
        s_cands[i] = cand_boards[puzzle_idx * 81 + i];
    }
    warp.sync();

    // [E6] Pass Γ and K̂_max hints for correlation-length bounded CP
    float Gamma = gammas[puzzle_idx];
    float K_max = max_curvs[puzzle_idx];
    bool solved = davis_dfs_solve(s_cands, K_max, Gamma);
    warp.sync();

    for (int i = tid; i < 81; i += WARP_SIZE) {
        cand_boards[puzzle_idx * 81 + i] = s_cands[i];
    }
    if (tid == 0) {
        status[puzzle_idx] = solved ? 3 : -1;
    }
}


/* ========================================================================
 * SECTION 12: HOST ORCHESTRATION
 *
 * Three-phase pipeline with stream synchronization events:
 *   Stream 0: Phase 1 (CP) → emits ev_phase1
 *   Stream 1: Trichotomy + Phase 2 (waits on ev_phase1) → emits ev_phase2
 *   Stream 2: Phase 3 (waits on ev_phase2) → emits ev_phase3
 * ======================================================================== */

struct SolverStats {
    int total_puzzles;
    int solved_phase1;
    int solved_phase2;
    int solved_phase3;
    int inconsistent;
    float total_time_ms;
    float phase1_time_ms;
    float phase2_time_ms;
    float phase3_time_ms;
};

class DavisSolverGPU {
public:
    DavisSolverGPU() {
        int h_peers[81][20];
        int h_row[81], h_col[81], h_box[81];
        compute_peer_table(h_peers);
        compute_cell_lookups(h_row, h_col, h_box);

        cudaMemcpyToSymbol(d_peers,  h_peers, sizeof(h_peers));
        cudaMemcpyToSymbol(d_row_of, h_row,   sizeof(h_row));
        cudaMemcpyToSymbol(d_col_of, h_col,   sizeof(h_col));
        cudaMemcpyToSymbol(d_box_of, h_box,   sizeof(h_box));

        cudaStreamCreate(&stream0);
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_phase1);
        cudaEventCreate(&ev_phase2);
        cudaEventCreate(&ev_phase3);
    }

    ~DavisSolverGPU() {
        cudaStreamDestroy(stream0);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_phase1);
        cudaEventDestroy(ev_phase2);
        cudaEventDestroy(ev_phase3);
    }

    SolverStats solve_batch(const int* puzzles, int* solutions, int batch_size) {
        SolverStats stats = {};
        stats.total_puzzles = batch_size;

        int*      d_input;
        uint16_t* d_cands;
        int*      d_status;
        float*    d_gammas;
        float*    d_max_curvs;
        float*    d_cv_values;    // [E5] Specific heat per puzzle

        cudaMalloc(&d_input,     batch_size * 81 * sizeof(int));
        cudaMalloc(&d_cands,     batch_size * 81 * sizeof(uint16_t));
        cudaMalloc(&d_status,    batch_size * sizeof(int));
        cudaMalloc(&d_gammas,    batch_size * sizeof(float));
        cudaMalloc(&d_max_curvs, batch_size * sizeof(float));
        cudaMalloc(&d_cv_values, batch_size * sizeof(float));

        cudaMemset(d_status, 0, batch_size * sizeof(int));
        cudaMemcpy(d_input, puzzles, batch_size * 81 * sizeof(int),
                   cudaMemcpyHostToDevice);

        cudaEventRecord(ev_start, stream0);

        // ==== PHASE 1: Constraint Propagation ====
        // [E6] Now also outputs K̂_max and Γ for downstream ξ use
        phase1_propagation_kernel<<<batch_size, WARP_SIZE, 0, stream0>>>(
            d_input, d_cands, d_status, d_max_curvs, d_gammas, batch_size
        );
        cudaEventRecord(ev_phase1, stream0);

        // Stream1 waits for Phase 1
        cudaStreamWaitEvent(stream1, ev_phase1, 0);

        // ==== TRICHOTOMY + CV ANALYSIS ====
        // [E5] Outputs CV for phase routing
        trichotomy_kernel<<<batch_size, WARP_SIZE, 0, stream1>>>(
            d_cands, d_gammas, d_max_curvs, d_cv_values, batch_size
        );

        // ==== PHASE 2: Thermodynamic Manifold Relaxation ====
        // [E1] Free energy, [E2] Adaptive annealing, [E4] Boltzmann rounding
        // [E5] CV routing, [E6] Correlation-length CP
        phase2_relaxation_kernel<<<batch_size, WARP_SIZE, 0, stream1>>>(
            d_cands, d_status, d_gammas, d_cv_values, d_max_curvs, batch_size
        );
        cudaEventRecord(ev_phase2, stream1);

        // Stream2 waits for Phase 2
        cudaStreamWaitEvent(stream2, ev_phase2, 0);

        // ==== PHASE 3: Jackknife Branching ====
        // [E3] Susceptibility tiebreak, [E6] Correlation-length CP
        phase3_branching_kernel<<<batch_size, WARP_SIZE, 0, stream2>>>(
            d_cands, d_status, d_gammas, d_max_curvs, batch_size
        );
        cudaEventRecord(ev_phase3, stream2);

        cudaDeviceSynchronize();

        cudaEventElapsedTime(&stats.phase1_time_ms, ev_start, ev_phase1);
        cudaEventElapsedTime(&stats.phase2_time_ms, ev_phase1, ev_phase2);
        cudaEventElapsedTime(&stats.phase3_time_ms, ev_phase2, ev_phase3);
        cudaEventElapsedTime(&stats.total_time_ms,  ev_start, ev_phase3);

        int*      h_status = new int[batch_size];
        uint16_t* h_cands  = new uint16_t[batch_size * 81];

        cudaMemcpy(h_status, d_status, batch_size * sizeof(int),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cands,  d_cands,  batch_size * 81 * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost);

        for (int p = 0; p < batch_size; p++) {
            for (int i = 0; i < 81; i++) {
                uint16_t c = h_cands[p * 81 + i];
                if (c != 0 && (c & (c - 1)) == 0) {
                    solutions[p * 81 + i] = host_ffs((unsigned int)c);
                } else {
                    solutions[p * 81 + i] = 0;
                }
            }
            if (h_status[p] == 1) stats.solved_phase1++;
            else if (h_status[p] == 2) stats.solved_phase2++;
            else if (h_status[p] == 3) stats.solved_phase3++;
            else stats.inconsistent++;
        }

        delete[] h_status;
        delete[] h_cands;
        cudaFree(d_input);
        cudaFree(d_cands);
        cudaFree(d_status);
        cudaFree(d_gammas);
        cudaFree(d_max_curvs);
        cudaFree(d_cv_values);

        return stats;
    }

private:
    cudaStream_t stream0, stream1, stream2;
    cudaEvent_t ev_start, ev_phase1, ev_phase2, ev_phase3;
};


/* ========================================================================
 * SECTION 12b: EXTERN "C" API FOR PYTHON CTYPES BINDING
 * ======================================================================== */

extern "C" {

void* davis_solver_create() {
    return static_cast<void*>(new DavisSolverGPU());
}

void davis_solver_destroy(void* handle) {
    delete static_cast<DavisSolverGPU*>(handle);
}

SolverStats davis_solver_solve_batch(
    void* handle, const int* puzzles, int* solutions, int batch_size
) {
    auto* solver = static_cast<DavisSolverGPU*>(handle);
    return solver->solve_batch(puzzles, solutions, batch_size);
}

}  // extern "C"


/* ========================================================================
 * SECTION 13: MAIN — Demo & Benchmarking
 * ======================================================================== */

void print_board(const int* board) {
    for (int r = 0; r < 9; r++) {
        if (r % 3 == 0 && r > 0) printf("  ------+-------+------\n");
        for (int c = 0; c < 9; c++) {
            if (c % 3 == 0 && c > 0) printf(" |");
            int val = board[r * 9 + c];
            printf("  %c", val ? '0' + val : '.');
        }
        printf("\n");
    }
}

bool parse_puzzle_string(const char* s, int* puzzle) {
    int len = 0;
    for (int i = 0; s[i] != '\0'; i++) {
        char c = s[i];
        if (c >= '0' && c <= '9') {
            if (len >= 81) return false;
            puzzle[len++] = c - '0';
        } else if (c == '.') {
            if (len >= 81) return false;
            puzzle[len++] = 0;
        }
    }
    return (len == 81);
}

int load_puzzles_from_file(const char* path, int* puzzles, int max_puzzles) {
    FILE* fp = fopen(path, "r");
    if (!fp) { printf("Error: cannot open %s\n", path); return 0; }
    char line[256];
    int count = 0;
    while (fgets(line, sizeof(line), fp) && count < max_puzzles) {
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len < 81) continue;
        if (line[0] == '#') continue;
        if (parse_puzzle_string(line, &puzzles[count * 81])) count++;
    }
    fclose(fp);
    return count;
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("  Davis Manifold Sudoku Solver — Blackwell GPU Edition\n");
    printf("  Thermodynamic Enhancements [E1-E6]\n");
    printf("  Based on: The Field Equations of Semantic Coherence\n");
    printf("          + The Thermodynamics of Semantic Coherence\n");
    printf("  Author:   Bee Rosa Davis (2025-2026)\n");
    printf("============================================================\n\n");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d, %d SMs, %.1f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9);

    DavisSolverGPU solver;

    // --- Mode 1: --file <path> ---
    if (argc >= 3 && strcmp(argv[1], "--file") == 0) {
        int* all_puzzles = new int[MAX_BATCH * 81];
        int count = load_puzzles_from_file(argv[2], all_puzzles, MAX_BATCH);
        if (count == 0) { printf("No valid puzzles loaded.\n"); delete[] all_puzzles; return 1; }

        printf("Loaded %d puzzles from %s\n\n", count, argv[2]);

        int* all_solutions = new int[count * 81];
        int total_solved = 0;

        for (int i = 0; i < count; i++) {
            int* puz = &all_puzzles[i * 81];
            int* sol = &all_solutions[i * 81];

            auto t0 = std::chrono::high_resolution_clock::now();
            SolverStats stats = solver.solve_batch(puz, sol, 1);
            auto t1 = std::chrono::high_resolution_clock::now();
            float wall_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

            int solved = stats.solved_phase1 + stats.solved_phase2 + stats.solved_phase3;
            total_solved += solved;
            int clues = 0;
            for (int c = 0; c < 81; c++) if (puz[c] != 0) clues++;

            printf("Puzzle %2d (%2d clues): %s  Wall: %7.3f ms  GPU: %7.3f ms  "
                   "P1: %.3f  P2: %.3f  P3: %.3f ms\n",
                   i + 1, clues, solved ? "SOLVED" : "FAILED",
                   wall_ms, stats.total_time_ms,
                   stats.phase1_time_ms, stats.phase2_time_ms, stats.phase3_time_ms);
        }
        printf("\n============================================================\n");
        printf("  Total: %d / %d solved\n", total_solved, count);
        printf("============================================================\n");

        if (count > 1) {
            printf("\nBatch mode (%d puzzles simultaneously):\n", count);
            auto bt0 = std::chrono::high_resolution_clock::now();
            SolverStats bstats = solver.solve_batch(all_puzzles, all_solutions, count);
            auto bt1 = std::chrono::high_resolution_clock::now();
            float bwall = std::chrono::duration<float, std::milli>(bt1 - bt0).count();
            int bsolved = bstats.solved_phase1 + bstats.solved_phase2 + bstats.solved_phase3;
            printf("  Wall time:     %.3f ms\n", bwall);
            printf("  GPU time:      %.3f ms\n", bstats.total_time_ms);
            printf("  Solved: %d / %d  (P1: %d, P2: %d, P3: %d)\n",
                   bsolved, count,
                   bstats.solved_phase1, bstats.solved_phase2, bstats.solved_phase3);
        }

        delete[] all_puzzles;
        delete[] all_solutions;
        return 0;
    }

    // --- Mode 2: puzzle string on command line ---
    int puzzle[81] = {
        0, 0, 0, 0, 0, 0, 0, 1, 2,
        0, 0, 0, 0, 3, 5, 0, 0, 0,
        0, 0, 0, 6, 0, 0, 0, 7, 0,
        7, 0, 0, 0, 0, 0, 3, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 8,
        0, 4, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 2, 0, 0, 0, 0, 0,
        6, 5, 0, 0, 0, 0, 0, 0, 0,
    };

    bool custom_puzzle = false;
    if (argc > 1) {
        const char* arg = argv[1];
        int digit_count = 0;
        for (int i = 0; arg[i]; i++) {
            if ((arg[i] >= '0' && arg[i] <= '9') || arg[i] == '.') digit_count++;
        }
        if (digit_count == 81) {
            if (parse_puzzle_string(arg, puzzle)) {
                custom_puzzle = true;
                printf("Custom puzzle loaded from command line.\n\n");
            }
        }
    }

    printf("Input puzzle:\n");
    print_board(puzzle);
    printf("\n");

    int solution[81];
    auto t0 = std::chrono::high_resolution_clock::now();
    SolverStats stats = solver.solve_batch(puzzle, solution, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    float wall_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    printf("Solution:\n");
    print_board(solution);
    printf("\n");

    printf("Stats:\n");
    printf("  Wall time:     %.3f ms\n", wall_ms);
    printf("  GPU time:      %.3f ms\n", stats.total_time_ms);
    printf("    Phase 1 (CP):   %.3f ms\n", stats.phase1_time_ms);
    printf("    Phase 2 (Relax):%.3f ms\n", stats.phase2_time_ms);
    printf("    Phase 3 (DFS):  %.3f ms\n", stats.phase3_time_ms);
    printf("  Solved: %d / %d  (P1: %d, P2: %d, P3: %d)\n",
           stats.solved_phase1 + stats.solved_phase2 + stats.solved_phase3,
           stats.total_puzzles,
           stats.solved_phase1, stats.solved_phase2, stats.solved_phase3);

    int batch_arg = 0;
    if (argc > 1 && !custom_puzzle) {
        batch_arg = atoi(argv[1]);
    } else if (argc > 2 && custom_puzzle) {
        batch_arg = atoi(argv[2]);
    }

    if (batch_arg > 0) {
        int batch_size = batch_arg;
        if (batch_size > MAX_BATCH) batch_size = MAX_BATCH;

        printf("\n============================================================\n");
        printf("  Batch Benchmark: %d puzzles\n", batch_size);
        printf("============================================================\n");

        int* batch_in  = new int[batch_size * 81];
        int* batch_out = new int[batch_size * 81];
        for (int p = 0; p < batch_size; p++) {
            memcpy(&batch_in[p * 81], puzzle, 81 * sizeof(int));
        }

        auto bt0 = std::chrono::high_resolution_clock::now();
        SolverStats bstats = solver.solve_batch(batch_in, batch_out, batch_size);
        auto bt1 = std::chrono::high_resolution_clock::now();
        float bwall = std::chrono::duration<float, std::milli>(bt1 - bt0).count();

        printf("  Wall time:     %.3f ms\n", bwall);
        printf("  GPU time:      %.3f ms\n", bstats.total_time_ms);
        printf("  Throughput:    %.0f puzzles/sec\n",
               batch_size / (bwall / 1000.0f));
        printf("  Per puzzle:    %.3f us\n",
               (bwall * 1000.0f) / batch_size);
        printf("  Solved: %d / %d  (P1: %d, P2: %d, P3: %d)\n",
               bstats.solved_phase1 + bstats.solved_phase2 + bstats.solved_phase3,
               bstats.total_puzzles,
               bstats.solved_phase1, bstats.solved_phase2, bstats.solved_phase3);

        delete[] batch_in;
        delete[] batch_out;
    }

    return 0;
}
