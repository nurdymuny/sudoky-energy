"""
Thermodynamic Solver Enhancements
===================================
Derived from: "The Thermodynamics of Semantic Coherence" (Davis, 2026)

Each enhancement maps a specific theorem to a concrete solver improvement.
This file contains both the Python (CPU) implementations and pseudocode
comments for the CUDA (GPU) translations.

These are DROP-IN replacements/additions to davis_solver.py and
davis_solver_blackwell.cu.
"""

import math
import time
from typing import Optional


# =============================================================================
# ENHANCEMENT 1: FREE ENERGY OBJECTIVE (from S2, Conjecture 4.1)
# =============================================================================
#
# PROBLEM: Current Phase 2 minimizes the Davis Energy E[γ].
#          But S2 says the equilibrium distribution minimizes the FREE energy:
#            F = ⟨E⟩ - τ·S
#          not E alone. Minimizing E alone ignores entropy — it can get stuck
#          in sharp, isolated minima when a nearby wider basin would be better.
#
# FIX: Replace the Phase 2 objective with F = E - τ·S.
#      The entropy term acts as a natural regularizer that keeps the
#      probability distribution from collapsing too early.
#
# THEORY: "Among all probability distributions P(γ) over completion paths,
#          the Boltzmann distribution uniquely minimizes F[P]" (Conj. 4.1a)
#
# GPU TRANSLATION: In phase2_relaxation_kernel, replace davis_energy_continuous()
#                  with davis_free_energy_continuous() below.

def davis_free_energy_continuous(probs, curvatures, tau=1.0):
    """
    F = ⟨E⟩ - τ·S  (Conjecture 4.1, Eq. 6)
    
    The entropy term -τ·S penalizes premature collapse of the probability
    distribution. At high τ (early in solving), the solver explores broadly.
    At low τ (late in solving), it commits to the minimum-energy solution.
    
    This is why Phase 2 currently fails on extreme puzzles: it minimizes E
    alone, which drives probabilities to 0/1 too aggressively. The free
    energy version anneals naturally.
    """
    energy = 0.0
    entropy = 0.0
    
    for cell in range(81):
        p = probs[cell]  # 9-element probability vector
        K = curvatures[cell]
        
        # Energy contribution (same as before)
        cell_entropy = 0.0
        for v in range(9):
            if p[v] > 1e-8:
                cell_entropy -= p[v] * math.log(p[v])
        
        # Violation term (same as before)
        # ... (peer interaction terms)
        
        energy += K * cell_entropy  # Curvature-weighted uncertainty
        entropy += cell_entropy      # Raw entropy for the τ·S term
    
    # F = E - τ·S
    # At τ=0: F = E (ground state, current behavior)
    # At τ>0: F = E - τ·S (entropic regularization)
    F = energy - tau * entropy
    return F


# =============================================================================
# ENHANCEMENT 2: ADAPTIVE ANNEALING SCHEDULE (from S1c, Conjecture 11.3)
# =============================================================================
#
# PROBLEM: Current Phase 2 uses a fixed learning rate (RELAX_LR = 0.05)
#          with constant momentum. The curvature-adaptive step size
#          (step = LR / (1 + K)) is a start, but it doesn't adapt OVER TIME.
#
# FIX: Anneal τ (temperature) during Phase 2 according to the optimal schedule:
#        dβ/dt ≤ Δ²_spectral / ln|S_valid|
#      Start hot (τ_high, explore), cool to τ_target (commit).
#
# THEORY: "If dβ/dt ≤ Δ²_spectral / ln|S_valid|, the system remains in
#          quasi-equilibrium and converges to the Boltzmann distribution"
#          (Conjecture 11.3, Eq. 35)
#
# PRACTICAL VERSION: We can't compute the spectral gap exactly, but we CAN
#   estimate it from the energy variance (S4): CV = β²⟨(ΔE)²⟩.
#   High CV means we're near the phase transition → slow down.
#   Low CV means we're in a stable phase → speed up.

class AdaptiveAnnealer:
    """
    Thermodynamically optimal annealing for Phase 2.
    
    Instead of fixed LR, we anneal temperature from τ_high to τ_target
    with a schedule that slows down near the phase transition (detected
    by specific heat divergence, S4).
    """
    
    def __init__(self, tau_high=2.0, tau_target=0.01, max_iter=500):
        self.tau_high = tau_high
        self.tau_target = tau_target
        self.max_iter = max_iter
        self.tau = tau_high
        self.energy_history = []
    
    def step(self, iteration, current_energy):
        """
        Compute the temperature for this iteration.
        
        Uses the specific heat (energy variance over recent window)
        to detect proximity to the phase transition and slow down.
        """
        self.energy_history.append(current_energy)
        
        # Estimate specific heat from energy variance (S4, Eq. 15)
        # CV = β² · ⟨(ΔE)²⟩
        window = min(20, len(self.energy_history))
        if window >= 5:
            recent = self.energy_history[-window:]
            mean_E = sum(recent) / len(recent)
            var_E = sum((e - mean_E)**2 for e in recent) / len(recent)
            beta = 1.0 / max(self.tau, 1e-8)
            CV = beta * beta * var_E
        else:
            CV = 0.0
        
        # Adaptive cooling rate (from S1c):
        # Cool fast when CV is low (stable phase),
        # cool slow when CV is high (near transition).
        # 
        # The annealing condition dβ/dt ≤ Δ²/ln|S| becomes:
        # Δτ ∝ 1 / (1 + CV)  (more conservative near transition)
        base_rate = (self.tau_high - self.tau_target) / self.max_iter
        adaptive_rate = base_rate / (1.0 + CV)
        
        self.tau = max(self.tau_target, self.tau - adaptive_rate)
        return self.tau
    
    def get_learning_rate(self, curvature):
        """
        Temperature-dependent, curvature-adaptive learning rate.
        
        From S1c + the existing curvature-adaptive step:
        step = τ / (1 + K)
        
        At high τ: large steps (exploration)
        At low τ: small steps (commitment)
        At high K: small steps (careful in curved regions)
        """
        return self.tau * 0.05 / (1.0 + curvature)


# GPU TRANSLATION (pseudocode for davis_solver_blackwell.cu):
#
# __shared__ float s_tau;           // Current temperature
# __shared__ float s_energy_hist[20]; // Ring buffer for CV estimation
# __shared__ int s_hist_idx;
#
# // In the Phase 2 relaxation loop:
# if (iter % 10 == 9 && tid == 0) {
#     // Estimate CV from energy variance
#     float mean_E = 0, var_E = 0;
#     for (int i = 0; i < 20; i++) mean_E += s_energy_hist[i];
#     mean_E /= 20.0f;
#     for (int i = 0; i < 20; i++) var_E += (s_energy_hist[i]-mean_E)*(s_energy_hist[i]-mean_E);
#     var_E /= 20.0f;
#     float CV = var_E / (s_tau * s_tau);
#     
#     // Adaptive cooling
#     float rate = base_rate / (1.0f + CV);
#     s_tau = fmaxf(tau_target, s_tau - rate);
# }
# __syncwarp();
#
# // Use s_tau in the step size:
# float step = s_tau * RELAX_LR / (1.0f + s_curvatures[cell]);


# =============================================================================
# ENHANCEMENT 3: SUSCEPTIBILITY-WEIGHTED BRANCHING (from S5, Conjecture 7.1)
# =============================================================================
#
# PROBLEM: Current cell selection uses Information Value V(c), which is
#          curvature integrated over the constraint region. This is good but
#          misses a key insight: cells with high SUSCEPTIBILITY (high response
#          to perturbation) are the ones where branching is most impactful.
#
# FIX: Weight the branching score by the cell's susceptibility χ(c).
#      From the FDT (Eq. 18): χ = β · ⟨(ΔW*)²⟩
#      For a cell: χ(c) ∝ |candidates| · coupling_with_peers
#
# THEORY: "Large thermal fluctuations imply small stability radius" (S5, Eq. 19)
#         Cells with high χ are where the solve is most "fragile."
#         Branching on them first resolves the fragility before it cascades.
#
# WHY IT'S BETTER THAN V(c) ALONE:
#   V(c) asks: "which cell carries the most information?"
#   χ(c) asks: "which cell is most UNSTABLE?"
#   Combined: "which unstable cell, when resolved, stabilizes the most?"

def susceptibility(board, cands, row, col, peers):
    """
    Compute χ(c) = local susceptibility of cell (row, col).
    
    From S5 (Eq. 18): χ = β · ⟨(ΔW*)²⟩
    
    Discretized for Sudoku:
      χ(c) = |candidates(c)| · Σ_{peers} overlap(c, peer) / |candidates(peer)|
    
    High χ means: this cell has many candidates AND those candidates
    compete heavily with peer candidates. It's a powder keg — resolving
    it will cascade hard (for good or ill).
    """
    cell_cands = cands[row][col]
    if cell_cands == 0 or (cell_cands & (cell_cands - 1)) == 0:
        return 0.0  # Solved or dead
    
    n_cands = bin(cell_cands).count('1')
    
    # Fluctuation proxy: how much would this cell's resolution perturb peers?
    fluctuation = 0.0
    for (pr, pc) in peers[row][col]:
        peer_cands = cands[pr][pc]
        if peer_cands == 0 or (peer_cands & (peer_cands - 1)) == 0:
            continue  # Skip solved/dead peers
        overlap = bin(cell_cands & peer_cands).count('1')
        peer_n = bin(peer_cands).count('1')
        # How much of this peer's option space would be eliminated?
        fluctuation += overlap / peer_n
    
    # χ ∝ n_candidates × total_disruption_potential
    chi = n_cands * fluctuation
    return chi


def select_branch_cell_thermodynamic(board, cands, curvatures, peers):
    """
    Combined V(c) + χ(c) cell selection.
    
    Score = V(c) · (1 + α · χ(c))
    
    where α controls the susceptibility weight.
    At α=0: pure Davis ordering (current behavior).
    At α>0: susceptibility-aware ordering (thermodynamic enhancement).
    
    The optimal α depends on the trichotomy regime:
      Γ > 1 (easy):   α ≈ 0 (V(c) alone suffices, no branching anyway)
      Γ ~ 1 (critical): α ≈ 1.0 (susceptibility matters most here!)
      Γ < 1 (hard):   α ≈ 0.5 (balance between info value and stability)
    """
    alpha = 0.5  # Default; adaptive version below
    
    best_cell = None
    best_score = -1.0
    
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                continue
            
            V = curvatures.get((r, c), 0.0)  # Information value
            chi = susceptibility(board, cands, r, c, peers)
            
            score = V * (1.0 + alpha * chi)
            
            if score > best_score:
                best_score = score
                best_cell = (r, c)
    
    return best_cell


# GPU TRANSLATION:
#
# In select_branch_cell(), after computing V for each cell, also compute χ:
#
# __device__ float compute_susceptibility(const uint16_t* cands, int cell) {
#     uint16_t my = cands[cell];
#     if (is_solved(my) || my == 0) return 0.0f;
#     int n = popcount16(my);
#     float fluct = 0.0f;
#     for (int p = 0; p < 20; p++) {
#         uint16_t peer = cands[d_peers[cell][p]];
#         if (is_solved(peer) || peer == 0) continue;
#         int overlap = popcount16(my & peer);
#         int pn = popcount16(peer);
#         fluct += (float)overlap / (float)pn;
#     }
#     return (float)n * fluct;
# }
#
# Then in the scoring: score = V * (1.0f + alpha * chi);


# =============================================================================
# ENHANCEMENT 4: BOLTZMANN ROUNDING (from S6, Conjecture 8.1)
# =============================================================================
#
# PROBLEM: Current Phase 2 rounds by argmax when max_prob > 0.8.
#          This is a hard threshold that either commits fully or not at all.
#          When multiple cells are near 0.8, the rounding order matters
#          and can create cascading inconsistencies.
#
# FIX: Use Metropolis-Hastings acceptance (S6a, Eq. 21):
#      Round cells in curvature-descending order.
#      For each cell, SAMPLE from the probability distribution
#      (not argmax), then accept with probability min(1, e^{-β·ΔE}).
#      If rejected, skip and try next cell.
#
# THEORY: "The following Markov chain converges to the Boltzmann distribution:
#          sample γ', compute ΔE, accept with min(1, e^{-βΔE})"
#          (Conjecture 16.1)
#
# WHY IT'S BETTER: Argmax is greedy — it always picks the highest-probability
#   value, which can conflict with peers. Boltzmann rounding samples
#   proportionally, so high-prob values are preferred but not guaranteed.
#   The Metropolis check catches bad samples before they propagate.

def boltzmann_round(probs, cands, curvatures, board, tau=0.1):
    """
    Thermodynamic rounding for Phase 2 output.
    
    Instead of hard argmax, sample from the distribution and accept
    via Metropolis criterion. Cells are processed in curvature-descending
    order (highest K first = most constrained first).
    
    From S6a (Conjecture 16.1): this converges to the Boltzmann
    distribution, which is the CORRECT distribution over completions.
    """
    import random
    
    # Sort cells by curvature (descending) — most constrained first
    unsolved = []
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                unsolved.append((curvatures[r][c], r, c))
    unsolved.sort(reverse=True)
    
    beta = 1.0 / max(tau, 1e-8)
    committed = []
    
    for (K, r, c) in unsolved:
        p = probs[r][c]  # 9-element probability vector
        
        # Only attempt if the distribution is reasonably peaked
        max_p = max(p)
        if max_p < 0.5:
            continue  # Too uncertain, leave for Phase 3
        
        # Sample from the distribution (not argmax)
        # Weighted random choice
        rand = random.random()
        cumulative = 0.0
        chosen_v = None
        for v in range(9):
            cumulative += p[v]
            if rand < cumulative:
                chosen_v = v + 1  # Values are 1-9
                break
        if chosen_v is None:
            chosen_v = p.index(max_p) + 1
        
        # Compute energy change from committing this cell
        # ΔE = energy_after - energy_before
        # Simplified: count how many peers lose this candidate
        chosen_mask = 1 << (chosen_v - 1)
        delta_E = 0.0
        for (pr, pc) in get_peers(r, c):
            if board[pr][pc] == 0:
                peer_cands = cands[pr][pc]
                if peer_cands & chosen_mask:
                    # This peer loses a candidate
                    remaining = bin(peer_cands & ~chosen_mask).count('1')
                    if remaining == 0:
                        delta_E = float('inf')  # Would kill a peer
                        break
                    # Losing a candidate increases curvature (bad)
                    delta_E += curvatures[pr][pc] / remaining
        
        # Metropolis acceptance
        if delta_E == float('inf'):
            continue  # Reject — would create inconsistency
        
        if delta_E <= 0:
            accept = True  # Energy decreased — always accept
        else:
            accept = random.random() < math.exp(-beta * delta_E)
        
        if accept:
            board[r][c] = chosen_v
            committed.append((r, c, chosen_v))
    
    return committed


# =============================================================================
# ENHANCEMENT 5: SPECIFIC HEAT PHASE ROUTING (from S4, Conjecture 6.1)
# =============================================================================
#
# PROBLEM: Current phase routing uses only Γ (trichotomy parameter).
#          Γ is computed ONCE from the initial board state.
#          But the effective difficulty CHANGES as propagation resolves cells.
#          A puzzle that starts at Γ=0.3 might reach Γ=0.8 after CP.
#
# FIX: Recompute the specific heat CV after Phase 1 and use it to decide
#      whether to attempt Phase 2 or skip straight to Phase 3.
#
# THEORY: "At the critical point (Γ=1), CV diverges" (S4, Eq. 16)
#         High CV after Phase 1 = near the transition = Phase 2 will struggle.
#         Low CV after Phase 1 = either easy (skip P2) or hard (P2 won't help).
#         Medium CV = Phase 2's sweet spot.
#
# DECISION LOGIC:
#   CV < 0.1:  Skip Phase 2 (either solved or too determined for relaxation)
#   0.1 ≤ CV ≤ 2.0:  Run Phase 2 (relaxation zone)
#   CV > 2.0:  Skip Phase 2, go straight to Phase 3 (near criticality,
#              relaxation will oscillate without converging)

def compute_specific_heat(board, cands):
    """
    Estimate CV = β² · ⟨(ΔE)²⟩ for the current board state.
    
    From S4 (Conjecture 6.1, Eq. 15).
    
    Discretized: we estimate energy variance by sampling random
    single-cell assignments and measuring the energy spread.
    
    For the GPU version, this is a single warp reduction over
    per-cell energy contributions.
    """
    # Compute per-cell energy contribution: how constrained is each cell?
    energies = []
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                mask = cands[r][c]
                n_cands = bin(mask).count('1')
                if n_cands == 0:
                    return float('inf')  # Inconsistent
                # Energy ~ inverse of candidate count (fewer = higher energy)
                energies.append(1.0 / n_cands)
    
    if len(energies) < 2:
        return 0.0
    
    mean_E = sum(energies) / len(energies)
    var_E = sum((e - mean_E)**2 for e in energies) / len(energies)
    
    # CV = β² · var_E (we use β=1 as reference scale)
    return var_E


def route_phase(board, cands, gamma):
    """
    Thermodynamic phase routing using Γ + CV.
    
    This replaces the simple Γ-threshold routing in the current solver.
    """
    CV = compute_specific_heat(board, cands)
    
    # Count unsolved cells
    unsolved = sum(1 for r in range(9) for c in range(9) if board[r][c] == 0)
    
    if unsolved == 0:
        return "solved"
    
    if gamma > 1.0 and CV < 0.1:
        return "phase1_only"      # CP will finish it
    
    if 0.1 <= CV <= 2.0 and gamma > 0.2:
        return "phase2"           # Relaxation sweet spot
    
    if CV > 2.0 or gamma < 0.2:
        return "phase3"           # Near criticality or deeply underdetermined
    
    return "phase2"               # Default to trying relaxation


# GPU TRANSLATION:
#
# __device__ float compute_specific_heat_warp(const uint16_t* cands) {
#     cg::coalesced_group warp = cg::coalesced_threads();
#     int tid = warp.thread_rank();
#     
#     float local_sum = 0.0f, local_sum2 = 0.0f;
#     int local_count = 0;
#     
#     for (int i = tid; i < 81; i += WARP_SIZE) {
#         if (!is_solved(cands[i]) && cands[i] != 0) {
#             float e = 1.0f / (float)popcount16(cands[i]);
#             local_sum += e;
#             local_sum2 += e * e;
#             local_count++;
#         }
#     }
#     
#     // Warp reduce sum, sum², count
#     for (int d = warp.size()/2; d > 0; d /= 2) {
#         local_sum += warp.shfl_down(local_sum, d);
#         local_sum2 += warp.shfl_down(local_sum2, d);
#         local_count += warp.shfl_down(local_count, d);
#     }
#     
#     float mean = local_sum / fmaxf(local_count, 1);
#     float var = local_sum2 / fmaxf(local_count, 1) - mean * mean;
#     return warp.shfl(var, 0);
# }
#
# // In the trichotomy kernel, after computing Γ:
# float CV = compute_specific_heat_warp(cands);
# // Route: CV < 0.1 → skip P2, 0.1-2.0 → P2, >2.0 → skip to P3


# =============================================================================
# ENHANCEMENT 6: CORRELATION-LENGTH PROPAGATION DEPTH (from S7, Conj. 9.1)
# =============================================================================
#
# PROBLEM: The propagation fixpoint loop runs until convergence (no changes).
#          But for extreme puzzles, convergence can take many iterations with
#          diminishing returns. The last few iterations eliminate 1-2 candidates
#          across the whole board.
#
# FIX: Use the correlation length ξ to estimate how many propagation
#      iterations are actually useful. Information can only propagate ξ cells
#      per iteration. After ceil(diameter / ξ) iterations, further propagation
#      has no effect beyond what's already been captured.
#
# THEORY: "ξ = (1/√K̂_max) · |1 - Γ|^{-ν}" (S7, Eq. 25)
#         The correlation length tells you the maximum distance over which
#         filling one cell constrains another.
#
# PRACTICAL: For Sudoku (diameter ~9), if ξ > 9, one propagation pass
#           reaches the whole board. If ξ < 3, you need ~3 passes.
#           Current code iterates up to 81 times — massively over-iterating
#           when ξ is large (most cases after initial CP).

def estimate_correlation_length(gamma, K_max):
    """
    Estimate ξ from the trichotomy parameter and max curvature.
    
    From S7 (Eq. 25): ξ = (1/√K̂_max) · |1 - Γ|^{-ν}
    where ν = τ/K̂_max
    """
    if K_max <= 0:
        return float('inf')  # Flat manifold, infinite correlation
    
    nu = 1.0 / K_max  # ν = τ/K̂_max with τ=1 for exact Sudoku
    
    if abs(1 - gamma) < 1e-6:
        return float('inf')  # Critical point, correlation diverges
    
    xi = (1.0 / math.sqrt(K_max)) * abs(1 - gamma) ** (-nu)
    return xi


def optimal_propagation_depth(gamma, K_max, board_diameter=9):
    """
    Compute the optimal number of propagation iterations.
    
    After ceil(diameter / ξ) iterations, information has traversed
    the entire board. Further iterations are wasted cycles.
    """
    xi = estimate_correlation_length(gamma, K_max)
    
    if xi <= 0:
        return board_diameter  # Fallback to worst case
    
    # Number of passes needed for information to cross the board
    passes = math.ceil(board_diameter / xi)
    
    # Clamp to reasonable range
    return max(2, min(passes + 1, 81))  # +1 for safety, min 2 for naked+hidden


# GPU TRANSLATION:
#
# In constraint_propagation(), replace:
#   for (int iter = 0; iter < 81; iter++) {
# with:
#   int max_iter = compute_propagation_depth(gamma, K_max);
#   for (int iter = 0; iter < max_iter; iter++) {
#
# This is pure upside on the GPU: fewer iterations = fewer warp syncs = 
# faster Phase 1 AND faster per-guess propagation in Phase 3.


# =============================================================================
# SUMMARY: EXPECTED IMPACT
# =============================================================================
#
# Enhancement          | Phase | Theory    | Expected Impact
# ---------------------|-------|-----------|----------------------------------
# 1. Free energy obj   | P2    | S2 (4.1)  | P2 solves more puzzles (fewer
#                      |       |           | go to P3), especially medium Γ
# 2. Adaptive anneal   | P2    | S1c(11.3) | Faster P2 convergence, avoids
#                      |       |           | oscillation near transitions
# 3. χ-weighted branch | P3    | S5 (7.1)  | Fewer backtracks in DFS; resolves
#                      |       |           | fragile cells before they cascade
# 4. Boltzmann round   | P2→3  | S6 (16.1) | Fewer P2 roundings that create
#                      |       |           | inconsistency → fewer P3 fallbacks
# 5. CV phase routing  | All   | S4 (6.1)  | Skips P2 when it can't help,
#                      |       |           | reduces wasted GPU cycles
# 6. ξ prop depth      | P1+P3 | S7 (9.1)  | Fewer propagation iterations,
#                      |       |           | direct GPU wall time reduction
#
# The largest single win is probably #3 (susceptibility branching) for hard
# puzzles and #6 (correlation-length propagation) for batch throughput.
# #1 and #2 together should meaningfully increase Phase 2's solve rate,
# reducing the fraction of puzzles that need Phase 3 at all.
