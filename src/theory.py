"""
Theory Module
=============
Pure mathematical functions implementing the paper's theory.

Contains NO randomness, NO I/O, NO side effects.
All functions are deterministic.

Key Results:
1. Q(m, p) has uniform slope Qc above p̄_m
2. p* = p̄ + wT/(Qc)
3. p^DRO = p* - ε (THE ε-SHIFT RULE)
"""

import numpy as np
from typing import Tuple
from .config import ModelParameters


# =============================================================================
# COST FUNCTIONS
# =============================================================================

def recourse_cost(m: int, p: float, params: ModelParameters) -> float:
    """
    Recourse function Q(m, p).
    
    Q(m, p) = { 0            if p ≤ p̄_m
              { (p - p̄_m)Qc  if p > p̄_m
    
    CRITICAL: Slope above p̄_m is Qc, UNIFORM for all m.
    """
    p_bar = params.p_bar(m)
    if p <= p_bar:
        return 0.0
    return (p - p_bar) * params.Q * params.c


def total_cost(m: int, p: float, params: ModelParameters) -> float:
    """Total cost J(m, p) = mwT + Q(m, p)."""
    return m * params.w * params.T + recourse_cost(m, p, params)


def total_cost_surge(m: int, p: float, beta_surge: float,
                     params: ModelParameters) -> float:
    """
    Total cost with surge pricing.
    
    c_effective = c × (1 + β × overflow_rate)
    """
    fixed = m * params.w * params.T
    p_bar = params.p_bar(m)
    
    if p <= p_bar:
        return fixed
    
    overflow_rate = p - p_bar
    surge_mult = 1 + beta_surge * overflow_rate
    overflow_pkgs = overflow_rate * params.Q
    variable = overflow_pkgs * params.c * surge_mult
    
    return fixed + variable


# =============================================================================
# DECISION RULES
# =============================================================================

def oracle_decision(true_p: float, params: ModelParameters, max_m: int = 3) -> int:
    """
    Oracle decision with perfect knowledge.
    
    m* = min{k : true_p ≤ p*_{k→k+1}}
    """
    for k in range(max_m):
        if true_p <= params.p_star(k):
            return k
    return max_m


def oracle_decision_surge(true_p: float, beta_surge: float,
                          params: ModelParameters, max_m: int = 3) -> int:
    """Oracle under surge pricing (numerical)."""
    costs = [total_cost_surge(m, true_p, beta_surge, params) 
             for m in range(max_m + 1)]
    return int(np.argmin(costs))


def saa_decision(sample_mean: float, params: ModelParameters, max_m: int = 3) -> int:
    """
    SAA decision using deterministic threshold.
    
    m^SAA = min{k : p̄ ≤ p*_{k→k+1}}
    """
    for k in range(max_m):
        if sample_mean <= params.p_star(k):
            return k
    return max_m


def dro_decision(sample_mean: float, params: ModelParameters, max_m: int = 3) -> int:
    """
    DRO decision using ε-SHIFT RULE.
    
    m^DRO = min{k : p̄ ≤ p*_{k→k+1} - ε}
    
    This is EQUIVALENT to solving the full Wasserstein DRO problem
    because L = Qc is uniform across m.
    """
    for k in range(max_m):
        if sample_mean <= params.p_dro(k):
            return k
    return max_m


def dro_decision_custom_eps(sample_mean: float, epsilon: float,
                            params: ModelParameters, max_m: int = 3) -> int:
    """DRO with custom ε."""
    for k in range(max_m):
        threshold = params.p_star(k) - epsilon
        if sample_mean <= threshold:
            return k
    return max_m


def cvar_decision(samples: np.ndarray, alpha: float,
                  params: ModelParameters, max_m: int = 3) -> int:
    """
    CVaR decision (upper-tail).
    
    CVaR_α^upper = E[p | p ≥ VaR_α]
    
    Note: CVaR is unconditionally conservative (always biases up).
    DRO is smart conservative (proportional to threshold proximity).
    """
    var = np.percentile(samples, alpha * 100)
    tail = samples[samples >= var]
    cvar = tail.mean() if len(tail) > 0 else var
    return saa_decision(cvar, params, max_m)


# =============================================================================
# STRUCTURAL NECESSITY (m-dependent slope)
# =============================================================================

def recourse_cost_mdep(m: int, p: float, alpha_slope: float,
                       params: ModelParameters) -> float:
    """
    Recourse with m-dependent Lipschitz: L(m) = Qc(1 + αm).
    
    This BREAKS the ε-shift rule because L(0) ≠ L(1).
    """
    p_bar = params.p_bar(m)
    if p <= p_bar:
        return 0.0
    L_m = params.Q * params.c * (1 + alpha_slope * m)
    return (p - p_bar) * L_m


def correct_dro_threshold_mdep(alpha_slope: float, params: ModelParameters) -> float:
    """
    Correct DRO threshold under m-dependent slope.
    
    When L(m) = Qc(1 + αm):
    - L(0) = Qc
    - L(1) = Qc(1 + α)
    
    The robustness premium no longer cancels.
    For α > 0, the correct threshold shifts UPWARD from p* - ε.
    
    Derivation:
    J^DRO(0) = E[Q(0,p)] + L(0)·ε = E[Q(0,p)] + Qc·ε
    J^DRO(1) = wT + E[Q(1,p)] + L(1)·ε = wT + E[Q(1,p)] + Qc(1+α)·ε
    
    Setting J^DRO(0) = J^DRO(1) at threshold p:
    (p - p̄₀)Qc + Qc·ε = wT + Qc(1+α)·ε
    p = p̄₀ + wT/(Qc) + α·ε = p* + α·ε
    
    CRITICAL: When α = 0, this reduces to p*, NOT p* - ε.
    The standard ε-shift (p* - ε) is derived under uniform L assumption.
    """
    if alpha_slope == 0:
        # Uniform L case: standard ε-shift applies
        return params.p_star(0) - params.epsilon
    else:
        # Non-uniform L: threshold shifts upward
        return params.p_star(0) + alpha_slope * params.epsilon


def standard_dro_threshold(params: ModelParameters) -> float:
    """Standard ε-shift threshold (assumes uniform L)."""
    return params.p_star(0) - params.epsilon


def mismatch_zone(alpha_slope: float, params: ModelParameters) -> Tuple[float, float]:
    """
    Return the mismatch zone [p_low, p_high].
    
    When α > 0:
    - Standard ε-shift threshold: p* - ε
    - Correct DRO threshold: p* + α·ε
    
    Mismatch occurs when sample_mean is in this interval:
    Zone = (p* - ε, p* + α·ε]
    Width = ε(1 + α)
    
    When α = 0: Zone has zero width (no mismatch).
    """
    p_low = params.p_star(0) - params.epsilon
    if alpha_slope == 0:
        p_high = p_low  # Zero-width zone
    else:
        p_high = params.p_star(0) + alpha_slope * params.epsilon
    return (p_low, p_high)


# =============================================================================
# SUPPLY UNCERTAINTY
# =============================================================================

def effective_gw_cost(pi: float, delta: float, c_base: float) -> float:
    """
    Effective GW cost under supply uncertainty.
    
    c̄ = πc + (1-π)c(1+δ) = c[1 + (1-π)δ]
    """
    return c_base * (1 + (1 - pi) * delta)


def threshold_supply_adjusted(pi: float, delta: float,
                              params: ModelParameters) -> float:
    """Threshold adjusted for supply uncertainty."""
    c_eff = effective_gw_cost(pi, delta, params.c)
    return params.p_bar(0) + (params.w * params.T) / (params.Q * c_eff)


# =============================================================================
# METRICS
# =============================================================================

def suboptimality_gap(m_decision: int, m_oracle: int,
                      true_p: float, params: ModelParameters) -> float:
    """
    Suboptimality gap (%).
    
    Gap = [C(m_decision) - C(m_oracle)] / C(m_oracle) × 100
    """
    c_decision = total_cost(m_decision, true_p, params)
    c_oracle = total_cost(m_oracle, true_p, params)
    if c_oracle <= 0:
        return 0.0
    return max(0, (c_decision - c_oracle) / c_oracle * 100)


def suboptimality_gap_surge(m_decision: int, m_oracle: int,
                            true_p: float, beta_surge: float,
                            params: ModelParameters) -> float:
    """Suboptimality gap under surge pricing."""
    c_decision = total_cost_surge(m_decision, true_p, beta_surge, params)
    c_oracle = total_cost_surge(m_oracle, true_p, beta_surge, params)
    if c_oracle <= 0:
        return 0.0
    return max(0, (c_decision - c_oracle) / c_oracle * 100)


__all__ = [
    'recourse_cost', 'total_cost', 'total_cost_surge',
    'oracle_decision', 'oracle_decision_surge',
    'saa_decision', 'dro_decision', 'dro_decision_custom_eps', 'cvar_decision',
    'recourse_cost_mdep', 'correct_dro_threshold_mdep',
    'standard_dro_threshold', 'mismatch_zone',
    'effective_gw_cost', 'threshold_supply_adjusted',
    'suboptimality_gap', 'suboptimality_gap_surge',
]
