"""
Configuration Module
====================
Central configuration for paths, parameters, and scenarios.

Design Principles:
- All paths use pathlib (OS-independent)
- No hardcoded absolute paths
- Automatic directory creation
- Single Source of Truth for all parameters
- Controlled numerical precision in JSON output
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
import json
import logging

# =============================================================================
# PATH MANAGEMENT
# =============================================================================

def get_project_root() -> Path:
    """Get project root (parent of src directory)."""
    return Path(__file__).parent.parent.resolve()


def get_output_dir() -> Path:
    """Get/create main output directory."""
    path = get_project_root() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_figure_dir() -> Path:
    """Get/create figure directory."""
    path = get_output_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_table_dir() -> Path:
    """Get/create table directory."""
    path = get_output_dir() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """Get/create data directory."""
    path = get_output_dir() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_log_dir() -> Path:
    """Get/create log directory."""
    path = get_output_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_path() -> Path:
    """Path to results.json (Single Source of Truth)."""
    return get_data_dir() / "results.json"


# Initialize on import
get_output_dir()


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(name: str = "ejor") -> logging.Logger:
    """Configure logging with file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    
    # File
    fh = logging.FileHandler(get_log_dir() / "simulation.log", mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    return logger


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

RANDOM_SEED = 42
N_REPLICATIONS = 1000
CONFIDENCE_LEVEL = 0.95
SAMPLE_SIZE_DEFAULT = 20


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

@dataclass
class ModelParameters:
    """
    Model parameters with economic interpretation.
    
    Key Property: Lipschitz constant L = Qc is UNIFORM across all m.
    This uniformity enables the ε-shift rule.
    
    Parameter sources (Table 6 in manuscript):
    - w: MHLW (2024) minimum wage statistics
    - T: Standard 8-hour shift
    - c: Yamato Transport interview (Mr. Yoshimoto, Oita branch, 2024)
    - rho: MLIT (2024) urban delivery density
    - t: Industry standard handling time
    """
    # Staff driver
    w: float = 1800.0    # Wage (JPY/hour)
    T: float = 8.0       # Shift (hours)
    n0: int = 1          # Existing SDs
    T0: float = 5.0      # Prior commitments (hours)
    
    # Gig worker
    c: float = 360.0     # Cost (JPY/pkg)
    
    # Demand
    A: float = 1.0       # Area (km²)
    rho: float = 200.0   # Density (pkg/km²)
    t: float = 5/60      # Handling time (hours/pkg) = 5 min
    
    # DRO
    epsilon: float = 0.02
    
    @property
    def Q(self) -> float:
        """Total packages."""
        return self.A * self.rho
    
    @property
    def S0(self) -> float:
        """Slack capacity (hours)."""
        return self.n0 * self.T - self.T0
    
    @property
    def L(self) -> float:
        """Lipschitz constant (UNIFORM across m)."""
        return self.Q * self.c
    
    @property
    def gamma(self) -> float:
        """Price ratio w/c."""
        return self.w / self.c
    
    def p_bar(self, m: int) -> float:
        """
        Capacity threshold p̄_m.
        
        Definition: Absence rate at which m SDs reach capacity.
        Formula: p̄_m = (S₀ + mT) / (Qt)
        """
        return (self.S0 + m * self.T) / (self.Q * self.t)
    
    def p_star(self, k: int = 0) -> float:
        """
        Switching threshold p*_{k→k+1}.
        
        Definition: Absence rate where J(k) = J(k+1).
        Formula: p* = p̄_k + wT/(Qc)
        """
        return self.p_bar(k) + (self.w * self.T) / (self.Q * self.c)
    
    def p_dro(self, k: int = 0) -> float:
        """
        DRO threshold p^DRO_{k→k+1}.
        
        THE ε-SHIFT RULE: p^DRO = p* - ε
        
        This holds because L = Qc is uniform, so robustness premium cancels.
        """
        return self.p_star(k) - self.epsilon
    
    def to_dict(self) -> Dict[str, Any]:
        """Export parameters with controlled precision."""
        return {
            'w': self.w,
            'T': self.T,
            'c': self.c,
            'Q': round(self.Q, 1),
            'S0': round(self.S0, 1),
            't': round(self.t, 6),
            'epsilon': self.epsilon,
            'gamma': round(self.gamma, 2),
            'L': round(self.L, 1),
            'p_bar_0': round(self.p_bar(0), 3),
            'p_bar_1': round(self.p_bar(1), 3),
            'p_star': round(self.p_star(0), 3),
            'p_dro': round(self.p_dro(0), 3),
        }


# =============================================================================
# SCENARIOS
# =============================================================================

@dataclass
class Scenario:
    """
    Test scenario with Beta distribution.
    
    Purpose:
    - A (μ > p*): DRO protects against underestimation
    - B (μ ≈ p*): Maximum DRO advantage (boundary)
    - C (μ < p*): DRO pays insurance premium
    """
    name: str
    description: str
    alpha: float  # Beta α
    beta: float   # Beta β
    
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def std(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b / ((a + b)**2 * (a + b + 1))) ** 0.5
    
    @property
    def variance(self) -> float:
        return self.std ** 2


SCENARIOS = {
    'A': Scenario('Urban', 'μ > p*: DRO protects', 10, 14),
    'B': Scenario('Suburban (Boundary)', 'μ ≈ p*: Max DRO value', 10, 16),
    'C': Scenario('Rural', 'μ < p*: Insurance cost', 12, 28),
}


# =============================================================================
# I/O WITH PRECISION CONTROL
# =============================================================================

def _round_floats(obj: Any, precision: int = 6) -> Any:
    """Recursively round floats to specified precision."""
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_floats(i, precision) for i in obj]
    return obj


def save_results(results: Dict[str, Any], precision: int = 6) -> Path:
    """
    Save results to JSON (Single Source of Truth).
    
    Applies rounding to avoid floating-point noise in output.
    """
    path = get_results_path()
    rounded = _round_floats(results, precision)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rounded, f, indent=2, ensure_ascii=False)
    return path


def load_results() -> Dict[str, Any]:
    """Load results from JSON."""
    path = get_results_path()
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


__all__ = [
    'get_project_root', 'get_output_dir', 'get_figure_dir',
    'get_table_dir', 'get_data_dir', 'get_log_dir', 'get_results_path',
    'setup_logging', 'RANDOM_SEED', 'N_REPLICATIONS', 'SAMPLE_SIZE_DEFAULT',
    'ModelParameters', 'Scenario', 'SCENARIOS', 'save_results', 'load_results',
]
