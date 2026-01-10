"""
EJOR ε-Shift Rule Package
========================
Simulation and analysis code for "The ε-Shift Rule: Simplifying Wasserstein DRO 
for Last-Mile Workforce Planning"

Modules:
- config: Path management, parameters, I/O
- theory: Pure mathematical functions
- simulation: Monte Carlo experiments
- visualization: Figure generation
- outputs: LaTeX tables and numerical values
"""

from .config import ModelParameters, SCENARIOS, load_results, save_results
from .simulation import SimulationEngine
from .visualization import FigureGenerator
from .outputs import OutputGenerator

__version__ = "2.0.0"
__author__ = "Shigeharu Mizuno"

__all__ = [
    'ModelParameters', 'SCENARIOS', 'load_results', 'save_results',
    'SimulationEngine', 'FigureGenerator', 'OutputGenerator',
]
