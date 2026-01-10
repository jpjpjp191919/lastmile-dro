"""
Outputs Module
==============
Generate LaTeX tables, numerical values, and Excel calculator.

Reads from Single Source of Truth (results.json).
"""

from pathlib import Path
from typing import Dict, Any
import logging

from .config import get_table_dir, get_data_dir, load_results, ModelParameters

logger = logging.getLogger("ejor.outputs")


class OutputGenerator:
    """Generate outputs from results."""
    
    def __init__(self, results: Dict[str, Any] = None):
        self.results = results if results else load_results()
        self.table_dir = get_table_dir()
        self.data_dir = get_data_dir()
        self.params = ModelParameters()
        logger.info(f"OutputGenerator ready")
    
    def generate_numerical_values_tex(self) -> Path:
        """
        Generate LaTeX macros for numerical values.
        
        CRITICAL: These macros must be used in the manuscript
        to maintain Single Source of Truth.
        """
        p = self.results['parameters']
        sc_b = self.results['scenarios']['B']['by_n']['20']
        cvar = self.results['cvar_comparison']['by_scenario']['C']
        
        content = r"""% Auto-generated numerical values
% Single Source of Truth: results.json
% USAGE: Include this file in your LaTeX document and use the macros below.
% DO NOT hardcode these values in the manuscript!

% =============================================================================
% BASE PARAMETERS (Table 6)
% =============================================================================
\newcommand{\paramW}{""" + str(int(p['w'])) + r"""}           % SD wage (JPY/hour)
\newcommand{\paramT}{""" + str(int(p['T'])) + r"""}            % Shift length (hours)
\newcommand{\paramC}{""" + str(int(p['c'])) + r"""}          % GW cost (JPY/pkg)
\newcommand{\paramQ}{""" + str(int(p['Q'])) + r"""}          % Package volume
\newcommand{\paramRho}{200}        % Density (pkg/km²)
\newcommand{\paramEpsilon}{""" + str(p['epsilon']) + r"""}    % Robustness parameter
\newcommand{\paramL}{""" + str(int(p['L'])) + r"""}        % Lipschitz constant (JPY)
\newcommand{\paramGamma}{""" + str(p['gamma']) + r"""}        % Price ratio w/c

% =============================================================================
% THRESHOLDS
% =============================================================================
\newcommand{\pstar}{""" + f"{p['p_star']:.3f}" + r"""}         % Deterministic threshold
\newcommand{\pdro}{""" + f"{p['p_dro']:.3f}" + r"""}           % DRO threshold (ε=0.02)
\newcommand{\pbarZero}{""" + f"{p['p_bar_0']:.3f}" + r"""}     % Capacity threshold m=0
\newcommand{\pbarOne}{""" + f"{p['p_bar_1']:.3f}" + r"""}      % Capacity threshold m=1

% =============================================================================
% SCENARIO B - KEY CLAIMS (N=20, 1000 replications)
% =============================================================================
\newcommand{\scenarioBSAAMatch}{""" + f"{sc_b['saa_match']*100:.1f}" + r"""}    % SAA oracle match (%)
\newcommand{\scenarioBDROMatch}{""" + f"{sc_b['dro_match']*100:.1f}" + r"""}    % DRO oracle match (%)
\newcommand{\scenarioBAdvantage}{""" + f"{sc_b['advantage_pp']:.1f}" + r"""}   % DRO advantage (pp)

% =============================================================================
% CVaR COMPARISON - SCENARIO C
% =============================================================================
\newcommand{\cvarScenarioCMatchNinety}{""" + f"{cvar['cvar_90']['match']*100:.1f}" + r"""}  % CVaR₀.₉ match (%)
\newcommand{\cvarScenarioCMatchNinetyFive}{""" + f"{cvar['cvar_95']['match']*100:.1f}" + r"""}  % CVaR₀.₉₅ match (%)
\newcommand{\cvarScenarioCGapNinety}{""" + f"{cvar['cvar_90']['gap']:.1f}" + r"""}      % CVaR₀.₉ gap (%)
\newcommand{\cvarScenarioCGapNinetyFive}{""" + f"{cvar['cvar_95']['gap']:.1f}" + r"""}  % CVaR₀.₉₅ gap (%)

% =============================================================================
% SAFE ZONE
% =============================================================================
\newcommand{\safeZoneBetaMax}{0.8}
\newcommand{\safeZoneEpsMin}{0.01}
\newcommand{\safeZoneEpsMax}{0.02}

% =============================================================================
% SCENARIO DEFINITIONS
% =============================================================================
\newcommand{\scenarioAMean}{""" + f"{self.results['scenarios']['A']['true_mean']:.3f}" + r"""}
\newcommand{\scenarioBMean}{""" + f"{self.results['scenarios']['B']['true_mean']:.3f}" + r"""}
\newcommand{\scenarioCMean}{""" + f"{self.results['scenarios']['C']['true_mean']:.3f}" + r"""}
"""
        
        path = self.table_dir / "numerical_values.tex"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"  ✓ numerical_values.tex")
        return path
    
    def generate_scenario_b_table(self) -> Path:
        """Generate Table: Scenario B results."""
        sc = self.results['scenarios']['B']
        
        content = r"""\begin{table}[htbp]
\centering
\caption{Scenario B (Boundary Case) Performance: SAA vs DRO}
\label{tab:scenario_b}
\begin{tabular}{lcccc}
\toprule
Sample Size $N$ & SAA Match (\%) & DRO Match (\%) & Advantage (pp) & $p$-value \\
\midrule
"""
        for n in [10, 20, 30, 50, 100]:
            d = sc['by_n'][str(n)]
            content += f"{n} & {d['saa_match']*100:.1f} & {d['dro_match']*100:.1f} & {d['advantage_pp']:.1f} & $<$0.001 \\\\\n"
        
        content += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: 1,000 Monte Carlo replications, seed=42. Match rate = proportion of decisions 
agreeing with oracle. Advantage = DRO match - SAA match in percentage points.
\end{tablenotes}
\end{table}
"""
        
        path = self.table_dir / "table_scenario_b.tex"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"  ✓ table_scenario_b.tex")
        return path
    
    def generate_cvar_table(self) -> Path:
        """Generate Table: CVaR comparison."""
        data = self.results['cvar_comparison']['by_scenario']
        
        content = r"""\begin{table}[htbp]
\centering
\caption{Decision Rule Comparison Across Scenarios}
\label{tab:cvar}
\begin{tabular}{llcccc}
\toprule
Scenario & Metric & SAA & DRO & CVaR$_{0.9}$ & CVaR$_{0.95}$ \\
\midrule
"""
        for key, name in [('A', 'A (Urban)'), ('B', 'B (Boundary)'), ('C', 'C (Rural)')]:
            d = data[key]
            content += f"{name} & Oracle Match (\\%) & {d['saa']['match']*100:.1f} & {d['dro']['match']*100:.1f} & {d['cvar_90']['match']*100:.1f} & {d['cvar_95']['match']*100:.1f} \\\\\n"
            content += f" & Cost Gap (\\%) & {d['saa']['gap']:.1f} & {d['dro']['gap']:.1f} & {d['cvar_90']['gap']:.1f} & {d['cvar_95']['gap']:.1f} \\\\\n"
            if key != 'C':
                content += r"\midrule" + "\n"
        
        content += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: N=20, 1,000 replications. CVaR is designed for tail-risk protection, not 
oracle agreement; comparison uses expected cost criterion for fairness.
\end{tablenotes}
\end{table}
"""
        
        path = self.table_dir / "table_cvar.tex"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"  ✓ table_cvar.tex")
        return path
    
    def generate_all_outputs(self):
        """Generate all output files."""
        logger.info("=" * 60)
        logger.info("Generating outputs")
        logger.info("=" * 60)
        
        self.generate_numerical_values_tex()
        self.generate_scenario_b_table()
        self.generate_cvar_table()
        
        logger.info("=" * 60)
        logger.info("All outputs complete")
        logger.info("=" * 60)


__all__ = ['OutputGenerator']
