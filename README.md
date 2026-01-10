# The ε-Shift Rule for Distributionally Robust Last-Mile Workforce Planning

## Overview

This repository contains the code and data for reproducing all numerical experiments in:

> **The ε-Shift Rule for Distributionally Robust Last-Mile Workforce Planning: Structural Property Identification and Operationalization**
> 
> Shigeharu Mizuno, Ritsumeikan Asia Pacific University

## Key Results

- **27.7 percentage point improvement** in oracle match rate in boundary cases (Scenario B)
- **DRO achieves 87.2% oracle match** vs SAA's 59.5% when distribution mean lies near decision threshold
- **1% Safe Zone Rule**: For ε ≥ 0.03 and moderate surge pricing, suboptimality remains below 1%

## Repository Structure

```
lastmile-dro/
├── epsilon_shift_main.tex          # Main paper source (165KB, 2,655 lines)
├── epsilon_shift_main.pdf          # Compiled PDF (72 pages)
├── numerical_values.tex            # Auto-generated LaTeX macros
├── references.bib                  # BibTeX references (34 entries)
├── epsilon_shift_calculator.xlsx   # Excel implementation tool
├── README.md                       # This file
│
├── src/                            # Python source code
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Parameters and experimental settings
│   ├── main.py                     # Main entry point (run simulations)
│   ├── simulation.py               # Monte Carlo simulation engine
│   ├── theory.py                   # Theoretical computations (thresholds)
│   ├── visualization.py            # Figure generation (Figures 1-10)
│   └── outputs.py                  # Output generation (tables, macros)
│
├── outputs/
│   ├── data/
│   │   └── results.json            # Complete simulation results (56KB)
│   ├── logs/
│   │   └── simulation.log          # Execution log
│   ├── figures/                    # Generated PDF figures (10 files)
│   │   ├── fig1_structural_necessity.pdf
│   │   ├── fig2_threshold_structure.pdf
│   │   ├── ...
│   │   └── fig10_calculator.pdf
│   └── tables/                     # Auto-generated LaTeX tables (13 files)
│       ├── numerical_values.tex
│       ├── table_main_result.tex
│       ├── table_scenario_b.tex
│       └── ...
│
└── figures/                        # Figures for paper compilation (10 files)
    ├── fig1_structural_necessity.pdf
    ├── fig2_threshold_structure.pdf
    ├── fig3_decision_regime.pdf
    ├── fig4_oos_performance.pdf
    ├── fig5_safe_zone.pdf
    ├── fig6_cvar_comparison.pdf
    ├── fig7_scenario_c.pdf
    ├── fig8_sensitivity.pdf
    ├── fig9_supply_uncertainty.pdf
    └── fig10_calculator.pdf
```

## Quick Start

### Requirements

```bash
pip install numpy scipy matplotlib pandas openpyxl
```

### Reproduce All Results

```bash
cd src
python main.py --seed 42 --replications 1000
```

This will:
1. Run all Monte Carlo simulations (1,000 replications per configuration)
2. Generate `outputs/data/results.json` with complete numerical results
3. Create `numerical_values.tex` with LaTeX macros for the paper
4. Generate all figures (fig1-fig10) in `outputs/figures/`
5. Generate all tables in `outputs/tables/`

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--seed` | 42 | Random seed for reproducibility |
| `--replications` | 1000 | Monte Carlo replications per scenario |
| `--eps` | 0.02 | Default robustness parameter |

## Module Descriptions

### `simulation.py` (Core Engine)

- `dro_decision()`: Implements Algorithm 1 (ε-shift rule)
- `saa_decision()`: Sample Average Approximation baseline
- `oracle_decision()`: Oracle with known true distribution
- `run_scenario()`: Execute full Monte Carlo experiment
- `compute_cvar_decision()`: CVaR-based conservative policy

### `theory.py` (Theoretical Computations)

- `compute_thresholds()`: Calculate p*_{m→m+1} from model parameters
- `compute_capacity_bounds()`: Calculate p̄_m values
- `verify_lipschitz()`: Validate uniform Lipschitz structure (L = Qc)

### `outputs.py` (Result Generation)

- `generate_latex_macros()`: Create numerical_values.tex
- `export_results_json()`: Export results.json (Single Source of Truth)
- `confidence_interval()`: Compute mean and 95% CI using t-distribution

### `visualization.py` (Figure Generation)

Generates all 10 figures for the paper:
- Figure 1: Structural necessity demonstration
- Figure 2: Threshold policy structure
- Figure 3: Decision regime visualization
- Figure 4: Out-of-sample performance comparison
- Figure 5: Safe zone validation
- Figure 6: CVaR comparison
- Figure 7: Scenario C analysis
- Figure 8: Sensitivity analysis
- Figure 9: Supply uncertainty extension
- Figure 10: Excel calculator screenshot

## Excel Calculator

`epsilon_shift_calculator.xlsx` provides a practical implementation tool:

1. **Input Sheet**: Enter your operational parameters (wage, GW cost, capacity, etc.)
2. **Calculation Sheet**: Automatically computes thresholds and DRO-shifted values
3. **Decision Sheet**: Provides staffing recommendation based on observed absence rate

No programming required—immediate deployment for logistics practitioners.

## Experimental Scenarios

| Scenario | Name | Distribution | Mean | Gap from p* |
|----------|------|--------------|------|-------------|
| A | Urban High-Density | Beta(10, 14) | 0.417 | +9.7% |
| B | Suburban Standard | Beta(10, 16) | 0.385 | +1.2% |
| C | Rural Depopulated | Beta(12, 28) | 0.300 | -21.1% |

## References

The paper cites 34 references (32 with DOI, 2 government URLs). All bibliographic entries are available in `references.bib` for BibTeX users.

## Reproducibility Checklist

- [x] Random seed fixed at 42
- [x] 1,000 Monte Carlo replications per configuration
- [x] 95% confidence intervals using t-distribution
- [x] All numerical values traced to results.json
- [x] LaTeX macros auto-generated from simulation outputs
- [x] All DOIs verified against publisher databases

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{mizuno2026epsilon,
  title={The $\varepsilon$-Shift Rule for Distributionally Robust Last-Mile Workforce Planning: Structural Property Identification and Operationalization},
  author={Mizuno, Shigeharu},
  school={College of International Management, Ritsumeikan Asia Pacific University},
  year={2026},
  address={Beppu, Japan},
  type={Bachelor's Thesis}
}
```

## Contact

- **Author**: Shigeharu Mizuno
- **Affiliation**: Ritsumeikan Asia Pacific University
- **Supervisor**: Associate Professor Hiroto Sato
