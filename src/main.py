"""
Main Entry Point
=================
Run complete simulation suite and generate all outputs.

Usage:
    cd project_root
    python -m src.main

Or:
    python src/main.py
"""

import sys
from pathlib import Path

# Add parent to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import save_results, setup_logging, get_results_path
from src.simulation import SimulationEngine
from src.visualization import FigureGenerator
from src.outputs import OutputGenerator


def main():
    """Run complete pipeline."""
    logger = setup_logging()
    
    logger.info("=" * 70)
    logger.info("ε-SHIFT RULE: SIMULATION AND OUTPUT GENERATION")
    logger.info("=" * 70)
    
    # Step 1: Run simulations
    logger.info("\n[STEP 1] Running simulations...")
    engine = SimulationEngine()
    results = engine.run_all_experiments()
    
    # Step 2: Save results (Single Source of Truth)
    logger.info("\n[STEP 2] Saving results to JSON...")
    path = save_results(results)
    logger.info(f"  ✓ Saved to {path}")
    
    # Step 3: Generate figures
    logger.info("\n[STEP 3] Generating figures...")
    fig_gen = FigureGenerator(results)
    fig_gen.generate_all_figures()
    
    # Step 4: Generate outputs
    logger.info("\n[STEP 4] Generating LaTeX outputs...")
    out_gen = OutputGenerator(results)
    out_gen.generate_all_outputs()
    
    # Step 5: Validation summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    p = results['parameters']
    logger.info(f"  p* = {p['p_star']:.3f}")
    logger.info(f"  p^DRO = {p['p_dro']:.3f}")
    logger.info(f"  L = Qc = {p['L']:.0f} JPY (uniform)")
    
    sc_b = results['scenarios']['B']['by_n']['20']
    logger.info(f"\n  Scenario B (N=20):")
    logger.info(f"    SAA oracle match: {sc_b['saa_match']*100:.1f}%")
    logger.info(f"    DRO oracle match: {sc_b['dro_match']*100:.1f}%")
    logger.info(f"    Advantage: +{sc_b['advantage_pp']:.1f} pp")
    
    # Verify Figure 1(c) fix
    panel_c = results['structural_necessity']['panel_c']
    logger.info(f"\n  Structural Necessity (Figure 1c):")
    logger.info(f"    α=0%: mismatch rate = {panel_c['mismatch_rate'][0]:.1f}% (should be 0.0%)")
    logger.info(f"    α=10%: mismatch rate = {panel_c['mismatch_rate'][2]:.1f}%")
    logger.info(f"    α=25%: mismatch rate = {panel_c['mismatch_rate'][5]:.1f}%")
    
    if panel_c['mismatch_rate'][0] == 0.0:
        logger.info("  ✓ Figure 1(c) bug FIXED: α=0% correctly shows 0% mismatch")
    else:
        logger.warning("  ⚠ Figure 1(c) bug NOT fixed!")
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nResults saved to: {get_results_path()}")
    
    return results


if __name__ == "__main__":
    main()
