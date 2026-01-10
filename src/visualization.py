"""
Figure Generator
================
Publication-quality figures from results.json.

VISUALIZATION ONLY - no computation.
Reads from Single Source of Truth (results.json).

FIX: Figure 7 now includes annotation about Y-axis range.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .config import get_figure_dir, load_results, ModelParameters


logger = logging.getLogger("ejor.visualization")


# Colorblind-friendly palette
COLORS = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00', 'purple': '#CC79A7', 'cyan': '#56B4E9',
    'gray': '#999999', 'black': '#000000',
}


def setup_style():
    """Configure matplotlib for publication."""
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'lines.linewidth': 1.8, 'lines.markersize': 6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': True, 'legend.framealpha': 0.9,
    })


setup_style()


class FigureGenerator:
    """Generate publication figures from results."""
    
    def __init__(self, results: Optional[Dict[str, Any]] = None):
        self.results = results if results else load_results()
        self.fig_dir = get_figure_dir()
        self.params = ModelParameters()
        self.p_star = self.params.p_star(0)
        self.p_dro = self.params.p_dro(0)
        logger.info(f"FigureGenerator ready: {self.fig_dir}")
    
    def _save(self, fig: plt.Figure, name: str):
        """Save figure in PDF and PNG."""
        for fmt in ['pdf', 'png']:
            fig.savefig(self.fig_dir / f"{name}.{fmt}", format=fmt,
                       dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"  ✓ {name}")
    
    # =========================================================================
    # FIGURE 1: STRUCTURAL NECESSITY
    # =========================================================================
    
    def figure1_structural_necessity(self):
        """Figure 1: Structural necessity of uniform Lipschitz."""
        data = self.results['structural_necessity']
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        plt.subplots_adjust(wspace=0.35)
        
        eps = np.array(data['eps_range'])
        
        # (a) Uniform L: exact match
        ax = axes[0]
        theory = data['panel_a']['theory']
        ax.plot(eps, theory, '-', color=COLORS['blue'], lw=2, label=r'Theory: $p^* - \varepsilon$')
        ax.scatter(eps[::5], [theory[i] for i in range(0, len(eps), 5)],
                  s=40, color=COLORS['orange'], zorder=5, label='Empirical DRO')
        ax.set_xlabel(r'Robustness $\varepsilon$')
        ax.set_ylabel(r'DRO threshold $p^{DRO}$')
        ax.set_title('(a) Uniform $L$: exact match')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 0.05)
        ax.set_ylim(0.32, 0.39)
        
        # (b) m-dependent L: divergence
        ax = axes[1]
        styles = [('-', r'Uniform ($\alpha$=0)'), ('--', r'$\alpha$=10%'), (':', r'$\alpha$=25%')]
        colors = [COLORS['blue'], COLORS['orange'], COLORS['red']]
        for (ls, label), color, key in zip(styles, colors,
                                           ['alpha_0.0', 'alpha_0.1', 'alpha_0.25']):
            if key in data['panel_b']:
                ax.plot(eps, data['panel_b'][key], ls, color=color, lw=1.8, label=label)
        ax.axhline(self.p_star, color=COLORS['gray'], ls='-.', lw=1, alpha=0.7, label=r'$p^*$')
        ax.set_xlabel(r'Robustness $\varepsilon$')
        ax.set_ylabel('Effective threshold')
        ax.set_title('(b) $m$-dependent $L$: divergence')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0, 0.05)
        ax.set_ylim(0.32, 0.42)
        
        # (c) Mismatch rate
        ax = axes[2]
        alphas = data['panel_c']['alpha_pct']
        rates = data['panel_c']['mismatch_rate']
        ax.bar(alphas, rates, width=4, color=COLORS['blue'], alpha=0.8, edgecolor='black', lw=0.5)
        ax.set_xlabel(r'Slope variation $\alpha$ (%)')
        ax.set_ylabel('Mismatch rate (%)')
        ax.set_title('(c) Decision quality impact')
        ax.set_xlim(-3, 33)
        
        self._save(fig, 'fig1_structural_necessity')
    
    # =========================================================================
    # FIGURE 2: THRESHOLD STRUCTURE
    # =========================================================================
    
    def figure2_threshold_structure(self):
        """Figure 2: Cost curves and optimal policy."""
        data = self.results['threshold_structure']
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.3)
        
        p_range = np.array(data['p_range'])
        th = data['thresholds']
        
        # (a) Cost curves
        ax = axes[0]
        colors = [COLORS['blue'], COLORS['green'], COLORS['orange']]
        for m, c in enumerate(colors):
            ax.plot(p_range, data['cost_curves'][f'm_{m}'], '-', color=c, lw=2, label=f'$m = {m}$')
        ax.axvline(th['p_star_01'], color=COLORS['red'], ls='--', lw=1.2, alpha=0.7)
        ax.axvline(th['p_star_12'], color=COLORS['red'], ls='--', lw=1.2, alpha=0.7)
        ax.annotate(r'$p^*_{0\to1}$', xy=(th['p_star_01'], 5), fontsize=9, ha='center', color=COLORS['red'])
        ax.annotate(r'$p^*_{1\to2}$', xy=(th['p_star_12'], 5), fontsize=9, ha='center', color=COLORS['red'])
        ax.set_xlabel('Absence rate $p$')
        ax.set_ylabel('Total cost (1000 JPY)')
        ax.set_title('(a) Cost curves by staffing level')
        ax.legend(loc='upper left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 75)
        
        # (b) Optimal policy
        ax = axes[1]
        p_steps = [0, th['p_star_01'], th['p_star_01'], th['p_star_12'], th['p_star_12'], 1.0]
        m_steps = [0, 0, 1, 1, 2, 2]
        ax.fill_between(p_steps, m_steps, step='post', alpha=0.2, color=COLORS['blue'])
        ax.step(p_steps, m_steps, where='post', lw=2.5, color=COLORS['blue'])
        ax.axvline(th['p_star_01'], color=COLORS['red'], ls='--', lw=1.2, alpha=0.7)
        ax.axvline(th['p_star_12'], color=COLORS['red'], ls='--', lw=1.2, alpha=0.7)
        ax.set_xlabel('Absence rate $p$')
        ax.set_ylabel('Optimal $m^*(p)$')
        ax.set_title('(b) Optimal staffing policy')
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 2.5)
        ax.set_yticks([0, 1, 2])
        
        self._save(fig, 'fig2_threshold_structure')
    
    # =========================================================================
    # FIGURE 3: DECISION REGIME
    # =========================================================================
    
    def figure3_decision_regime(self):
        """Figure 3: Decision regime map."""
        data = self.results['decision_regime']
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.35)
        
        gamma = np.array(data['gamma_range'])
        rho = np.array(data['rho_range'])
        
        for ax, regime, title in zip(axes,
                                     [np.array(data['normal_regime']), np.array(data['peak_regime'])],
                                     ['(a) Normal: $p = 0.20$', '(b) Peak: $p = 0.40$']):
            im = ax.imshow(regime, extent=[min(gamma), max(gamma), min(rho), max(rho)],
                          origin='lower', aspect='auto', cmap='Blues', vmin=0, vmax=2)
            ax.set_xlabel(r'Price ratio $\gamma = w/c$')
            ax.set_ylabel(r'Density $\rho$ (pkg/km²)')
            ax.set_title(title)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label(r'$m^*$')
        cbar.set_ticks([0, 1, 2])
        
        self._save(fig, 'fig3_decision_regime')
    
    # =========================================================================
    # FIGURE 4: OOS PERFORMANCE
    # =========================================================================
    
    def figure4_oos_performance(self):
        """Figure 4: Out-of-sample performance."""
        data = self.results['oos_performance']
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.3)
        
        sizes = data['sample_sizes']
        saa_cost = [data['by_n'][str(n)]['saa_cost_mean'] for n in sizes]
        dro_cost = [data['by_n'][str(n)]['dro_cost_mean'] for n in sizes]
        saa_match = [data['by_n'][str(n)]['saa_match'] * 100 for n in sizes]
        dro_match = [data['by_n'][str(n)]['dro_match'] * 100 for n in sizes]
        
        # (a) Cost
        ax = axes[0]
        ax.plot(sizes, saa_cost, 'o-', color=COLORS['red'], lw=2, label='SAA')
        ax.plot(sizes, dro_cost, 's-', color=COLORS['blue'], lw=2, label=r'DRO ($\varepsilon$=0.02)')
        ax.fill_between(sizes, dro_cost, saa_cost, alpha=0.2, color=COLORS['green'], label='DRO advantage')
        ax.set_xlabel('Sample size $N$')
        ax.set_ylabel('Mean OOS cost (1000 JPY)')
        ax.set_title('(a) Cost comparison')
        ax.legend(loc='upper right')
        
        # (b) Accuracy
        ax = axes[1]
        ax.plot(sizes, saa_match, 'o-', color=COLORS['red'], lw=2, label='SAA')
        ax.plot(sizes, dro_match, 's-', color=COLORS['blue'], lw=2, label=r'DRO ($\varepsilon$=0.02)')
        ax.set_xlabel('Sample size $N$')
        ax.set_ylabel('Oracle agreement (%)')
        ax.set_title('(b) Decision accuracy')
        ax.legend(loc='lower right')
        ax.set_ylim(50, 105)
        
        self._save(fig, 'fig4_oos_performance')
    
    # =========================================================================
    # FIGURE 5: SAFE ZONE
    # =========================================================================
    
    def figure5_safe_zone(self):
        """Figure 5: 1% Safe Zone."""
        data = self.results['safe_zone']
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.3)
        
        beta_range = data['beta_range']
        eps_range = data['eps_range']
        
        # (a) Gap vs β
        ax = axes[0]
        markers = ['v', 'o', 's', '^', 'D']
        for (key, marker) in zip(['eps_0.005', 'eps_0.01', 'eps_0.02', 'eps_0.03', 'eps_0.05'], markers):
            if key in data['gap_by_beta']:
                eps_val = float(key.split('_')[1])
                label = rf'$\varepsilon$={eps_val}'
                if eps_val in [0.03, 0.05]:  # Updated recommended range
                    label += r' $\star$'
                ax.plot(beta_range, data['gap_by_beta'][key], f'{marker}-', lw=1.5, label=label)
        ax.axhline(1.0, ls='--', color=COLORS['gray'], lw=1, label='1% threshold')
        ax.set_xlabel(r'Surge parameter $\beta_{surge}$')
        ax.set_ylabel('Suboptimality gap (%)')
        ax.set_title('(a) Gap vs. surge intensity')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_ylim(0, 4)
        
        # (b) Gap vs ε
        ax = axes[1]
        for key in ['beta_0.4', 'beta_0.6', 'beta_0.8', 'beta_1.0', 'beta_1.2']:
            if key in data['gap_by_eps']:
                beta_val = float(key.split('_')[1])
                ax.plot(eps_range, data['gap_by_eps'][key], '-', lw=1.5, label=rf'$\beta$={beta_val}')
        ax.axhline(1.0, ls='--', color=COLORS['gray'], lw=1)
        ax.axvspan(0.03, 0.05, alpha=0.2, color=COLORS['blue'])  # Updated recommended range
        ax.set_xlabel(r'Robustness $\varepsilon$')
        ax.set_ylabel('Suboptimality gap (%)')
        ax.set_title(r'(b) Gap vs. $\varepsilon$ (trade-off)')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 4)
        
        self._save(fig, 'fig5_safe_zone')
    
    # =========================================================================
    # FIGURE 6: CVaR COMPARISON
    # =========================================================================
    
    def figure6_cvar_comparison(self):
        """Figure 6: CVaR vs DRO."""
        data = self.results['cvar_comparison']['by_scenario']
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.3)
        
        scenarios = ['A', 'B', 'C']
        x = np.arange(len(scenarios))
        width = 0.2
        
        # (a) Oracle match
        ax = axes[0]
        matches = {'SAA': [], 'DRO': [], 'CVaR90': [], 'CVaR95': []}
        for s in scenarios:
            matches['SAA'].append(data[s]['saa']['match'] * 100)
            matches['DRO'].append(data[s]['dro']['match'] * 100)
            matches['CVaR90'].append(data[s]['cvar_90']['match'] * 100)
            matches['CVaR95'].append(data[s]['cvar_95']['match'] * 100)
        
        ax.bar(x - 1.5*width, matches['SAA'], width, label='SAA', color=COLORS['red'])
        ax.bar(x - 0.5*width, matches['DRO'], width, label='DRO', color=COLORS['blue'])
        ax.bar(x + 0.5*width, matches['CVaR90'], width, label='CVaR$_{0.9}$', color=COLORS['orange'])
        ax.bar(x + 1.5*width, matches['CVaR95'], width, label='CVaR$_{0.95}$', color=COLORS['purple'])
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Oracle agreement (%)')
        ax.set_title('(a) Decision accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend(loc='upper right', fontsize=8)
        
        # (b) Cost gap
        ax = axes[1]
        gaps = {'SAA': [], 'DRO': [], 'CVaR90': [], 'CVaR95': []}
        for s in scenarios:
            gaps['SAA'].append(data[s]['saa']['gap'])
            gaps['DRO'].append(data[s]['dro']['gap'])
            gaps['CVaR90'].append(data[s]['cvar_90']['gap'])
            gaps['CVaR95'].append(data[s]['cvar_95']['gap'])
        
        ax.bar(x - 1.5*width, gaps['SAA'], width, label='SAA', color=COLORS['red'])
        ax.bar(x - 0.5*width, gaps['DRO'], width, label='DRO', color=COLORS['blue'])
        ax.bar(x + 0.5*width, gaps['CVaR90'], width, label='CVaR$_{0.9}$', color=COLORS['orange'])
        ax.bar(x + 1.5*width, gaps['CVaR95'], width, label='CVaR$_{0.95}$', color=COLORS['purple'])
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Cost gap vs. oracle (%)')
        ax.set_title('(b) Cost penalty')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend(loc='upper right', fontsize=8)
        
        self._save(fig, 'fig6_cvar_comparison')
    
    # =========================================================================
    # FIGURE 7: SCENARIO C (WITH Y-AXIS ANNOTATION)
    # =========================================================================
    
    def figure7_scenario_c(self):
        """
        Figure 7: Scenario C trade-off.
        
        FIX: Added annotation explaining Y-axis starts at 85% to highlight small differences.
        """
        sc = self.results['scenarios']['C']
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        sizes = [10, 20, 30, 50, 100]
        saa = [sc['by_n'][str(n)]['saa_match'] * 100 for n in sizes]
        dro = [sc['by_n'][str(n)]['dro_match'] * 100 for n in sizes]
        
        ax.plot(sizes, saa, 'o-', color=COLORS['red'], lw=2, markersize=7, label='SAA')
        ax.plot(sizes, dro, 's-', color=COLORS['blue'], lw=2, markersize=7, label=r'DRO ($\varepsilon$=0.02)')
        ax.axhline(100, ls=':', color=COLORS['gray'], lw=1)
        ax.set_xlabel('Sample size $N$')
        ax.set_ylabel('Oracle agreement (%)')
        ax.set_title('Scenario C: DRO pays insurance premium')
        ax.legend(loc='lower right')
        ax.set_xlim(5, 105)
        ax.set_ylim(85, 102)
        
        # FIX: Add annotation about Y-axis range
        ax.text(0.98, 0.02, 'Note: Y-axis starts at 85%\nto highlight small differences',
                transform=ax.transAxes, fontsize=8, style='italic',
                ha='right', va='bottom', color=COLORS['gray'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['gray'], alpha=0.8))
        
        self._save(fig, 'fig7_scenario_c')
    
    # =========================================================================
    # FIGURE 8: SENSITIVITY
    # =========================================================================
    
    def figure8_sensitivity(self):
        """Figure 8: Parameter sensitivity."""
        data = self.results['sensitivity']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        variations = np.array(data['variations'])
        base = data['base_threshold']
        
        def pct(vals):
            return [(v - base) / base * 100 for v in vals]
        
        ax.plot(variations * 100 - 100, pct(data['wage']), 'o-', color=COLORS['blue'], lw=2, label='SD wage $w$')
        ax.plot(variations * 100 - 100, pct(data['gw_cost']), 's-', color=COLORS['orange'], lw=2, label='GW cost $c$')
        ax.plot(variations * 100 - 100, pct(data['density']), '^-', color=COLORS['green'], lw=2, label=r'Density $\rho$')
        ax.axhline(0, ls='--', color=COLORS['gray'], lw=1)
        ax.axvline(0, ls='--', color=COLORS['gray'], lw=1)
        ax.set_xlabel('Parameter variation (%)')
        ax.set_ylabel(r'Threshold change $\Delta p^*$ (%)')
        ax.set_title('Sensitivity of switching threshold')
        ax.legend(loc='best')
        ax.set_xlim(-35, 35)
        
        self._save(fig, 'fig8_sensitivity')
    
    # =========================================================================
    # FIGURE 9: SUPPLY UNCERTAINTY
    # =========================================================================
    
    def figure9_supply_uncertainty(self):
        """Figure 9: Supply uncertainty."""
        data = self.results['supply_uncertainty']
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.3)
        
        pi_vals = data['pi_values']
        
        # (a) Threshold vs π
        ax = axes[0]
        for key, color in zip(['delta_0.3', 'delta_0.5', 'delta_0.7', 'delta_1.0'],
                              [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red']]):
            delta = float(key.split('_')[1])
            ax.plot(pi_vals, data['thresholds'][key], '-', color=color, lw=2, label=rf'$\delta$={delta}')
        ax.axhline(self.p_star, ls='--', color=COLORS['gray'], lw=1, label=r'$p^*$ (no uncertainty)')
        ax.set_xlabel(r'GW availability $\pi$')
        ax.set_ylabel(r'Adjusted threshold $p^*(\pi)$')
        ax.set_title('(a) Threshold under supply uncertainty')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0.7, 1.0)
        
        # (b) Strategy comparison
        ax = axes[1]
        comp = data['comparison']
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, [comp['ignore']['mean'], comp['account']['mean']], width, label='Expected cost', color=COLORS['blue'])
        ax.bar(x + width/2, [comp['ignore']['p95'], comp['account']['p95']], width, label='95th percentile', color=COLORS['red'])
        ax.set_ylabel('Cost (1000 JPY)')
        ax.set_title(rf'(b) Strategy comparison ($\pi$={comp["pi"]}, $\delta$={comp["delta"]})')
        ax.set_xticks(x)
        ax.set_xticklabels(['Ignore\nuncertainty', 'Account for\nuncertainty'])
        ax.legend(loc='upper right')
        
        self._save(fig, 'fig9_supply_uncertainty')
    
    # =========================================================================
    # FIGURE 10: CALCULATOR
    # =========================================================================
    
    def figure10_calculator(self):
        """Figure 10: Calculator schema."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        boxes = [
            {'name': 'INPUT\nParameters', 'pos': (0.1, 0.7), 'color': COLORS['blue']},
            {'name': 'INPUT\nHistorical Data', 'pos': (0.1, 0.3), 'color': COLORS['blue']},
            {'name': 'COMPUTE\nThresholds', 'pos': (0.4, 0.5), 'color': COLORS['green']},
            {'name': 'DECISION\nm*_DRO', 'pos': (0.7, 0.5), 'color': COLORS['orange']},
        ]
        
        for box in boxes:
            rect = mpatches.FancyBboxPatch(box['pos'], 0.18, 0.15, boxstyle='round,pad=0.02',
                                           facecolor=box['color'], alpha=0.3,
                                           edgecolor=box['color'], linewidth=2)
            ax.add_patch(rect)
            ax.text(box['pos'][0] + 0.09, box['pos'][1] + 0.075, box['name'],
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        arrows = [((0.28, 0.775), (0.4, 0.575)), ((0.28, 0.375), (0.4, 0.525)), ((0.58, 0.55), (0.7, 0.55))]
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
        
        ax.text(0.5, 0.2, r'$m^*_{DRO} = \min\{k : \bar{p} \leq p^*_{k\to k+1} - \varepsilon\}$',
               ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['gray']))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Excel Calculator: ε-Shift Decision Tool', fontsize=14, fontweight='bold')
        
        self._save(fig, 'fig10_calculator')
    
    # =========================================================================
    # MAIN
    # =========================================================================
    
    def generate_all_figures(self):
        """Generate all publication figures."""
        logger.info("=" * 60)
        logger.info("Generating figures")
        logger.info("=" * 60)
        
        self.figure1_structural_necessity()
        self.figure2_threshold_structure()
        self.figure3_decision_regime()
        self.figure4_oos_performance()
        self.figure5_safe_zone()
        self.figure6_cvar_comparison()
        self.figure7_scenario_c()
        self.figure8_sensitivity()
        self.figure9_supply_uncertainty()
        self.figure10_calculator()
        
        logger.info("=" * 60)
        logger.info("All figures complete")
        logger.info("=" * 60)


__all__ = ['FigureGenerator', 'COLORS']
