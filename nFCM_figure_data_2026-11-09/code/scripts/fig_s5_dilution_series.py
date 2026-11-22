"""Figure S5: scatter-triggered event rate of each serial-dilution series, with its fit.

Input (in the folder of this script):
  Figure_S5_A_acquisitions.csv  one row per acquisition (36 tubes and 3 buffers): sample, conc_nM (nominal), replicate
                                (dilution chain), acq_order, ss_events_per_s (VSSC-1-H >= 200, raw). The table is computed
                                from the FCS files by code/scripts/dilution_01_summarize.py.
Output (same folder):
  Figure_S5_A.svg               one panel per chain and the three fits together; points are not labeled
  Figure_S5_A.csv               plotted points and their residuals
  Figure_S5_A_fits.csv          fitted parameters of each chain
  Figure_S5_A_curves.csv        fitted curves as drawn

Fit of each chain: R = buffer + k C^alpha, least squares on ln R, buffer fixed at the mean of the three buffers; 95% CI of
alpha from the Jacobian. All 36 tubes are used, including the five above 7,000 events/s.

Needs numpy, pandas, scipy and matplotlib. Run:  python Figure_S5_A.py
(In the code folder this file is fig_s5_dilution_series.py; build_figure_package.py uses it and copies it to the panel.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats

STEM = 'Figure_S5_A'
INK, MUTED = '#222222', '#6b6b6b'
COLOR = {1: '#3354D1', 2: '#2E9E4F', 3: '#F28E2B'}
MARKER = {1: 'o', 2: 's', 3: '^'}
STYLE = {'font.family': 'Arial', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5, 'xtick.labelsize': 8,
         'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
         'svg.fonttype': 'none', 'svg.hashsalt': STEM, 'legend.frameon': False, 'lines.linewidth': 1.4}


def fit_chain(q, buffer):
    """Parameters and residuals ln(observed / fit) of one dilution chain."""
    c, y, n = q.conc_nM.values, np.log(q.ss_events_per_s.values), len(q)
    r = optimize.least_squares(lambda p: np.log(buffer + 10 ** p[0] * c ** p[1]) - y, [4, 0.5])
    cov = np.linalg.inv(r.jac.T @ r.jac) * np.sum(r.fun ** 2) / (n - 2)
    ci = stats.t.ppf(0.975, n - 2) * np.sqrt(cov[1, 1])
    return dict(log10k=r.x[0], alpha=r.x[1], alpha_ci95=ci, rss=float(np.sum(r.fun ** 2)), n_tubes=n), -r.fun


def analyze(acquisitions):
    """Buffer rate and the three tables (points, fits, curves)."""
    is_buffer = acquisitions.conc_nM == 0
    buffer = acquisitions[is_buffer].ss_events_per_s.mean()
    z = acquisitions[~is_buffer][['sample', 'replicate', 'conc_nM', 'acq_order', 'ss_events_per_s']]
    fits, points = [], []
    for rep in [1, 2, 3]:
        q = z[z.replicate == rep].sort_values('conc_nM')
        f, residual = fit_chain(q, buffer)
        fits.append(dict(replicate=rep, **f)); points.append(q.assign(residual_ln=residual))
    fits = pd.DataFrame(fits); x = np.logspace(-4.1, 0.65, 100)
    curves = pd.concat([pd.DataFrame(dict(replicate=f.replicate, conc_nM=x, fit_rate=buffer + 10 ** f.log10k * x ** f.alpha))
                        for f in fits.itertuples()])
    return buffer, {'tubes': pd.concat(points), 'fits': fits, 'curves': curves}


def draw(buffer, tables, svg):
    with plt.style.context('default', after_reset=True), plt.rc_context(STYLE):
        points, fits, curves = tables['tubes'], tables['fits'], tables['curves']
        fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.1), sharey=True)
        for ax, f in zip(axes[:3], fits.itertuples()):
            q, c = points[points.replicate == f.replicate], curves[curves.replicate == f.replicate]
            ax.axhline(buffer, color=MUTED, ls=':', lw=0.8)
            ax.text(1.1e-4, buffer * 0.72, 'buffer', fontsize=6.5, color=MUTED)
            ax.plot(c.conc_nM, c.fit_rate, color=COLOR[f.replicate], lw=1.3)
            ax.scatter(q.conc_nM, q.ss_events_per_s, s=26, marker=MARKER[f.replicate], color=COLOR[f.replicate],
                       edgecolor='white', lw=0.5, zorder=3)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(15, 2.5e4); ax.set_xlabel('nominal concentration (nM)')
            ax.set_title(f'Replicate {f.replicate}: slope {f.alpha:.2f} ± {f.alpha_ci95:.2f}\n'
                         f'(residual SS {f.rss:.2f}, {f.n_tubes} tubes)', loc='left', fontsize=8.5)
        axes[0].set_ylabel('scatter-triggered events/s')
        ax = axes[3]
        ax.axhline(buffer, color=MUTED, ls=':', lw=0.8)
        for f in fits.itertuples():
            q, c = points[points.replicate == f.replicate], curves[curves.replicate == f.replicate]
            ax.plot(c.conc_nM, c.fit_rate, color=COLOR[f.replicate], lw=1.3, label=f'replicate {f.replicate} ({f.alpha:.2f})')
            ax.scatter(q.conc_nM, q.ss_events_per_s, s=12, marker=MARKER[f.replicate], color=COLOR[f.replicate], alpha=0.5, lw=0)
        ax.set_xscale('log'); ax.set_xlabel('nominal concentration (nM)'); ax.legend(fontsize=6.5, loc='upper left')
        ax.set_title('All three fits', loc='left', fontsize=8.5)
        fig.tight_layout()
        fig.savefig(svg, bbox_inches='tight', metadata={'Date': None})     # no date, so a rerun gives an identical file
        plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    source = os.path.join(here, f'{STEM}_acquisitions.csv')
    if not os.path.isfile(source):
        raise SystemExit(f'{source} not found: run the copy in Figure_S5/Panel_A (Figure_S5_A.py).')
    buffer, tables = analyze(pd.read_csv(source))
    for name, df in tables.items():
        df.to_csv(os.path.join(here, STEM + ('' if name == 'tubes' else f'_{name}') + '.csv'), index=False)
    draw(buffer, tables, os.path.join(here, f'{STEM}.svg'))


if __name__ == '__main__':
    main()
