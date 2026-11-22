"""Serial dilution, step 14: each replicate series with its own fit.

Each replicate is treated as its own dilution chain (replicate 1: a1 ... l1; replicate 2: a2 ... l2; replicate 3:
a3 ... l3), with every tube as recorded. Fit per chain: R = buffer + k C^alpha (least squares on log rate; buffer = mean
of the three buffers, fixed); 95% CI of alpha from the Jacobian. Residuals ln(observed / fit) per tube.
Figure D18  one panel per chain (tubes labelled, fit, buffer) and the three fits together.
Run after dilution_01_summarize.py:  python dilution_14_replicate_series.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
import dilution_common as C


def fit(q, b):
    c = q.conc_nM.values; y = np.log(q.ss_events_per_s.values); n = len(y)
    r = optimize.least_squares(lambda p: np.log(b + 10 ** p[0] * c ** p[1]) - y, [4, 0.5])
    cov = np.linalg.inv(r.jac.T @ r.jac) * np.sum(r.fun ** 2) / (n - 2)
    ci = stats.t.ppf(0.975, n - 2) * np.sqrt(cov[1, 1])
    return dict(log10k=r.x[0], alpha=r.x[1], alpha_ci95=ci, rss=float(np.sum(r.fun ** 2)), n_tubes=n), -r.fun


def main():
    C.style()
    S = C.read('dilution_files.csv'); b = S[S.series == 'buffer'].ss_events_per_s.mean()
    z = S[S.series != 'buffer'][['sample', 'replicate', 'conc_nM', 'acq_order', 'ss_events_per_s']].copy()
    fits, res = [], []
    for rep in [1, 2, 3]:
        q = z[z.replicate == rep].sort_values('conc_nM')
        f, r = fit(q, b); fits.append(dict(replicate=rep, **f))
        res.append(q.assign(residual_ln=r))
    F = pd.DataFrame(fits); Z = pd.concat(res)
    xx = np.logspace(-4.1, 0.65, 100)
    curves = pd.concat([pd.DataFrame(dict(replicate=r.replicate, conc_nM=xx, fit_rate=b + 10 ** r.log10k * xx ** r.alpha)) for r in F.itertuples()])
    print(F.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.1), sharey=True)
    for ax, r in zip(axes[:3], F.itertuples()):
        q = Z[Z.replicate == r.replicate]; colr = C.REP_COL[r.replicate]
        ax.axhline(b, color=C.MUTED, ls=':', lw=0.8); ax.text(1.1e-4, b * 0.72, 'buffer', fontsize=6.5, color=C.MUTED)
        cv = curves[curves.replicate == r.replicate]
        ax.plot(cv.conc_nM, cv.fit_rate, color=colr, lw=1.3)
        ax.scatter(q.conc_nM, q.ss_events_per_s, s=26, marker=C.REP_MK[r.replicate], color=colr, edgecolor='white', lw=0.5, zorder=3)
        for t in q.itertuples():
            ax.annotate(t.sample, (t.conc_nM, t.ss_events_per_s), xytext=(3, 3 if t.residual_ln >= 0 else -9), textcoords='offset points',
                        fontsize=6.3, color=C.INK)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(15, 2.5e4); ax.set_xlabel('nominal concentration (nM)')
        ax.set_title(f'Replicate {r.replicate}: slope {r.alpha:.2f} ± {r.alpha_ci95:.2f}\n(residual SS {r.rss:.2f}, {r.n_tubes} tubes)',
                     loc='left', fontsize=8.5)
    axes[0].set_ylabel('scatter-triggered events/s')
    ax = axes[3]
    ax.axhline(b, color=C.MUTED, ls=':', lw=0.8)
    for r in F.itertuples():
        cv = curves[curves.replicate == r.replicate]; q = Z[Z.replicate == r.replicate]
        ax.plot(cv.conc_nM, cv.fit_rate, color=C.REP_COL[r.replicate], lw=1.3, label=f'replicate {r.replicate} ({r.alpha:.2f})')
        ax.scatter(q.conc_nM, q.ss_events_per_s, s=12, marker=C.REP_MK[r.replicate], color=C.REP_COL[r.replicate], alpha=0.5, lw=0)
    ax.set_xscale('log'); ax.set_xlabel('nominal concentration (nM)'); ax.legend(fontsize=6.5, loc='upper left')
    ax.set_title('All three fits', loc='left', fontsize=8.5)
    fig.tight_layout()
    C.save(fig, 'dilution_D18_replicate_series_fits', {'_tubes': Z, '_fits': F, '_curves': curves})


if __name__ == '__main__':
    main()
