"""SI figure: acquisition order vs spacer length in the antibody session (04/18).

A  Observed/chance ratio (rho) of every antibody acquisition against clock time; lines join the tubes of one series
   (coating x replicate set), acquired in the order no biotin, 3, 6, 12 nt. Silica is shown faded (not analysed).
B  Antibody event rate (B525-H >= 250, buffer-subtracted) of the same acquisitions.
C  Decomposition of the per-antibody signal. Equations 2-4 give  excess = 100% x 2 tau rho0 x R_Ab x (F - 1),  so within a
   series F - 1 is proportional to excess / R_Ab. For each series, F - 1, the excess and 1 / R_Ab at each spacer length are
   divided by their 3-nt value (thin lines); thick lines are geometric means over series per coating.
D  rho of the no-biotin tubes against time in the session (drift between series), with a least-squares line.
Values are plotted as measured. PLL-PEG 12 nt, replicate C had no detectable origami and has no rho, F or excess.

Inputs: figures_svg/tccd_acquisition_order.csv, figures_svg/fig35_antibody_length_dependence_samples.csv
Out:    figures_svg/proposed_panels/figS_run_order_confound.svg and _acquisitions.csv, _decomposition.csv,
        _decomposition_means.csv, _statistics.csv
Run:    python fig39_run_order_confound.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import fig37_proposed_flow_panels as P
import make_manuscript_figures as M

CMAP = {'Uncoated': P.COL['NoCoating'], 'PLL-PEG': P.COL['PLL'], 'Silica': P.COL['Silica']}
CLBL = {'Uncoated': 'bare', 'PLL-PEG': 'PLL-PEG', 'Silica': 'silica (not analysed)'}
MK = {'N': 'o', '3': 's', '6': '^', '12': 'D'}
fmt = lambda p: 'p < 0.001' if p < 0.001 else (f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}')


def load():
    t = pd.read_csv(os.path.join(M.OUT, 'tccd_acquisition_order.csv'))
    t['start'] = pd.to_datetime(t.start); t0 = t.start.min()
    t['hours'] = (t.start - t0).dt.total_seconds() / 3600
    t['series'] = t.coating + ' ' + t.replicate
    t['extension'] = t.extension.astype(str)
    q = pd.read_csv(os.path.join(M.OUT, 'fig35_antibody_length_dependence_samples.csv'))[['sample', 'fold_change_vs_no_biotin',
                                                                                         'excess_pct_of_detected_origami']]
    return t.merge(q, on='sample', how='left'), t0


def decomposition(t):
    b = t[t.coating.isin(['Uncoated', 'PLL-PEG']) & t.valid & (t.extension != 'N')].copy()
    b['F_minus_1'] = b.fold_change_vs_no_biotin - 1
    b['excess'] = b.excess_pct_of_detected_origami
    b['inv_R_Ab'] = 1 / b.R_B
    b['identity_constant_2tau_rho0'] = b.excess / (100 * b.F_minus_1 * b.R_B)      # constant within a series
    rows = []
    for s, z in b.groupby('series'):
        z = z.set_index('extension')
        if '3' not in z.index:
            continue
        for e in z.index:
            rows.append(dict(series=s, coating=z.loc[e, 'coating'], replicate=z.loc[e, 'replicate'], spacer_nt=int(e),
                             minutes_since_series_start=z.loc[e, 'minutes_since_first_sample_of_set'],
                             F_minus_1=z.loc[e, 'F_minus_1'], excess_pct=z.loc[e, 'excess'], R_Ab=z.loc[e, 'R_B'],
                             F_minus_1_rel_3nt=z.loc[e, 'F_minus_1'] / z.loc['3', 'F_minus_1'],
                             excess_rel_3nt=z.loc[e, 'excess'] / z.loc['3', 'excess'],
                             inv_R_Ab_rel_3nt=z.loc[e, 'inv_R_Ab'] / z.loc['3', 'inv_R_Ab'],
                             identity_constant_2tau_rho0=z.loc[e, 'identity_constant_2tau_rho0']))
    D = pd.DataFrame(rows)
    gm = lambda v: float(np.exp(np.log(v).mean()))
    G = (D.groupby(['coating', 'spacer_nt'])[['F_minus_1_rel_3nt', 'excess_rel_3nt', 'inv_R_Ab_rel_3nt']].agg(gm)
         .join(D.groupby(['coating', 'spacer_nt']).size().rename('n_series')).reset_index())
    return b, D, G


def statistics(t, b, D):
    a = t[t.coating.isin(['Uncoated', 'PLL-PEG']) & t.valid].copy()
    out = []
    r = stats.pearsonr(b.extension.astype(int), b.minutes_since_first_sample_of_set)
    out.append(dict(test='spacer length vs minutes since series start (biotin tubes)', n=len(b), estimate=r[0], unit='Pearson r', p=r[1]))
    for y, lab in [('observed_over_chance', 'rho'), ('R_B', 'antibody event rate')]:
        bio = a[a.extension != 'N'].assign(ly=lambda z: np.log(z[y]))
        m = smf.ols('ly ~ minutes_since_first_sample_of_set + C(series)', data=bio).fit()
        k = 'minutes_since_first_sample_of_set'
        out.append(dict(test=f'{lab}, biotin tubes within a series: log y ~ minutes + series', n=int(m.nobs),
                        estimate=100 * (np.exp(10 * m.params[k]) - 1), unit='% per 10 min', p=m.pvalues[k]))
        n = a[a.extension == 'N']
        lr = stats.linregress(n.minutes_in_session, np.log(n[y]))
        out.append(dict(test=f'{lab}, no-biotin tubes across the session: log y ~ minutes in session', n=len(n),
                        estimate=100 * (np.exp(10 * lr.slope) - 1), unit='% per 10 min', p=lr.pvalue))
    w = D[D.spacer_nt == 12]
    for col, lab in [('F_minus_1_rel_3nt', 'F - 1'), ('excess_rel_3nt', 'excess'), ('inv_R_Ab_rel_3nt', '1 / antibody event rate')]:
        v = np.log(w[col])
        out.append(dict(test=f'{lab}, 12 nt relative to 3 nt (geometric mean over series)', n=len(v), estimate=float(np.exp(v.mean())),
                        unit='ratio', p=stats.ttest_1samp(v, 0).pvalue))
    out.append(dict(test='identity excess = 100 x k x R_Ab x (F - 1): largest within-series spread of k', n=len(b),
                    estimate=float(b.groupby('series').identity_constant_2tau_rho0.agg(lambda v: v.std() / v.mean()).max()),
                    unit='coefficient of variation', p=np.nan))
    return pd.DataFrame(out)


def main():
    t, t0 = load()
    b, D, G = decomposition(t)
    S = statistics(t, b, D)
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15], hspace=0.55, wspace=0.45)
    axA = fig.add_subplot(gs[0, :]); axB = fig.add_subplot(gs[1, :], sharex=axA)
    for ax, y, lab in [(axA, 'observed_over_chance', 'observed ÷ chance (ρ)'), (axB, 'R_B', 'antibody events per second')]:
        for s, z in t.groupby('series'):
            z = z.sort_values('order_in_set'); c = z.coating.iloc[0]; al = 0.4 if c == 'Silica' else 1
            zv = z[z.valid] if y == 'observed_over_chance' else z
            ax.plot(zv.hours, zv[y], '-', color=CMAP[c], lw=1, alpha=al)
            for e, zz in zv.groupby('extension'):
                ax.plot(zz.hours, zz[y], ls='', marker=MK[e], color=CMAP[c], mec='white', mew=0.5, ms=5, alpha=al)
        ax.set_ylabel(lab)
    axB.set_xlabel(f'hours since the first acquisition ({t0:%H:%M}, 04/18)')
    for ax, y in [(axA, 'rho'), (axB, 'antibody event rate')]:
        w = S[S.test.str.startswith(f'{y}, biotin')].iloc[0]; d0 = S[S.test.str.startswith(f'{y}, no-biotin')].iloc[0]
        ax.text(0.995 if y == 'rho' else 0.25, 0.03 if y == 'rho' else 0.98,
                f'biotin tubes within a series: {w.estimate:+.1f}% per 10 min ({fmt(w.p)})\n'
                f'no-biotin tubes across the session: {d0.estimate:+.1f}% per 10 min ({fmt(d0.p)})',
                transform=ax.transAxes, ha='right' if y == 'rho' else 'left', va='bottom' if y == 'rho' else 'top', fontsize=6.5)
    h = [plt.Line2D([], [], color=CMAP[c], lw=1.5, label=CLBL[c]) for c in CMAP]
    h += [plt.Line2D([], [], ls='', marker=MK[e], color=P.MUTED, label='no biotin' if e == 'N' else f'{e} nt') for e in MK]
    axA.legend(handles=h, ncol=7, loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=6.5, handlelength=1.5, columnspacing=0.9)
    axA.text(-0.07, 1.12, 'A', transform=axA.transAxes, fontsize=11, fontweight='bold')
    axB.text(-0.07, 1.05, 'B', transform=axB.transAxes, fontsize=11, fontweight='bold')
    # C: decomposition
    axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    for ax, col, lab in zip(axes, ['F_minus_1_rel_3nt', 'excess_rel_3nt', 'inv_R_Ab_rel_3nt'],
                            ['F − 1 (per antibody)', 'excess (per origami)', '1 / antibody event rate']):
        for s, z in D.groupby('series'):
            z = z.sort_values('spacer_nt')
            ax.plot(z.spacer_nt, z[col], '-', color=CMAP[z.coating.iloc[0]], lw=0.8, alpha=0.45)
        for c, g in G.groupby('coating'):
            g = g.sort_values('spacer_nt')
            ax.plot(g.spacer_nt, g[col], 'o-', color=CMAP[c], lw=2, ms=4, label=CLBL[c])
        ax.axhline(1, color=P.MUTED, ls='--', lw=0.8)
        ax.set_yscale('log'); ax.set_ylim(0.1, 40); ax.set_xticks([3, 6, 12]); ax.set_xlabel('spacer (nt)')
        ax.set_title(lab, fontsize=8)
        ax.set_yticks([0.1, 0.3, 1, 3, 10, 30]); ax.set_yticklabels(['0.1', '0.3', '1', '3', '10', '30']); ax.minorticks_off()
    axes[0].set_ylabel('relative to the 3-nt tube'); axes[0].legend(loc='upper left', fontsize=6.5)
    axes[0].text(-0.45, 1.1, 'C', transform=axes[0].transAxes, fontsize=11, fontweight='bold')
    # D: no-biotin baseline across the session
    ax = fig.add_subplot(gs[2, 3])
    n = t[(t.extension == 'N') & t.valid]
    for c, z in n.groupby('coating'):
        ax.plot(z.hours, z.observed_over_chance, 'o', color=CMAP[c], mec='white', ms=5, alpha=0.4 if c == 'Silica' else 1)
    nn = n[n.coating.isin(['Uncoated', 'PLL-PEG'])]
    lr = stats.linregress(nn.hours, nn.observed_over_chance); xs = np.linspace(nn.hours.min(), nn.hours.max(), 10)
    ax.plot(xs, lr.intercept + lr.slope * xs, color=P.INK, lw=1, ls='--')
    d0 = S[S.test.str.startswith('rho, no-biotin')].iloc[0]
    ax.set_title(f'no-biotin ρ across the session\n({fmt(d0.p)})', fontsize=8)
    ax.set_xlabel('hours'); ax.set_ylabel('ρ'); ax.set_ylim(9.5, 12.5)
    ax.text(-0.45, 1.1, 'D', transform=ax.transAxes, fontsize=11, fontweight='bold')
    cols = ['sample', 'coating', 'replicate', 'extension', 'series', 'start', 'hours', 'minutes_in_session', 'order_in_set',
            'minutes_since_first_sample_of_set', 'observed_over_chance', 'R_B', 'R_A', 'valid', 'fold_change_vs_no_biotin',
            'excess_pct_of_detected_origami']
    A = t[cols].rename(columns={'R_B': 'antibody_events_per_s', 'R_A': 'origami_events_per_s', 'hours': 'hours_since_first_acquisition'})
    P.save(fig, 'figS_run_order_confound', {'_acquisitions': A, '_decomposition': D, '_decomposition_means': G, '_statistics': S})
    pd.set_option('display.width', 220)
    print(S.round(4).to_string(index=False)); print(G.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
