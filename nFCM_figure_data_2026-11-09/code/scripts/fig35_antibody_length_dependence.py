"""Figure 35: spacer-length dependence of antibody binding (antibody session 04/18, bare and PLL-PEG-coated origami).

A  Observed / chance coincidence ratio for no-biotin (N) and 3-, 6-, 12-nt spacers
B  Fold change of that ratio relative to the matched no-biotin sample (same coating and replicate) vs spacer length
C  Biotin-specific coincidence excess (% of detected origami; Equation 4, the quantity in Figure 5F) vs spacer length
D  The fold change of B against the time since the set's no-biotin acquisition: every set was acquired in the order
   N, 3, 6, 12 nt, so spacer length and run order cannot be separated in this dataset
Thin lines join the samples of one replicate set; thick lines and large markers are the mean over replicate sets.
One sample (PLL-PEG, 12 nt, replicate C) had no detectable origami and is absent.
Trend statistics (per coating and quantity): Spearman correlation with spacer length over samples, least-squares slope per
nucleotide with replicate set as a fixed effect, and the paired 12-nt minus 3-nt difference.

Inputs: fig29_tccd_coincidence_samples.csv, fig29d_tccd_ddG_samples.csv, tccd_acquisition_order.csv (figures_svg/)
Run:    python fig35_antibody_length_dependence.py
Out:    figures_svg/fig35_antibody_length_dependence.svg, _samples.csv, _means.csv, _trend_statistics.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import make_manuscript_figures as M
from shared_utils import COAT_COLORS

COATS = [('Uncoated', 'Bare origami', COAT_COLORS['NoCoating'], 'o'), ('PLL-PEG', 'PLL-PEG coated', COAT_COLORS['PLL'], 's')]
INK, MUTED = '#222222', '#6b6b6b'


def load():
    q = pd.read_csv(os.path.join(M.OUT, 'fig29_tccd_coincidence_samples.csv'))
    q = q[q.probe.str.contains('IgG') & q.coating.isin([c[0] for c in COATS]) & q.valid].copy()
    q['observed_over_chance'] = q.obs_exp_deficit
    d = pd.read_csv(os.path.join(M.OUT, 'fig29d_tccd_ddG_samples.csv'))[['sample', 'excess_simple', 'log2_fold_change_simple']]
    t = pd.read_csv(os.path.join(M.OUT, 'tccd_acquisition_order.csv'))[['sample', 'order_in_set', 'minutes_since_first_sample_of_set']]
    q = q.merge(d, on='sample', how='left').merge(t, on='sample', how='left')
    q['fold_change_vs_no_biotin'] = 2 ** q.log2_fold_change_simple
    q['excess_pct_of_detected_origami'] = q.excess_simple
    q['spacer'] = q.extension.astype(str)
    q['spacer_nt'] = pd.to_numeric(q.spacer, errors='coerce')
    return q[['sample', 'coating', 'replicate', 'spacer', 'spacer_nt', 'order_in_set', 'minutes_since_first_sample_of_set',
              'observed_over_chance', 'fold_change_vs_no_biotin', 'excess_pct_of_detected_origami']]


def trends(q):
    rows = []
    for coat, lab, *_ in COATS:
        z = q[(q.coating == coat) & q.spacer_nt.notna()]
        for col in ['observed_over_chance', 'fold_change_vs_no_biotin', 'excess_pct_of_detected_origami']:
            sp = stats.spearmanr(z.spacer_nt, z[col])
            X = pd.get_dummies(z.replicate, drop_first=True).astype(float); X['spacer_nt'] = z.spacer_nt.values; X.index = z.index
            m = sm.OLS(z[col], sm.add_constant(X)).fit()
            w = z.pivot(index='replicate', columns='spacer_nt', values=col)[[3.0, 12.0]].dropna()
            dif = w[12.0] - w[3.0]
            rows.append(dict(coating=lab, quantity=col, n_samples=len(z), spearman_r=sp[0], spearman_p=sp[1],
                             slope_per_nt=m.params['spacer_nt'], slope_se=m.bse['spacer_nt'], slope_p=m.pvalues['spacer_nt'],
                             mean_12nt_minus_3nt=dif.mean(), n_pairs=len(dif),
                             paired_p=stats.ttest_1samp(dif, 0).pvalue if len(dif) > 1 else np.nan))
    return pd.DataFrame(rows)


def series(ax, q, col, xcol, order=None):
    """Replicate lines (thin) and mean over replicate sets (thick) for each coating."""
    means = []
    for coat, lab, colr, mk in COATS:
        z = q[q.coating == coat]
        if order is not None:
            z = z.assign(x=z[xcol].map({k: i for i, k in enumerate(order)}))
        else:
            z = z.assign(x=z[xcol])
        z = z.dropna(subset=['x', col])
        for _, r in z.groupby('replicate'):
            r = r.sort_values('x')
            ax.plot(r.x, r[col], '-', color=colr, lw=1, alpha=0.45, zorder=2)
            ax.scatter(r.x, r[col], s=40, marker=mk, color=colr, alpha=0.55, edgecolor='white', linewidth=0.6, zorder=3)
        if xcol != 'minutes_since_first_sample_of_set':
            g = z.groupby('x')[col].agg(['mean', 'std', 'count']).reset_index()
            ax.plot(g.x, g['mean'], '-', color=colr, lw=2.5, zorder=4)
            ax.scatter(g.x, g['mean'], s=110, marker=mk, color=colr, edgecolor='white', linewidth=2, zorder=5, label=lab)
            means += [dict(coating=lab, quantity=col, x=xv, mean=mv, sd=sv, n=nv) for xv, mv, sv, nv in zip(g.x, g['mean'], g['std'], g['count'])]
        else:
            ax.scatter([], [], s=60, marker=mk, color=colr, label=lab)
    return means


def style(ax, title, xlabel, ylabel, loc='best'):
    ax.set_title(title, fontsize=14, color=INK, loc='left')
    ax.set_xlabel(xlabel, color=INK); ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis='y', color='#e6e6e6', lw=0.8); ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc=loc)


def main():
    q = load(); T = trends(q)
    b = q[q.spacer_nt.notna()]
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    means = []
    ax = axes[0, 0]
    means += series(ax, q, 'observed_over_chance', 'spacer', order=['N', '3', '6', '12'])
    ax.set_xticks(range(4)); ax.set_xticklabels(['No biotin', '3', '6', '12'])
    style(ax, 'A  Coincidence ratio by sample', 'Spacer length (nt)', 'Observed / chance coincidences')

    ax = axes[0, 1]
    means += series(ax, b, 'fold_change_vs_no_biotin', 'spacer_nt')
    ax.axhline(1, color=MUTED, ls='--', lw=1.2); ax.set_xticks([3, 6, 12])
    style(ax, 'B  Fold change relative to matched no-biotin sample', 'Spacer length (nt)', 'Fold change in coincidence ratio')

    ax = axes[1, 0]
    means += series(ax, b, 'excess_pct_of_detected_origami', 'spacer_nt')
    ax.axhline(0, color=MUTED, ls='--', lw=1.2); ax.set_xticks([3, 6, 12])
    style(ax, 'C  Biotin-specific excess', 'Spacer length (nt)', 'Excess (% of detected origami)')

    ax = axes[1, 1]
    series(ax, b, 'fold_change_vs_no_biotin', 'minutes_since_first_sample_of_set')
    ax.axhline(1, color=MUTED, ls='--', lw=1.2)
    r = stats.pearsonr(b.minutes_since_first_sample_of_set, np.log(b.fold_change_vs_no_biotin))
    sl = np.polyfit(b.minutes_since_first_sample_of_set, b.fold_change_vs_no_biotin, 1)
    xx = np.linspace(b.minutes_since_first_sample_of_set.min(), b.minutes_since_first_sample_of_set.max(), 50)
    ax.plot(xx, np.polyval(sl, xx), color=INK, lw=1.5, zorder=1)
    for nt, x0 in b.groupby('spacer_nt').minutes_since_first_sample_of_set.median().items():
        ax.text(x0, 0.90, f'{nt:g} nt', transform=ax.get_xaxis_transform(), ha='center', va='bottom', color=MUTED, fontsize=12)
    ax.text(0.98, 0.075, f'both coatings: r = {r[0]:.2f}, p = {r[1]:.3f}', transform=ax.transAxes, ha='right', color=INK, fontsize=12)
    style(ax, 'D  Same fold change against run time (order N, 3, 6, 12 in every set)',
          'Minutes since the no-biotin sample of the set', 'Fold change in coincidence ratio')
    ax.legend(frameon=False, fontsize=12, loc='lower right', bbox_to_anchor=(0.99, 0.13))

    fig.suptitle('Spacer-length dependence of anti-biotin IgG binding (thin lines: replicate sets; thick: mean)', y=0.995, color=INK)
    fig.tight_layout()
    M.save(fig, 'fig35_antibody_length_dependence', {'_samples': q, '_means': pd.DataFrame(means), '_trend_statistics': T})
    pd.set_option('display.width', 220)
    print(T.round(4).to_string(index=False))
    print(b.groupby(['coating', 'spacer_nt'])[['observed_over_chance', 'fold_change_vs_no_biotin', 'excess_pct_of_detected_origami']].mean().round(3))


if __name__ == '__main__':
    main()
