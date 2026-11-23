"""Figure 29b: summary of the TCCD-style coincidence analysis (reads the outputs of fig29_tccd_coincidence.py).

Primary metric: observed / chance coincidences = N_coinc / E_chance, E_chance = R_A * R_B * (2 tau) * t, with
A = origami gate, B = B525-H >= trigger (250 antibody, 150 streptavidin), coincident = origami-gate events with
B525-H >= trigger, 2 tau = coincidence window measured from the inter-event-interval deficit (session median).
Specific comparison: fold change of observed/chance vs the N + probe sample of the SAME coating and replicate
(2 tau cancels). Coatings are never pooled.
Statistics per probe x coating x extension: geometric-mean fold change, 95% CI, one-sample t-test on log2 (n <= 3).
'All extensions' row: the three extensions are averaged within each replicate set first (they share one N), then
tested across sets. Flagged tubes (fig29 criteria, post hoc) and invalid samples are excluded.
Rows: anti-biotin IgG (04/18), streptavidin (05/25).
  A/D  observed / chance per sample, with the probe-only control (no origami) and 1 (pure chance)
  B/E  fold change vs own N
  C    antibody: change vs change in free-antibody brightness (robustness)
  F    streptavidin: change vs change in origami count (why it is not interpretable)

Run (after fig29_tccd_coincidence.py):  python fig29b_tccd_summary.py
Out:  figures_svg/fig29b_tccd_summary.svg, _summary.csv, _points.csv, _controls_and_window.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp, t as tdist
import make_manuscript_figures as M
from shared_utils import COATINGS, COAT_COLORS

IN = os.path.join(M.OUT, 'fig29_tccd_coincidence')
PROBES = ['Anti-biotin IgG', 'Streptavidin']
HUES = [M.COAT_LBL[c] for c in COATINGS]
PAL = {M.COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}
OFF = dict(zip(HUES, [-0.27, 0.0, 0.27]))


def fold_axis(ax):
    """Log2 fold-change axis labelled with plain fold values."""
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    lo, hi = ax.get_ylim()
    cand = [0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 2, 4, 8, 16, 32] if hi / lo < 2.5 else [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 4, 8, 16, 32]
    ticks = [t for t in cand if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))


def fold_stats(v):
    v = pd.Series(v).dropna(); n = len(v)
    out = dict(n_fold_change=n, mean_log2=v.mean() if n else np.nan, sem_log2=v.sem() if n >= 2 else np.nan)
    out['fold_change'] = 2 ** out['mean_log2'] if n else np.nan
    if n >= 2:
        h = tdist.ppf(0.975, n - 1) * out['sem_log2']
        out.update(fold_ci95_low=2 ** (out['mean_log2'] - h), fold_ci95_high=2 ** (out['mean_log2'] + h),
                   p_vs_1=ttest_1samp(v, 0).pvalue)
    else:
        out.update(fold_ci95_low=np.nan, fold_ci95_high=np.nan, p_vs_1=np.nan)
    return out


def main():
    D = pd.read_csv(IN + '_samples.csv', dtype={'extension': str, 'replicate': str})
    C = pd.read_csv(IN + '_controls.csv'); W = pd.read_csv(IN + '_coincidence_window.csv')
    use = D.valid & ~D.tube_flag
    P = D[use].copy()
    P['fold_change_vs_N'] = 2 ** P.log2_obs_exp_deficit_vs_N
    pts = P[['probe', 'coating', 'replicate', 'extension', 'sample', 'paired_N_sample', 'N_A', 'N_B', 'N_coinc',
             'E_chance_deficit', 'f_c', 'Q_A_deficit', 'obs_exp_deficit', 'log2_obs_exp_deficit_vs_N', 'fold_change_vs_N',
             'change_free_probe_median', 'log2_origami_rate_ratio']].rename(
        columns={'E_chance_deficit': 'E_chance', 'Q_A_deficit': 'Q_A', 'obs_exp_deficit': 'observed_over_chance',
                 'log2_obs_exp_deficit_vs_N': 'log2_fold_change_vs_N'})
    rows = []
    for probe in PROBES:
        for coat in HUES:
            z = pts[(pts.probe == probe) & (pts.coating == coat)]
            for e in ['N', '3', '6', '12']:
                q = z[z.extension == e]
                r = dict(probe=probe, coating=coat, extension=e, n_samples=len(q),
                         f_c_mean=q.f_c.mean(), Q_A_mean=q.Q_A.mean(),
                         observed_over_chance_mean=q.observed_over_chance.mean(),
                         observed_over_chance_sem=q.observed_over_chance.sem())
                if e != 'N':
                    r.update({k: v for k, v in fold_stats(q.log2_fold_change_vs_N).items()})
                    r['n_samples_above_N'] = int((q.log2_fold_change_vs_N > 0).sum())
                rows.append(r)
            sets = z[z.extension != 'N'].groupby('replicate').log2_fold_change_vs_N.mean().dropna()
            r = dict(probe=probe, coating=coat, extension='all extensions (per replicate set)', n_samples=len(sets))
            r.update(fold_stats(sets)); r['n_samples_above_N'] = int((sets > 0).sum())
            rows.append(r)
            for xcol in ['change_free_probe_median', 'log2_origami_rate_ratio']:
                zz = z[z.extension != 'N'].dropna(subset=['log2_fold_change_vs_N', xcol])
                if len(zz) >= 4:
                    pr = pearsonr(zz[xcol], zz.log2_fold_change_vs_N)
                    rows.append(dict(probe=probe, coating=coat, extension=f'correlation with {xcol}', n_samples=len(zz),
                                     pearson_r=pr.statistic, pearson_p=pr.pvalue))
    S = pd.DataFrame(rows)
    cw = [dict(item='coincidence window 2tau (us), interval deficit', session=s, median=z.two_tau_us.median(),
               iqr_low=z.two_tau_us.quantile(.25), iqr_high=z.two_tau_us.quantile(.75), n_files=len(z))
          for s, z in W.groupby('session', sort=False)]
    cw += [dict(item='median pulse width VSSC-1-W (scatter-triggered)', session=s,
                median=z.median_VSSC1_W_scatter_triggered.median(), n_files=len(z)) for s, z in W.groupby('session', sort=False)]
    pc = C[C.control.str.contains('only')]
    cw += [dict(item=f'probe-only control (no origami): {r.control}', session=r.probe, observed_over_chance=r.obs_exp_deficit,
                f_c=r.f_c, Q_A=r.Q_A_deficit) for r in pc.itertuples()]
    CW = pd.DataFrame(cw)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(27, 15))
    for pi, probe in enumerate(PROBES):
        Z = pts[pts.probe == probe]
        fl = D[(D.probe == probe) & D.valid & D.tube_flag]
        ctrl = pc[pc.probe == probe].iloc[0]
        # levels
        ax = axes[pi, 0]
        for coat in HUES:
            for i, e in enumerate(['N', '3', '6', '12']):
                v = Z[(Z.coating == coat) & (Z.extension == e)].observed_over_chance; x = i + OFF[coat]
                if len(v):
                    ax.bar(x, v.mean(), 0.25, color=PAL[coat], edgecolor='black', label=coat if i == 0 else None)
                    if len(v) >= 2:
                        ax.errorbar(x, v.mean(), v.sem(), color='black', capsize=3, lw=1.2)
                    ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=22, color='black', zorder=3)
                f = fl[(fl.coating == coat) & (fl.extension == e)].obs_exp_deficit
                if len(f):
                    ax.scatter(np.full(len(f), x), f, s=110, marker='X', color='white', edgecolor='black', zorder=4,
                               label='flagged tube (excluded)' if (coat, e) == ('Uncoated', '6') else None)
        ax.axhline(ctrl.obs_exp_deficit, color='#E6862E', ls=':', lw=2,
                   label=f'{ctrl.control.split(" (uncoated")[0]}, no origami: {ctrl.obs_exp_deficit:.0f}×')
        ax.axhline(1, color='red', ls='--', lw=1.2, label='pure chance (1×)')
        if probe == 'Streptavidin':
            ax.set_yscale('log')
        else:
            ax.set_ylim(0, 19)
        ax.set_xticks(range(4)); ax.set_xticklabels(['N', '3', '6', '12'])
        ax.set_xlabel('Sample (N = no biotin + probe)'); ax.set_ylabel('Observed / chance coincidences')
        ax.set_title(f"{'AD'[pi]}  {probe}: coincidences above chance", fontsize=14)
        ax.legend(fontsize=10, frameon=False, loc='upper left', ncol=2 if probe != 'Streptavidin' else 1)
        # fold change vs own N
        ax = axes[pi, 1]
        for coat in HUES:
            for i, e in enumerate(['3', '6', '12']):
                v = Z[(Z.coating == coat) & (Z.extension == e)].log2_fold_change_vs_N.dropna(); x = i + OFF[coat]
                if len(v):
                    st = fold_stats(v)
                    lo = 2 ** (st['mean_log2'] - (st['sem_log2'] if len(v) >= 2 else 0))
                    hi = 2 ** (st['mean_log2'] + (st['sem_log2'] if len(v) >= 2 else 0))
                    ax.bar(x, st['fold_change'] - 1, 0.25, bottom=1, color=PAL[coat], edgecolor='black',
                           label=coat if i == 0 else None)
                    if len(v) >= 2:
                        ax.errorbar(x, st['fold_change'], [[st['fold_change'] - lo], [hi - st['fold_change']]],
                                    color='black', capsize=3, lw=1.2)
                    ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), 2 ** v, s=22, color='black', zorder=3)
                    if len(v) >= 2:
                        ax.text(x, max(hi, (2 ** v).max()) + 0.01, f"{st['fold_change']:.2f}×\np={st['p_vs_1']:.2f}",
                                ha='center', va='bottom', fontsize=9)
                f = fl[(fl.coating == coat) & (fl.extension == e)].log2_obs_exp_deficit_vs_N.dropna()
                if len(f):
                    ax.scatter(np.full(len(f), x), 2 ** f, s=110, marker='X', color='white', edgecolor='black', zorder=4)
        ax.axhline(1, color='red', ls='--', lw=1.2)
        ax.set_yscale('log', base=2); fold_axis(ax)
        ax.set_xticks(range(3)); ax.set_xticklabels(['3', '6', '12'])
        ax.set_xlabel('Biotin extension (nt)'); ax.set_ylabel('Observed/chance, fold change vs own N (log scale)')
        ax.set_title(f"{'BE'[pi]}  {probe}: change vs no biotin (same coating and replicate)", fontsize=14)
        ax.legend(fontsize=10, frameon=False, loc='lower left')
        # diagnostics
        ax = axes[pi, 2]
        xcol, xl = (('change_free_probe_median', 'Change in free-probe brightness vs N\n(median B525-H, VSSC-1-H < 50)')
                    if probe != 'Streptavidin' else ('log2_origami_rate_ratio', 'log$_2$(origami rate / N)'))
        txt = []
        for coat in HUES:
            z = Z[(Z.coating == coat) & (Z.extension != 'N')].dropna(subset=[xcol, 'log2_fold_change_vs_N'])
            ax.scatter(z[xcol], 2 ** z.log2_fold_change_vs_N, s=80, color=PAL[coat], edgecolor='black', label=coat, zorder=3)
            cr = S[(S.probe == probe) & (S.coating == coat) & (S.extension == f'correlation with {xcol}')]
            if len(cr):
                txt.append(f'{coat}: r = {cr.iloc[0].pearson_r:+.2f} (p = {cr.iloc[0].pearson_p:.2f}, n = {int(cr.iloc[0].n_samples)})')
        ax.axhline(1, color='0.6', lw=1); ax.axvline(0, color='0.6', lw=1)
        ax.set_yscale('log', base=2); fold_axis(ax)
        ax.text(0.02, 0.98, '\n'.join(txt), transform=ax.transAxes, va='top', fontsize=11, zorder=5,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
        ax.set_xlabel(xl); ax.set_ylabel('Fold change vs own N')
        ax.set_title(f"{'CF'[pi]}  {probe}: " + ('not driven by brightness' if probe != 'Streptavidin'
                                                   else 'tracks origami count (aggregates dominate)'), fontsize=14)
        ax.legend(fontsize=10, frameon=False, loc='lower right')
    w = CW[CW.item.str.startswith('coincidence window')]
    fig.suptitle('Coincidence analysis (TCCD-style): observed / chance origami-probe coincidences; chance from the '
                 f'measured coincidence window (2τ = {w.iloc[0]["median"]:.1f} µs antibody session, '
                 f'{w.iloc[1]["median"]:.1f} µs streptavidin session)', y=1.02)
    M.save(fig, 'fig29b_tccd_summary', {'_summary': S, '_points': pts, '_controls_and_window': CW})
    pd.set_option('display.width', 220)
    show = S[S.extension.isin(['3', '6', '12', 'all extensions (per replicate set)'])]
    print(show[['probe', 'coating', 'extension', 'n_samples', 'n_fold_change', 'fold_change', 'fold_ci95_low', 'fold_ci95_high', 'p_vs_1',
                'n_samples_above_N']].round(3).to_string(index=False))
    print(S[S.extension.str.startswith('correlation')][['probe', 'coating', 'extension', 'n_samples', 'pearson_r',
                                                        'pearson_p']].round(2).to_string(index=False))
    print(CW.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
