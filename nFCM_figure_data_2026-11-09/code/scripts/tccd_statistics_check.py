"""Checks behind the TCCD / delta-U statements in the manuscript (antibody session 04/18, bare vs PLL-PEG).

Reads the outputs of fig29_tccd_coincidence.py, fig29b_tccd_summary.py and fig29d_tccd_ddG.py and reports:
  1  validity of the normalisation in Equation 4: in the no-biotin samples the coincident fraction f_c = R_c / R_O should be
     proportional to the antibody event rate R_Ab (log-log slope 1) and rho = observed / chance should be constant
  2  how much event rates differ between a biotin sample and its matched no-biotin sample
  3  biotin effect on rho (fold change vs no biotin) per coating
  4  antibody event rate by coating (the delta-U formula assumes equal antibody concentration)
  5  delta U = kT ln(excess_bare / excess_PLL) with small-sample statistics: ratio of set means, delta-method SE,
     percentile bootstrap (as plotted in Figure 5G), paired t-test on log set means (sets share an origami preparation),
     Welch tests, exact permutation; and the same after dividing each sample's excess by its antibody event rate
     (does not assume equal antibody concentration) or using the relative enrichment 1 - 1/F
Run:  python tccd_statistics_check.py
Out:  figures_svg/tccd_statistics_check.csv (one row per quantity), figures_svg/tccd_statistics_check_sets.csv
"""
import os, itertools
import numpy as np, pandas as pd
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures_svg')
rows = []
add = lambda group, quantity, value, note='': rows.append(dict(group=group, quantity=quantity, value=value, note=note))


def main():
    q = pd.read_csv(os.path.join(OUT, 'fig29_tccd_coincidence_samples.csv'))
    q = q[q.probe.str.contains('IgG') & (q.coating != 'Silica')].copy()
    add('exclusions', 'samples without detectable origami after buffer subtraction', ', '.join(q[~q.valid]['sample']))
    q = q[q.valid].copy()
    q['fc'] = 100 * q.N_coinc / q.N_A; q['rho'] = q.obs_exp_deficit
    n = q[q.extension.astype(str) == 'N']

    # 1 normalisation in the controls
    sl = stats.linregress(np.log(n.R_B), np.log(n.fc))
    add('normalisation', 'no-biotin: log f_c vs log R_Ab slope', sl.slope, f'SE {sl.stderr:.3f}, r {sl.rvalue:.3f}, n {len(n)}')
    add('normalisation', 'no-biotin: rho mean', n.rho.mean()); add('normalisation', 'no-biotin: rho CV %', 100 * n.rho.std() / n.rho.mean())
    add('normalisation', 'no-biotin: R_Ab range (events/s)', f'{n.R_B.min():.0f}-{n.R_B.max():.0f}', f'{n.R_B.max() / n.R_B.min():.1f}-fold')
    add('normalisation', 'no-biotin: f_c range (%)', f'{n.fc.min():.1f}-{n.fc.max():.1f}')

    # 2 rate differences within matched pairs
    N = n.set_index(['coating', 'replicate'])
    b = q[q.extension.astype(str) != 'N'].copy()
    b['RAb_ratio'] = [r.R_B / N.loc[(r.coating, r.replicate), 'R_B'] for r in b.itertuples()]
    b['RO_ratio'] = [r.R_A / N.loc[(r.coating, r.replicate), 'R_A'] for r in b.itertuples()]
    add('rates', 'biotin / matched no-biotin antibody event rate, range', f'{b.RAb_ratio.min():.2f}-{b.RAb_ratio.max():.2f}')
    add('rates', 'biotin / matched no-biotin origami event rate, range', f'{b.RO_ratio.min():.2f}-{b.RO_ratio.max():.2f}')

    # 3 fold change of rho vs no biotin
    s = pd.read_csv(os.path.join(OUT, 'fig29b_tccd_summary_summary.csv'))
    for r in s[s.probe.str.contains('IgG') & s.extension.str.contains('all')].itertuples():
        add('biotin effect', f'{r.coating}: fold change of rho vs no biotin', r.fold_change,
            f'95% CI {r.fold_ci95_low:.3f}-{r.fold_ci95_high:.3f}, p {r.p_vs_1:.4f}, n 3 sets')

    # 4 antibody event rate by coating
    per = q.groupby(['coating', 'replicate']).R_B.mean().unstack(0)
    add('antibody rate', 'bare / PLL-PEG antibody event rate (geometric, per sample)',
        np.exp(np.log(q[q.coating == 'Uncoated'].R_B).mean() - np.log(q[q.coating == 'PLL-PEG'].R_B).mean()),
        f"Welch p {stats.ttest_ind(np.log(q[q.coating == 'Uncoated'].R_B), np.log(q[q.coating == 'PLL-PEG'].R_B), equal_var=False).pvalue:.4f}")
    add('antibody rate', 'bare / PLL-PEG antibody event rate (set means)', per['Uncoated'].mean() / per['PLL-PEG'].mean(),
        f"paired t on log, p {stats.ttest_rel(np.log(per['Uncoated']), np.log(per['PLL-PEG'])).pvalue:.3f}")
    add('antibody rate', 'low-scatter antibody rate bare / PLL-PEG', q[q.coating == 'Uncoated'].low_SS_bright_per_s.mean() / q[q.coating == 'PLL-PEG'].low_SS_bright_per_s.mean())
    add('antibody rate', 'free-antibody median B525-H, bare; PLL-PEG', f"{q[q.coating == 'Uncoated'].free_probe_median_B525.mean():.0f}; {q[q.coating == 'PLL-PEG'].free_probe_median_B525.mean():.0f}")

    # 5 delta U
    smp = pd.read_csv(os.path.join(OUT, 'fig29d_tccd_ddG_samples.csv'))
    smp = smp[smp.coating != 'Silica'].merge(q[['sample', 'R_B']], on='sample')
    smp['F'] = 2 ** smp.log2_fold_change_simple
    smp['excess per antibody event rate'] = smp.excess_simple / smp.R_B
    smp['relative enrichment 1 - 1/F'] = 1 - 1 / smp.F
    smp['excess (as in manuscript)'] = smp.excess_simple
    cols = ['excess (as in manuscript)', 'excess per antibody event rate', 'relative enrichment 1 - 1/F']
    sets = smp.groupby(['coating', 'replicate'])[cols].mean()
    sets.reset_index().to_csv(os.path.join(OUT, 'tccd_statistics_check_sets.csv'), index=False)
    for col in cols:
        x = sets.loc['Uncoated', col].values; y = sets.loc['PLL-PEG', col].values
        du = np.log(x.mean() / y.mean())
        se = np.sqrt((stats.sem(x) / x.mean()) ** 2 + (stats.sem(y) / y.mean()) ** 2)
        add(col, 'ratio of set means (bare / PLL-PEG)', x.mean() / y.mean()); add(col, 'delta U (kT) from ratio of means', du, f'delta-method SE {se:.3f}')
        bs = [np.log(np.mean(i) / np.mean(j)) for i in itertools.product(x, repeat=3) for j in itertools.product(y, repeat=3)]
        add(col, 'percentile bootstrap 95% interval (kT)', '%.2f to %.2f' % tuple(np.percentile(bs, [2.5, 97.5])), 'n = 3 per coating: too narrow')
        d = np.log(x) - np.log(y); tt = stats.ttest_1samp(d, 0); ci = stats.t.interval(0.95, 2, loc=d.mean(), scale=stats.sem(d))
        add(col, 'paired t on log set means: delta U (kT)', d.mean(), f'95% CI {ci[0]:.2f} to {ci[1]:.2f}, p {tt.pvalue:.3f}')
        add(col, 'Welch t on log set means, p', stats.ttest_ind(np.log(x), np.log(y), equal_var=False).pvalue)
        add(col, 'Welch t on raw set means, p', stats.ttest_ind(x, y, equal_var=False).pvalue)
        allv = np.concatenate([x, y]); obs = abs(x.mean() - y.mean())
        cnt = sum(abs(allv[list(i)].mean() - np.delete(allv, list(i)).mean()) >= obs - 1e-12 for i in itertools.combinations(range(6), 3))
        add(col, 'exact permutation two-sided p', cnt / 20)
        for nm, v in [('bare', x), ('PLL-PEG', y)]:
            add(col, f'{nm} set mean', v.mean(), f'SE {stats.sem(v):.3g}; one-sample t vs 0, p {stats.ttest_1samp(v, 0).pvalue:.3f}')
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, 'tccd_statistics_check.csv'), index=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 70)
    print(R.to_string(index=False))


if __name__ == '__main__':
    main()
