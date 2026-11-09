"""Figure 5G values with matched spacer lengths: pooled delta U of the PLL-PEG coating from the excess and from log2FC.

Within each replicate set (A, B, C) only the spacer lengths measured for BOTH coatings are used: PLL-PEG, 12 nt,
replicate C had no detectable origami, so the bare 12-nt sample of set C is left out as well (sets A and B use 3, 6
and 12 nt; set C uses 3 and 6 nt). The matched samples of a coating are averaged within the set, then
    from excess   delta U = kT ln(mean excess_bare / mean excess_PLL-PEG)
    from log2FC   delta U = kT ln[(F_bare - 1) / (F_PLL-PEG - 1)],  F = 2 ** mean(log2FC)
Summary per method: mean and SD of the three per-set values, SE, t-based 95% CI and paired t-test p against zero.

Input:  figures_svg/fig35_antibody_length_dependence_samples.csv
Out:    figures_svg/fig5g_delta_U_matched_lengths.csv         mean, SD, SE, 95% CI, p and the per-set values
        figures_svg/fig5g_delta_U_matched_lengths_prism.csv   the three per-set values, one column per method
Run:    python fig5g_delta_U_matched_lengths.py
"""
import os
import numpy as np, pandas as pd
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures_svg')


def main():
    q = pd.read_csv(os.path.join(OUT, 'fig35_antibody_length_dependence_samples.csv'))
    q = q[q.spacer_nt.notna()].copy()
    q['excess'] = q.excess_pct_of_detected_origami; q['log2FC'] = np.log2(q.fold_change_vs_no_biotin)
    # keep a (replicate, spacer) only if both coatings have it
    n = q.groupby(['replicate', 'spacer_nt']).coating.nunique()
    q = q[[n[(r, s)] == 2 for r, s in zip(q.replicate, q.spacer_nt)]]
    used = q.groupby('replicate').spacer_nt.agg(lambda z: ', '.join(f'{v:g}' for v in sorted(set(z))))
    g = q.groupby(['replicate', 'coating'])[['excess', 'log2FC']].mean().unstack('coating')
    du = pd.DataFrame({
        'From excess': np.log(g[('excess', 'Uncoated')] / g[('excess', 'PLL-PEG')]),
        'From log2FC': np.log((2 ** g[('log2FC', 'Uncoated')] - 1) / (2 ** g[('log2FC', 'PLL-PEG')] - 1))})
    rows = []
    for m in du.columns:
        v = du[m].values; ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
        rows.append(dict(method=m, mean_delta_U_kT=v.mean(), sd_kT=v.std(ddof=1), se_kT=stats.sem(v), ci95_low_kT=ci[0], ci95_high_kT=ci[1],
                         p_paired_t_vs_zero=stats.ttest_1samp(v, 0).pvalue, n_replicate_sets=len(v),
                         **{f'set_{r}_kT': du.loc[r, m] for r in du.index},
                         **{f'set_{r}_spacer_lengths_nt': used[r] for r in du.index}))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, 'fig5g_delta_U_matched_lengths.csv'), index=False)
    du.rename_axis('replicate_set').reset_index().to_csv(os.path.join(OUT, 'fig5g_delta_U_matched_lengths_prism.csv'), index=False)
    pd.set_option('display.width', 220)
    print(S.round(3).T.to_string()); print(du.round(3))


if __name__ == '__main__':
    main()
