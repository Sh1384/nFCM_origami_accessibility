"""log2 fold change of the coincidence ratio (antibody session 04/18) and the delta U computed from it.

Per sample (biotin on a 3-, 6- or 12-nt spacer; bare, PLL-PEG and silica):
    fold change F = (observed / chance)_sample / (observed / chance)_matched no-biotin sample     (same coating and replicate)
    log2FC = log2 F          enrichment = F - 1 = biotin-specific coincidences / background coincidences
Per replicate set (A, B, C; bare and PLL-PEG share an origami preparation): the log2FC values of the spacer lengths are
averaged, F_set = 2 ** mean(log2FC), and
    delta U = kT ln[(F_bare - 1) / (F_PLL-PEG - 1)]                  in units of kT
F - 1 is the specific signal per unit of background, and the background is proportional to the antibody event rate with
the same constant for both coatings (observed / chance without biotin: 10.9 bare, 10.8 PLL-PEG), so this is the
per-antibody version of delta U; it does not assume equal antibody concentration in the two coatings.
Summary: mean of the three per-set values, SE, t-based 95% CI (2 degrees of freedom), paired t-test vs zero; and the
same quantity from the coating means (ratio of mean enrichments). Silica is listed at sample level only (confounded by
particulates; not used for delta U).

Inputs: figures_svg/fig29b_tccd_summary_points.csv, tccd_acquisition_order.csv
Out:    figures_svg/tccd_log2FC_and_delta_U.csv   (column 'level': sample / replicate set / summary)
Run:    python tccd_log2fc_delta_U.py
"""
import os
import numpy as np, pandas as pd
from scipy import stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures_svg')
LBL = {'Uncoated': 'Bare', 'PLL-PEG': 'PLL-PEG', 'Silica': 'Silica'}


def main():
    p = pd.read_csv(os.path.join(OUT, 'fig29b_tccd_summary_points.csv'))
    p = p[p.probe.str.contains('IgG') & (p.extension.astype(str) != 'N')].copy()
    t = pd.read_csv(os.path.join(OUT, 'tccd_acquisition_order.csv'))[['sample', 'order_in_set', 'minutes_since_first_sample_of_set', 'R_B', 'R_A']]
    p = p.merge(t, on='sample', how='left')
    rows = []
    p['_c'] = p.coating.map({'Uncoated': 0, 'PLL-PEG': 1, 'Silica': 2}); p['_e'] = p.extension.astype(int)
    for r in p.sort_values(['_c', 'replicate', '_e']).itertuples():
        rows.append(dict(level='sample', coating=LBL[r.coating], replicate_set=r.replicate, spacer_nt=int(r.extension), sample=r.sample,
                         matched_no_biotin_sample=r.paired_N_sample, observed_over_chance=r.observed_over_chance,
                         fold_change=r.fold_change_vs_N, log2FC=r.log2_fold_change_vs_N,
                         enrichment_pct=100 * (r.fold_change_vs_N - 1) if np.isfinite(r.fold_change_vs_N) else np.nan,
                         antibody_events_per_s=r.R_B, origami_events_per_s=r.R_A, order_in_set=r.order_in_set,
                         minutes_since_no_biotin_sample=r.minutes_since_first_sample_of_set,
                         note='' if np.isfinite(r.log2_fold_change_vs_N) else 'no detectable origami after buffer subtraction; excluded'))
    S = pd.DataFrame(rows)

    g = S[S.coating.isin(['Bare', 'PLL-PEG'])].groupby(['coating', 'replicate_set']).log2FC.agg(['mean', 'count']).unstack(0)
    du = []
    for rep in g.index:
        lb, lp = g.loc[rep, ('mean', 'Bare')], g.loc[rep, ('mean', 'PLL-PEG')]
        fb, fp = 2 ** lb, 2 ** lp
        d = float(np.log((fb - 1) / (fp - 1))); du.append(d)
        rows.append(dict(level='replicate set', coating='Bare vs PLL-PEG', replicate_set=rep, log2FC_bare=lb, log2FC_PLL_PEG=lp,
                         fold_change_bare=fb, fold_change_PLL_PEG=fp, enrichment_pct_bare=100 * (fb - 1), enrichment_pct_PLL_PEG=100 * (fp - 1),
                         ratio_of_enrichments=(fb - 1) / (fp - 1), delta_U_kT=d,
                         note=f"mean log2FC over {int(g.loc[rep, ('count', 'Bare')])} bare and {int(g.loc[rep, ('count', 'PLL-PEG')])} PLL-PEG spacer lengths"))
    du = np.array(du); ci = stats.t.interval(0.95, len(du) - 1, loc=du.mean(), scale=stats.sem(du))
    rows.append(dict(level='summary', coating='Bare vs PLL-PEG', replicate_set='mean of A, B, C', delta_U_kT=du.mean(), delta_U_se_kT=stats.sem(du),
                     delta_U_ci95_low_kT=ci[0], delta_U_ci95_high_kT=ci[1], p_paired_t_vs_zero=stats.ttest_1samp(du, 0).pvalue,
                     ratio_of_enrichments=float(np.exp(du.mean())), note='per-set values averaged; t-based interval, n = 3 replicate sets'))
    mb, mp = g[('mean', 'Bare')].mean(), g[('mean', 'PLL-PEG')].mean()
    rows.append(dict(level='summary', coating='Bare vs PLL-PEG', replicate_set='from coating means', log2FC_bare=mb, log2FC_PLL_PEG=mp,
                     fold_change_bare=2 ** mb, fold_change_PLL_PEG=2 ** mp, enrichment_pct_bare=100 * (2 ** mb - 1),
                     enrichment_pct_PLL_PEG=100 * (2 ** mp - 1), ratio_of_enrichments=(2 ** mb - 1) / (2 ** mp - 1),
                     delta_U_kT=float(np.log((2 ** mb - 1) / (2 ** mp - 1))), note='delta U from the mean log2FC of each coating'))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, 'tccd_log2FC_and_delta_U.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(S[['coating', 'replicate_set', 'spacer_nt', 'observed_over_chance', 'fold_change', 'log2FC', 'enrichment_pct']].round(4).to_string(index=False))
    print(R[R.level != 'sample'].dropna(axis=1, how='all').round(4).T.to_string())


if __name__ == '__main__':
    main()
