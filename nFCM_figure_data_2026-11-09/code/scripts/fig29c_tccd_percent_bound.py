"""Figure 29c: convert the TCCD fold change (observed/chance vs own no-biotin) into percent of origami bound.

Antibody session (04/18) only (streptavidin coincidences are dominated by streptavidin aggregates). Coatings are
never pooled. Inputs: fig29_tccd_coincidence_samples.csv (counts, fold change) and fig27d_mixture_model_samples.csv
(single-antibody brightness per replicate set and the sample's gain from its free antibody).

1  Detected biotin-specific excess (no model):
     F = (observed/chance)_ext / (observed/chance)_N   (same coating and replicate)
     If biotin added nothing, the coincidences would be the N sample's coincidences per origami per detectable
     antibody, scaled to this sample's rates, i.e. N_coinc / F. Excess = N_coinc (1 - 1/F).
     excess_detected_pct = 100 * N_coinc / N_A * (1 - 1/F)     (% of gated origami; can be negative)
2  Detection efficiency of one bound antibody: an origami counts as coincident only if its B525-H >= 250. Binding
   one antibody adds Y to the origami's own B525-H X. Efficiency = P(X + Y >= 250) - P(X >= 250)
   X: the sample's own gated B525-H events; Y: lognormal, median = single-antibody median of the replicate set
   (truncated-lognormal fit to its N + antibody free antibody, fig27d) x the sample's free-antibody gain,
   sigma 0.303. Sensitivity: antibody median x 0.6 and x 1.5.
3  Percent bound (biotin-specific, efficiency-corrected) = excess_detected_pct / efficiency.
Statistics per coating x extension (mean, SEM, one-sample t-test vs 0), and per coating with the three extensions
averaged within each replicate set first (they share one N) = primary. Percentages refer to origami above the
scatter trigger in the origami gate, relative to the no-biotin + antibody sample (non-specific binding and chance
coincidences are subtracted through that comparison). Run order (N acquired first) is not corrected.

Run (after fig29_tccd_coincidence.py and make_manuscript_figures.py fig27d):  python fig29c_tccd_percent_bound.py
Out:  figures_svg/fig29c_tccd_percent_bound.svg, _samples.csv, _summary.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, t as tdist
import make_manuscript_figures as M
from shared_utils import CH_S, COATINGS, EXTENSIONS, COAT_COLORS, compute_monomer_gates, apply_gate

T, SIGMA, NDRAW = 250, 0.303, 400_000
SCALES = {'main': 1.0, 'dim_x0.6': 0.6, 'bright_x1.5': 1.5}
HUES = [M.COAT_LBL[c] for c in COATINGS]
PAL = {M.COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}
OFF = dict(zip(HUES, [-0.27, 0.0, 0.27]))
INV = {v: k for k, v in M.COAT_LBL.items()}


def stats(v):
    v = pd.Series(v).dropna(); n = len(v)
    out = dict(n=n, mean=v.mean() if n else np.nan, sem=v.sem() if n >= 2 else np.nan)
    if n >= 2:
        h = tdist.ppf(0.975, n - 1) * out['sem']
        out.update(ci95_low=out['mean'] - h, ci95_high=out['mean'] + h, p_vs_0=ttest_1samp(v, 0).pvalue)
    else:
        out.update(ci95_low=np.nan, ci95_high=np.nan, p_vs_0=np.nan)
    return out


def main():
    D = pd.read_csv(os.path.join(M.OUT, 'fig29_tccd_coincidence_samples.csv'), dtype={'extension': str})
    G = pd.read_csv(os.path.join(M.OUT, 'fig27d_mixture_model_samples.csv'), dtype={'extension': str})
    D = D[(D.probe == 'Anti-biotin IgG') & (D.extension != 'N') & D.valid & ~D.tube_flag].dropna(
        subset=['log2_obs_exp_deficit_vs_N']).copy()
    D = D.merge(G[['sample', 'single_antibody_median_N', 's_free_antibody']], on='sample', how='left')
    s4 = M.SESSIONS['Antibody session (04/18)']
    dfs = M.session_dfs(s4, exts=tuple(EXTENSIONS))
    gates = compute_monomer_gates(dfs, ctl=s4['noprobe'])
    rng = np.random.default_rng(0)
    rows = []
    for r in D.itertuples():
        d = dfs[(r.extension, INV[r.coating], s4['probe'], r.replicate)]
        x = rng.choice(apply_gate(d, INV[r.coating], r.replicate, gates)[CH_S].values, NDRAW)
        F = 2 ** r.log2_obs_exp_deficit_vs_N
        row = dict(coating=r.coating, replicate=r.replicate, extension=r.extension, sample=r.sample,
                   paired_N_sample=r.paired_N_sample, N_A=r.N_A, N_coinc=r.N_coinc,
                   coincident_pct_of_origami=100 * r.N_coinc / r.N_A, fold_change_vs_N=F,
                   excess_detected_pct=100 * r.N_coinc / r.N_A * (1 - 1 / F),
                   single_antibody_median=r.single_antibody_median_N * r.s_free_antibody,
                   P_unbound_above_T=float(np.mean(x >= T)))
        for lab, sc in SCALES.items():
            y = rng.lognormal(np.log(row['single_antibody_median'] * sc), SIGMA, NDRAW)
            eff = float(np.mean(x + y >= T)) - row['P_unbound_above_T']
            row[f'detection_efficiency_{lab}'] = eff
            row[f'pct_bound_{lab}'] = row['excess_detected_pct'] / eff
        rows.append(row)
    P = pd.DataFrame(rows)
    summ = []
    cols = ['excess_detected_pct'] + [f'pct_bound_{l}' for l in SCALES] + [f'detection_efficiency_{l}' for l in SCALES]
    for coat in HUES:
        z = P[P.coating == coat]
        for e in ['3', '6', '12']:
            q = z[z.extension == e]
            for col in cols:
                summ.append(dict(coating=coat, extension=e, quantity=col, **stats(q[col])))
        sets = z.groupby('replicate')[cols].mean()
        for col in cols:
            summ.append(dict(coating=coat, extension='all extensions (per replicate set)', quantity=col, **stats(sets[col])))
    S = pd.DataFrame(summ)

    fig, axes = plt.subplots(1, 3, figsize=(27, 8.5))
    rngp = np.random.default_rng(1)
    for ax, col, ylab, title in [
            (axes[0], 'excess_detected_pct', 'Detected biotin-specific excess\n(% of gated origami)',
             'A  Detected excess: coincident % x (1 - 1/fold change)'),
            (axes[2], 'pct_bound_main', 'Biotin-specific % bound\n(corrected for detection efficiency)',
             'C  Estimated % of origami bound (vs no biotin)')]:
        for coat in HUES:
            for i, e in enumerate(['3', '6', '12', 'all']):
                if e == 'all':
                    v = P[P.coating == coat].groupby('replicate')[col].mean()
                else:
                    v = P[(P.coating == coat) & (P.extension == e)][col]
                x = i + OFF[coat]
                if len(v):
                    ax.bar(x, v.mean(), 0.25, color=PAL[coat], edgecolor='black', label=coat if i == 0 else None,
                           alpha=1 if e != 'all' else 0.6)
                    if len(v) >= 2:
                        ax.errorbar(x, v.mean(), v.sem(), color='black', capsize=3, lw=1.2)
                    ax.scatter(x + rngp.uniform(-0.05, 0.05, len(v)), v, s=22, color='black', zorder=3)
                    if col == 'pct_bound_main':
                        for lab, mk in [('dim_x0.6', '_'), ('bright_x1.5', '_')]:
                            vv = (P[P.coating == coat].groupby('replicate')[f'pct_bound_{lab}'].mean() if e == 'all'
                                  else P[(P.coating == coat) & (P.extension == e)][f'pct_bound_{lab}'])
                            ax.scatter(x, vv.mean(), marker=mk, s=300, color='#E6862E', linewidth=2.5, zorder=4,
                                       label='antibody brightness x0.6 / x1.5' if (coat, e, lab) == ('Uncoated', '3', 'dim_x0.6') else None)
        ax.axhline(0, color='red', ls='--', lw=1.2)
        ax.set_xticks(range(4)); ax.set_xticklabels(['3', '6', '12', 'all\n(per set)'])
        ax.set_xlabel('Biotin extension (nt)'); ax.set_ylabel(ylab); ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10, frameon=False, loc='upper left')
    ax = axes[1]
    for coat in HUES:
        z = P[P.coating == coat]
        ax.scatter(z.single_antibody_median, 100 * z.detection_efficiency_main, s=80, color=PAL[coat], edgecolor='black',
                   label=coat, zorder=3)
        for lab, mk in [('dim_x0.6', 'v'), ('bright_x1.5', '^')]:
            ax.scatter(z.single_antibody_median, 100 * z[f'detection_efficiency_{lab}'], s=30, marker=mk,
                       color=PAL[coat], alpha=0.5)
    ax.set_xlabel('Single-antibody median B525-H (set fit x sample gain)')
    ax.set_ylabel('Detection efficiency of one bound antibody (%)\n(circles; ▼ x0.6, ▲ x1.5 brightness)')
    ax.set_title('B  Fraction of bound origami that cross B525-H 250', fontsize=14)
    ax.legend(fontsize=10, frameon=False)
    fig.suptitle('Antibody session (04/18): TCCD fold change converted to percent of origami bound, each replicate vs '
                 'its own no-biotin + antibody sample (run-order confound not corrected)', y=1.03)
    M.save(fig, 'fig29c_tccd_percent_bound', {'_samples': P, '_summary': S})
    pd.set_option('display.width', 220)
    show = S[S.quantity.isin(['excess_detected_pct', 'pct_bound_main', 'pct_bound_dim_x0.6', 'pct_bound_bright_x1.5'])]
    show = show.assign(s=show.apply(lambda r: f"{r['mean']:+.2f}±{r['sem']:.2f} (p={r.p_vs_0:.2f})" if r.n >= 2
                                    else (f"{r['mean']:+.2f} (n=1)" if r.n == 1 else '-'), axis=1))
    print(show.pivot_table(index=['coating', 'quantity'], columns='extension', values='s', aggfunc='first').to_string())
    print('\nDetection efficiency (main), %:', (100 * P.groupby('coating').detection_efficiency_main.agg(['mean', 'min', 'max'])).round(1).to_dict())
    print('Coincident % of origami:', P.groupby('coating').coincident_pct_of_origami.agg(['mean', 'min', 'max']).round(2).to_dict())


if __name__ == '__main__':
    main()
