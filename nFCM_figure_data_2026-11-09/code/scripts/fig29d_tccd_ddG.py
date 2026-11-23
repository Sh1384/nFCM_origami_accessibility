"""Figure 29d: apparent binding free-energy penalty of PLL-PEG vs bare origami for the anti-biotin IgG,
from the TCCD-style biotin-specific excess (antibody session 04/18).

Per sample (each biotin sample vs the N + antibody sample of the same coating and replicate), rates in events/s:
  detected excess (% of gated origami) = 100 * [C - C_N * (A * B) / (A_N * B_N)] / A
      C coincident events, A origami-gate events, B probe events (B525-H >= 250); identical to
      coincident % x (1 - 1/fold change), the fold change being observed/chance vs own N.
  Null models for how coincidences scale without biotin:
      simple      B = all probe events                               (primary)
      free_probe  B = probe events not already coincident (B - C)
      free_probe_minus_aggregates (also subtracts probe aggregates in the gate, scaled from the
      antibody-only control) is shown only in the diagnostic: it makes the fold change track origami count
      (r = 0.76), so it is rejected.
  Efficiency-corrected excess = detected excess / detection efficiency of one bound antibody (fig29c).
Pooling: the extensions (3, 6, 12 nt) are averaged within each replicate set first (they share one N), then
the three set means are averaged per coating, i.e. one value per coating, extension length treated alike.
Because occupancy is far below saturation (theta ~ [Ab]/Kd), Kd_PLL / Kd_bare = excess_bare / excess_PLL with equal
antibody added, and
      ddG = RT ln(excess_bare / excess_PLL)          RT at 22 C (0.586 kcal/mol)
independent of the absolute antibody / origami concentrations. 95% CI: exhaustive bootstrap of the set means
(3^3 x 3^3 resamples); SE: delta method. Per-extension ddG (replicates within one extension) is a secondary check.
Silica shows no positive excess, so only a one-sided bound is given (upper 95% t-limit of the silica set mean).
Caveats: apparent penalty (a single antibody concentration cannot separate a uniform affinity loss from a fraction
of biotins being fully blocked); n = 3 sets per coating; acquisition order (N first) not corrected.

Run (after fig29_tccd_coincidence.py and fig29c_tccd_percent_bound.py):  python fig29d_tccd_ddG.py
Out:  figures_svg/fig29d_tccd_ddG.svg, _samples.csv, _sets.csv, _ddG.csv (kcal/mol, kJ/mol and kT, each with SE and
      95% CI), _null_diagnostics.csv
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, t as tdist
import make_manuscript_figures as M
from shared_utils import COATINGS, COAT_COLORS

T_C = 22.0
RT = 0.0019872 * (273.15 + T_C)                      # kcal/mol
NULLS = ['simple', 'free_probe']
HUES = [M.COAT_LBL[c] for c in COATINGS]
PAL = {M.COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}


def excess(r, n, mode, f_agg=0.0):
    """Detected biotin-specific excess, % of gated origami, for one sample r vs its N pair n."""
    be, bn = (r.R_B, n.R_B) if mode == 'simple' else (r.R_B - r.rC, n.R_B - n.rC)
    ce, cn = r.rC - f_agg * r.R_B, n.rC - f_agg * n.R_B
    return 100 * (ce - cn * (r.R_A * be) / (n.R_A * bn)) / r.R_A


def ddg_estimate(b, p):
    """ddG (kcal/mol) from two arrays of per-set (or per-replicate) excess values, with bootstrap CI and SE."""
    b, p = np.asarray(b, float), np.asarray(p, float)
    if b.mean() <= 0 or p.mean() <= 0:
        return dict(ratio=np.nan, ddG_kcal=np.nan, se_kcal=np.nan, ci95_low=np.nan, ci95_high=np.nan, n_boot=0)
    boots = [RT * np.log(np.mean(bb) / np.mean(pp)) for bb in itertools.product(b, repeat=len(b))
             for pp in itertools.product(p, repeat=len(p)) if np.mean(bb) > 0 and np.mean(pp) > 0]
    se = np.sqrt((b.std(ddof=1) / np.sqrt(len(b)) / b.mean()) ** 2 + (p.std(ddof=1) / np.sqrt(len(p)) / p.mean()) ** 2) \
        if len(b) > 1 and len(p) > 1 else np.nan
    ratio = b.mean() / p.mean()
    return dict(ratio=ratio, ddG_kcal=RT * np.log(ratio), se_kcal=RT * se,
                ci95_low=np.percentile(boots, 2.5), ci95_high=np.percentile(boots, 97.5), n_boot=len(boots))


def main():
    D = pd.read_csv(os.path.join(M.OUT, 'fig29_tccd_coincidence_samples.csv'), dtype={'extension': str, 'replicate': str})
    E = pd.read_csv(os.path.join(M.OUT, 'fig29c_tccd_percent_bound_samples.csv'), dtype={'extension': str})
    C = pd.read_csv(os.path.join(M.OUT, 'fig29_tccd_coincidence_controls.csv'))
    D = D[(D.probe == 'Anti-biotin IgG') & D.valid].copy()
    D['rC'] = D.N_coinc / D.acq_time_s
    N = D[D.extension == 'N'].set_index(['coating', 'replicate'])
    X = D[(D.extension != 'N') & ~D.tube_flag].merge(E[['sample', 'detection_efficiency_main', 'excess_detected_pct']],
                                                     on='sample', how='left')
    ab = C[C.control.str.startswith('Antibody only')].iloc[0]; f_agg = ab.N_coinc / ab.N_B
    rows, diag = [], []
    for r in X.itertuples():
        n = N.loc[(r.coating, r.replicate)]
        row = dict(coating=r.coating, replicate=r.replicate, extension=r.extension, sample=r.sample,
                   paired_N_sample=r.paired_N_sample, detection_efficiency=r.detection_efficiency_main,
                   change_free_antibody_median=r.change_free_probe_median, log2_origami_rate_ratio=r.log2_origami_rate_ratio)
        for mode in NULLS:
            row[f'excess_{mode}'] = excess(r, n, mode)
            row[f'excess_{mode}_eff_corrected'] = row[f'excess_{mode}'] / r.detection_efficiency_main
        for mode, fa, lab in [('simple', 0, 'simple'), ('free', 0, 'free_probe'), ('free', f_agg, 'free_probe_minus_aggregates')]:
            be, bn = (r.R_B, n.R_B) if mode == 'simple' else (r.R_B - r.rC, n.R_B - n.rC)
            row[f'log2_fold_change_{lab}'] = np.log2(((r.rC - fa * r.R_B) / (r.R_A * be)) / ((n.rC - fa * n.R_B) / (n.R_A * bn)))
        rows.append(row)
    P = pd.DataFrame(rows)
    chk = (P.excess_simple - X.set_index('sample').loc[P['sample'], 'excess_detected_pct'].values).abs().max()
    print(f'check vs fig29c excess_detected_pct: max |difference| = {chk:.2e}')
    for lab in ['simple', 'free_probe', 'free_probe_minus_aggregates']:
        y = P[f'log2_fold_change_{lab}']
        rb = pearsonr(P.change_free_antibody_median, y); ro = pearsonr(P.log2_origami_rate_ratio, y)
        diag.append(dict(null_model=lab, n=len(P), r_brightness=rb.statistic, p_brightness=rb.pvalue,
                         r_origami_count=ro.statistic, p_origami_count=ro.pvalue,
                         accepted=abs(rb.statistic) < 0.5 and abs(ro.statistic) < 0.5))
    Dg = pd.DataFrame(diag)
    qcols = [f'excess_{m}' for m in NULLS] + [f'excess_{m}_eff_corrected' for m in NULLS]
    Sets = P.groupby(['coating', 'replicate'])[qcols].mean().reset_index()
    Sets['n_extensions'] = P.groupby(['coating', 'replicate']).size().values
    est = []
    for col in qcols:
        mode = 'simple' if 'simple' in col else 'free_probe'
        corr = 'efficiency-corrected' if 'eff' in col else 'uncorrected'
        b = Sets[Sets.coating == 'Uncoated'][col].values; p = Sets[Sets.coating == 'PLL-PEG'][col].values
        est.append(dict(comparison='PLL-PEG vs bare', scope='pooled over extensions (per replicate set)', null_model=mode,
                        correction=corr, bare_mean=b.mean(), PLL_mean=p.mean(), n_bare=len(b), n_PLL=len(p), **ddg_estimate(b, p)))
        for e in ['3', '6', '12']:
            b = P[(P.coating == 'Uncoated') & (P.extension == e)][col].values
            p = P[(P.coating == 'PLL-PEG') & (P.extension == e)][col].values
            est.append(dict(comparison='PLL-PEG vs bare', scope=f'{e} nt only', null_model=mode, correction=corr,
                            bare_mean=b.mean(), PLL_mean=p.mean(), n_bare=len(b), n_PLL=len(p), **ddg_estimate(b, p)))
        s = Sets[Sets.coating == 'Silica'][col].values; b = Sets[Sets.coating == 'Uncoated'][col].values
        up = s.mean() + tdist.ppf(0.95, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))
        est.append(dict(comparison='silica vs bare', scope='pooled over extensions (per replicate set)', null_model=mode,
                        correction=corr, bare_mean=b.mean(), silica_mean=s.mean(), silica_upper95=up, n_bare=len(b),
                        n_silica=len(s), ddG_lower_bound_kcal=RT * np.log(b.mean() / up) if up > 0 else np.nan,
                        note='no positive silica excess; one-sided lower bound only'))
    G = pd.DataFrame(est)
    for c_ in ['ddG_kcal', 'se_kcal', 'ci95_low', 'ci95_high', 'ddG_lower_bound_kcal']:
        if c_ in G:
            G[c_.replace('kcal', 'kJ') if 'kcal' in c_ else c_ + '_kJ'] = G[c_] * 4.184
    for c_ in ['ddG', 'se', 'ci95_low', 'ci95_high', 'ddG_lower_bound']:
        G[f'{c_}_kT'] = G[f'{c_}_kcal' if c_ in ('ddG', 'se', 'ddG_lower_bound') else c_] / RT
    G.insert(0, 'temperature_C', T_C)
    G.insert(1, 'RT_kcal_per_mol', RT)

    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    rng = np.random.default_rng(0)
    off = dict(zip(HUES, [-0.27, 0.0, 0.27]))
    ax = axes[0, 0]
    for coat in HUES:
        for i, e in enumerate(['3', '6', '12']):
            v = P[(P.coating == coat) & (P.extension == e)]
            x = i + off[coat]
            ax.bar(x, v.excess_simple.mean(), 0.25, color=PAL[coat], edgecolor='black', alpha=0.45 if coat == 'Silica' else 1,
                   label=coat if i == 0 else None)
            for rr in v.itertuples():
                xx = x + rng.uniform(-0.06, 0.06)
                ax.scatter(xx, rr.excess_simple, s=30, color='black', zorder=3)
                ax.annotate(rr.replicate, (xx, rr.excess_simple), xytext=(4, 0), textcoords='offset points', fontsize=9)
    ax.axhline(0, color='red', ls='--', lw=1.2)
    ax.set_xticks(range(3)); ax.set_xticklabels(['3', '6', '12'])
    ax.set_xlabel('Biotin extension (nt)'); ax.set_ylabel('Detected biotin-specific excess\n(% of gated origami, simple null)')
    ax.set_title('A  Per-sample excess (letter = replicate set)', fontsize=14); ax.legend(fontsize=11, frameon=False)
    ax = axes[0, 1]
    labs = [('excess_simple', 'simple'), ('excess_free_probe', 'free probe')]
    for j, (col, lab) in enumerate(labs):
        for k, coat in enumerate(HUES):
            v = Sets[Sets.coating == coat][col]
            x = j + (k - 1) * 0.27
            ax.bar(x, v.mean(), 0.25, color=PAL[coat], edgecolor='black', alpha=0.45 if coat == 'Silica' else 1,
                   label=coat if j == 0 else None)
            ax.errorbar(x, v.mean(), v.sem(), color='black', capsize=3, lw=1.2)
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=30, color='black', zorder=3)
            ax.text(x, max(v.max(), v.mean() + v.sem()) + 0.08, f'{v.mean():.2f}', ha='center', fontsize=11)
    ax.axhline(0, color='red', ls='--', lw=1.2)
    ax.set_xticks(range(2)); ax.set_xticklabels(['simple null', 'free-probe null'])
    ax.set_ylabel('Detected excess, mean of replicate sets\n(% of gated origami; extensions averaged per set)')
    ax.set_title('B  One value per coating (extension lengths treated alike)', fontsize=14); ax.legend(fontsize=11, frameon=False)
    ax = axes[1, 0]
    F = G[G.comparison == 'PLL-PEG vs bare'].reset_index(drop=True)
    order = []
    for mode in NULLS:
        for corr in ['uncorrected', 'efficiency-corrected']:
            order.append((mode, corr, 'pooled over extensions (per replicate set)'))
    for e in ['3', '6', '12']:
        order.append(('simple', 'uncorrected', f'{e} nt only'))
    for yi, (mode, corr, scope) in enumerate(order):
        rr = F[(F.null_model == mode) & (F.correction == corr) & (F.scope == scope)].iloc[0]
        primary = (mode, corr, scope) == ('simple', 'uncorrected', 'pooled over extensions (per replicate set)')
        colr = '#54278F' if 'pooled' in scope else '#9E9AC8'
        ax.errorbar(rr.ddG_kcal, yi, xerr=[[rr.ddG_kcal - rr.ci95_low], [rr.ci95_high - rr.ddG_kcal]], fmt='D' if primary else 'o',
                    color=colr, ms=11 if primary else 8, capsize=4, lw=2)
        ax.text(max(rr.ci95_high, rr.ddG_kcal) + 0.05, yi,
                f'{rr.ddG_kcal:.2f} kcal/mol [{rr.ci95_low:.2f}, {rr.ci95_high:.2f}]\n'
                f'{rr.ddG_kT:.2f} ± {rr.se_kT:.2f} kT [{rr.ci95_low_kT:.2f}, {rr.ci95_high_kT:.2f}]',
                va='center', fontsize=10)
    ax.axvline(0, color='red', ls='--', lw=1.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{'pooled' if 'pooled' in s else s}, {m.replace('_', ' ')} null, {c}" for m, c, s in order], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('ΔΔG = RT ln(excess$_{bare}$ / excess$_{PLL-PEG}$)  (kcal/mol, 22 °C), 95% CI')
    ax.set_title('C  Apparent binding penalty of PLL-PEG (diamond = primary)', fontsize=14, pad=34)
    ax.set_xlim(left=min(-0.2, F.ci95_low.min() - 0.1), right=F.ci95_high.max() + 1.3)
    top = ax.secondary_xaxis('top', functions=(lambda v: v / RT, lambda v: v * RT))
    top.set_xlabel('ΔΔG (kT)')
    ax.text(0.99, 0.01, 'label: estimate [95% CI]; kT line: estimate ± SE (delta method) [95% CI]',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='0.3')
    ax = axes[1, 1]
    x = np.arange(len(Dg))
    ax.bar(x - 0.18, Dg.r_brightness, 0.34, color='#BDBDBD', edgecolor='black', label='r with brightness change')
    ax.bar(x + 0.18, Dg.r_origami_count, 0.34, color='#6A3D9A', edgecolor='black', label='r with origami-count change')
    for i, rr in Dg.iterrows():
        ax.text(i, max(rr.r_brightness, rr.r_origami_count, 0) + 0.05, 'accepted' if rr.accepted else 'rejected',
                ha='center', fontsize=12, color='black' if rr.accepted else 'red')
    ax.axhline(0, color='black', lw=1); ax.axhline(0.5, color='0.6', ls=':', lw=1); ax.axhline(-0.5, color='0.6', ls=':', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(['simple', 'free probe', 'free probe −\naggregates'])
    ax.set_ylabel('Pearson r of log$_2$ fold change (n = 25, all coatings)')
    ax.set_ylim(-1, 1.05)
    ax.set_title('D  Null-model check: fold change should not track confounds', fontsize=14)
    ax.legend(fontsize=11, frameon=False, loc='lower right')
    fig.suptitle('Anti-biotin IgG: apparent binding free-energy penalty of PLL-PEG vs bare origami (coincidence excess)\n'
                 'one value per coating; extensions pooled within replicate sets', y=1.04)
    M.save(fig, 'fig29d_tccd_ddG', {'_samples': P, '_sets': Sets, '_ddG': G, '_null_diagnostics': Dg})
    pd.set_option('display.width', 220)
    print(Sets.round(3).to_string(index=False))
    print(G[['comparison', 'scope', 'null_model', 'correction', 'ratio', 'ddG_kcal', 'se_kcal', 'ci95_low', 'ci95_high', 'ddG_kT', 'se_kT', 'ci95_low_kT', 'ci95_high_kT', 'ddG_lower_bound_kT',
             'ddG_lower_bound_kcal']].round(3).to_string(index=False))
    print(Dg.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
