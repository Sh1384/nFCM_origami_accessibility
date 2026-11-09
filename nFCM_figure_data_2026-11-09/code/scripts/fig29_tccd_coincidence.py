"""Figure 29: two-colour coincidence detection (TCCD) metrics adapted to nFCM, per coating (never pooled).

Channels (both recorded completely, because the trigger is VSSC-1-H >= 200 OR B525-H >= T):
  A  origami   events in the shared origami gate (VSSC-1-H >= 200 up to the per-coating cap, compensated
               V525-H limit); does not depend on B525-H
  B  probe     events with B525-H >= T at any side scatter (T = session B525 trigger: 250 antibody 04/18,
               150 streptavidin 05/25)
  coincident   origami-gate events with B525-H >= T
Counts are buffer-subtracted (session buffer, same criteria, scaled by acquisition time).
  f_c      = N_coinc / (N_A + N_B - N_coinc)                coincidence fraction
  E_chance = R_A * R_B * (2 tau) * t                        expected chance coincidences
  Q_A      = (N_coinc - E_chance) / (N_A + N_B)             association quotient
  obs/exp  = N_coinc / E_chance
  also: bound fraction of origami beyond chance = (N_coinc - E_chance) / N_A
2 tau (coincidence window) is estimated from the data: two particles closer in time than the instrument can
resolve are merged or lost, which shows up as a deficit of short inter-event intervals relative to a Poisson
process. 2 tau = sum over intervals 0-46 us of [1 - observed / Poisson-expected interval density] x bin width
(bins of 16 time ticks = 7.63 us, matching the timestamp quantisation), per file with >= 20,000 events; the session median is used. Sensitivity: 2 tau = median
VSSC-1-W of scatter-triggered events (pulse width; units assumed to be us), an upper bound.
Change vs no biotin: each extension sample vs the N + probe sample of the same coating and replicate.
Tubes are flagged (post hoc) if their gated origami rate is < 25% of their N pair or their low-scatter
fluorescence-triggered rate is > 3x; flagged tubes are drawn as X and excluded from bars and from the
'excluding flagged' statistics.
LIMITATION: E_chance counts only chance pairs in which both particles are detectable on their own. Most single
antibodies / streptavidins are below the B525 trigger; such a probe passing with an origami can still lift the
event's B525-H above T, and it is not in R_B. E_chance is therefore a lower bound, and Q_A > 0 does not by
itself show binding in this setup; the paired comparison with N + probe (same probe, same handling) is the test.

Run:  python fig29_tccd_coincidence.py
Out:  figures_svg/fig29_tccd_coincidence.svg and _samples.csv, _by_extension.csv, _correlations.csv,
      _coincidence_window.csv, _controls.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp
import make_manuscript_figures as M
from shared_utils import (CH_SS, CH_S, COATINGS, EXTENSIONS, REPLICATES, COAT_COLORS, SS_TRIGGER,
                          load_fcs, acq_time, compute_monomer_gates, apply_gate)

LOW_SS, MIN_EVENTS_WINDOW, XMAX = 50, 20000, 6 * 16 * 2.0 ** -21   # 6 bins of 7.63 us
PROBES = [('Anti-biotin IgG', 'Antibody session (04/18)', 250,
           ('probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs'), 'Antibody only (03/13)'),
          ('Streptavidin', 'Streptavidin session (05/25)', 150,
           ('probe_only_controls', 'Streptavidin_Only__Shervin5_03.fcs'), 'Streptavidin only (05/03)')]


def interval_deficit(time):
    """Inter-event interval density relative to Poisson; returns (2tau estimate, bin centres, ratio)."""
    t = np.sort(np.asarray(time, float)); dt = np.diff(t)
    bw = 16 * 2.0 ** -21                     # 16 ticks of 2^-21 s: timestamps cluster on 4/8/16-tick multiples
    R = len(dt) / (t[-1] - t[0])
    edges = np.arange(0, XMAX + bw, bw)
    obs = np.histogram(dt, edges)[0] / len(dt)
    exp = np.exp(-R * edges[:-1]) - np.exp(-R * edges[1:])
    ratio = obs / exp
    return float(np.sum((1 - ratio) * bw)), (edges[:-1] + edges[1:]) / 2, ratio


def tccd(d, t, g, T, buf, tb, gb, win):
    a = len(g) - len(gb) * t / tb
    b = (d[CH_S] >= T).sum() - (buf[CH_S] >= T).sum() * t / tb
    c = (g[CH_S] >= T).sum() - (gb[CH_S] >= T).sum() * t / tb
    out = dict(N_A=a, N_B=b, N_coinc=c, R_A=a / t, R_B=b / t)
    out['valid'] = a > 0 and b > 0
    out['f_c'] = c / (a + b - c) if out['valid'] else np.nan
    for lab, w in win.items():
        e = (a / t) * (b / t) * w * t if out['valid'] else np.nan
        out[f'E_chance_{lab}'] = e
        out[f'Q_A_{lab}'] = (c - e) / (a + b) if out['valid'] else np.nan
        out[f'obs_exp_{lab}'] = c / e if out['valid'] and e > 0 else np.nan
        out[f'bound_fraction_origami_{lab}'] = (c - e) / a if out['valid'] else np.nan
    return out


def compute():
    rows, ctrl, wins, curves = [], [], [], {}
    for probe, sname, T, (cf, cfile), clabel in PROBES:
        s = M.SESSIONS[sname]
        dfs = M.session_dfs(s, exts=tuple(EXTENSIONS))
        gates = compute_monomer_gates(dfs, ctl=s['noprobe'])
        buf = dfs['Buffer']; tb = acq_time(buf)
        # coincidence window from every file of the session with enough events
        est, pw = [], []
        for k, d in dfs.items():
            if isinstance(k, tuple) and len(d) >= MIN_EVENTS_WINDOW:
                w, x, r = interval_deficit(d['Time'].values)
                est.append(w); curves.setdefault(sname, []).append((x, r))
                pw.append(d[d[CH_SS] >= SS_TRIGGER]['VSSC-1-W'].median())
                wins.append(dict(session=sname, file='_'.join(k), n_events=len(d), two_tau_us=w * 1e6,
                                 median_VSSC1_W_scatter_triggered=pw[-1]))
        win = {'deficit': float(np.median(est)), 'pulse_width': float(np.median(pw)) * 1e-6}
        print(f'{sname}: 2tau from interval deficit {win["deficit"] * 1e6:.1f} us (IQR '
              f'{np.percentile(est, 25) * 1e6:.1f}-{np.percentile(est, 75) * 1e6:.1f}, n {len(est)}); '
              f'pulse-width alternative {win["pulse_width"] * 1e6:.1f} us')
        cd = load_fcs(cf, cfile); tc = acq_time(cd)
        for lab, d, t in [(clabel + ' (uncoated gate, no buffer subtraction)', cd, tc)]:
            g = apply_gate(d, 'NoCoating', 'A', gates)
            ctrl.append(dict(probe=probe, control=lab, **tccd(d, t, g, T, buf.iloc[:0], tb, g.iloc[:0], win)))
        for c in COATINGS:
            gb = apply_gate(buf, c, 'A', gates)
            for r in REPLICATES:
                d0 = dfs.get(('N', c, s['noprobe'], r))
                if d0 is not None:
                    t0 = acq_time(d0)
                    ctrl.append(dict(probe=probe, control=f'No-probe origami {M.COAT_LBL[c]} {r}',
                                     **tccd(d0, t0, apply_gate(d0, c, r, gates), T, buf, tb, gb, win)))
                for e in EXTENSIONS:
                    d = dfs.get((e, c, s['probe'], r))
                    if d is None:
                        continue
                    t = acq_time(d); g = apply_gate(d, c, r, gates)
                    low = d[(d[CH_SS] < LOW_SS) & (d[CH_S] >= T)]
                    rows.append(dict(probe=probe, session=sname, coating=M.COAT_LBL[c], replicate=r, extension=e,
                                     sample=f"{e}_{c}_{s['probe']}_{r}", paired_N_sample=f"N_{c}_{s['probe']}_{r}",
                                     B525_threshold=T, acq_time_s=t, two_tau_deficit_us=win['deficit'] * 1e6,
                                     two_tau_pulse_width_us=win['pulse_width'] * 1e6,
                                     low_SS_bright_per_s=len(low) / t, free_probe_median_B525=float(low[CH_S].median()),
                                     **tccd(d, t, g, T, buf, tb, gb, win)))
    D = pd.DataFrame(rows)
    key = ['probe', 'coating', 'replicate']
    N = D[D.extension == 'N'].set_index(key)

    def pair(col, how):
        out = []
        for v, k in zip(D[col], zip(D.probe, D.coating, D.replicate)):
            n = N.loc[k, col] if k in N.index else np.nan
            if not (np.isfinite(v) and np.isfinite(n)):
                out.append(np.nan)
            elif how == 'diff':
                out.append(v - n)
            else:
                out.append(np.log2(v / n) if v > 0 and n > 0 else np.nan)
        return out

    for q in ['f_c', 'Q_A_deficit', 'Q_A_pulse_width', 'bound_fraction_origami_deficit']:
        D[f'delta_{q}_pct_points'] = [x * 100 if np.isfinite(x) else np.nan for x in pair(q, 'diff')]
    D['log2_obs_exp_deficit_vs_N'] = pair('obs_exp_deficit', 'log2')
    D['log2_origami_rate_ratio'] = pair('R_A', 'log2')
    D['change_free_probe_median'] = pair('free_probe_median_B525', 'diff')
    D['low_SS_rate_ratio'] = np.exp2(pair('low_SS_bright_per_s', 'log2'))
    D['tube_flag'] = (np.exp2(D.log2_origami_rate_ratio) < 0.25) | (D.low_SS_rate_ratio > 3)
    dcols = [c_ for c_ in D.columns if c_.startswith('delta_') or c_ == 'log2_obs_exp_deficit_vs_N']
    D.loc[D.extension == 'N', dcols] = np.nan
    return D, pd.DataFrame(ctrl), pd.DataFrame(wins), curves


CHANGES = ['delta_Q_A_deficit_pct_points', 'delta_Q_A_pulse_width_pct_points', 'delta_f_c_pct_points',
           'delta_bound_fraction_origami_deficit_pct_points', 'log2_obs_exp_deficit_vs_N']


def summarise(D):
    by, cor = [], []
    X = D[D.extension != 'N']
    for subset, Z in [('all valid', X), ('excluding flagged tubes', X[~X.tube_flag])]:
        for (probe, coat), z in Z.groupby(['probe', 'coating'], sort=False):
            for col in CHANGES:
                for e in ['3', '6', '12']:
                    v = z[z.extension == e][col].dropna()
                    by.append(dict(subset=subset, probe=probe, coating=coat, quantity=col, extension=e,
                                   mean=v.mean() if len(v) else np.nan, sem=v.sem() if len(v) >= 2 else np.nan,
                                   n=len(v), p_vs_0=ttest_1samp(v, 0).pvalue if len(v) >= 2 else np.nan))
                for xcol in ['change_free_probe_median', 'log2_origami_rate_ratio']:
                    zz = z.dropna(subset=[col, xcol])
                    pr = pearsonr(zz[xcol], zz[col]) if len(zz) >= 4 else None
                    cor.append(dict(subset=subset, probe=probe, coating=coat, quantity=col, versus=xcol, n=len(zz),
                                    pearson_r=pr.statistic if pr else np.nan, p=pr.pvalue if pr else np.nan))
    return pd.DataFrame(by), pd.DataFrame(cor)


def draw(D, W, curves, Cor):
    hues = [M.COAT_LBL[c] for c in COATINGS]
    pal = {M.COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}
    off = dict(zip(hues, [-0.27, 0.0, 0.27]))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 4, figsize=(32, 21))
    panels = [('f_c', 'Coincidence fraction f$_c$ (%)', EXTENSIONS, 100, False),
              ('obs_exp_deficit', 'Observed / chance coincidences', EXTENSIONS, 1, True),
              ('Q_A_deficit', 'Association quotient Q$_A$ (%)', EXTENSIONS, 100, False),
              ('delta_Q_A_deficit_pct_points', 'Δ Q$_A$ vs own N (points)', ['3', '6', '12'], 1, False)]
    for pi, (probe, *_r) in enumerate(PROBES):
        P = D[D.probe == probe]
        for qi, (col, ylab, exts, mult, logy) in enumerate(panels):
            ax = axes[pi, qi]
            for coat in hues:
                for i, e in enumerate(exts):
                    z = P[(P.coating == coat) & (P.extension == e)]
                    ok = z[~z.tube_flag][col].dropna() * mult; fl = z[z.tube_flag][col].dropna() * mult
                    x = i + off[coat]
                    if len(ok):
                        ax.bar(x, ok.mean(), 0.25, color=pal[coat], edgecolor='black', label=coat if i == 0 else None)
                        if len(ok) >= 2:
                            ax.errorbar(x, ok.mean(), ok.sem(), color='black', capsize=3, lw=1.2)
                        ax.scatter(x + rng.uniform(-0.05, 0.05, len(ok)), ok, s=22, color='black', zorder=3)
                    if len(fl):
                        ax.scatter(np.full(len(fl), x), fl, s=110, marker='X', color='white', edgecolor='black', zorder=4)
            ax.set_xticks(range(len(exts))); ax.set_xticklabels(exts)
            ax.set_xlabel('Biotin extension (nt)' if qi == 3 else 'Sample (N = no biotin + probe)')
            ax.set_ylabel(ylab)
            if logy:
                ax.set_yscale('log'); ax.axhline(1, color='red', ls='--', lw=1.2)
            if qi in (2, 3):
                ax.axhline(0, color='red', ls='--', lw=1.2)
            ax.set_title(f"{'ABCD'[qi]}{pi + 1}  {probe}", fontsize=14); ax.legend(fontsize=10, frameon=False)
    # coincidence window
    ax = axes[2, 0]
    for (sname, cl), colr in zip(curves.items(), ['#377EB8', '#E41A1C']):
        x = cl[0][0]; R = np.vstack([r for _, r in cl])
        ax.plot(x * 1e6, np.median(R, 0), color=colr, lw=2, label=f'{sname} (median of {len(cl)} files)')
        ax.fill_between(x * 1e6, np.percentile(R, 25, 0), np.percentile(R, 75, 0), color=colr, alpha=0.2)
    ax.axhline(1, color='0.4', ls='--', lw=1)
    ax.set_xlabel('Interval between consecutive events (µs)'); ax.set_ylabel('Observed / Poisson-expected density')
    ax.set_title('E  Short intervals are depleted: coincidence window', fontsize=14); ax.legend(fontsize=10, frameon=False)
    ax = axes[2, 1]
    for i, (sname, z) in enumerate(W.groupby('session', sort=False)):
        ax.scatter(np.full(len(z), i) + rng.uniform(-0.12, 0.12, len(z)), z.two_tau_us, s=40, color='#54278F',
                   edgecolor='black', label='2τ from interval deficit' if i == 0 else None)
        ax.scatter(np.full(len(z), i + 0.35) + rng.uniform(-0.08, 0.08, len(z)), z.median_VSSC1_W_scatter_triggered,
                   s=40, color='#BDBDBD', edgecolor='black', label='median VSSC-1-W (scatter-triggered)' if i == 0 else None)
        ax.text(i, z.two_tau_us.median(), f' {z.two_tau_us.median():.1f}', va='center', fontsize=11)
    ax.set_xticks([0.17, 1.17]); ax.set_xticklabels([s.split(' (')[0] + '\n' + s.split(' ')[-1] for s in W.session.unique()])
    ax.set_ylabel('Window (µs) / pulse width'); ax.set_title('F  Coincidence window per file', fontsize=14)
    ax.legend(fontsize=10, frameon=False)
    for pi, (probe, *_r) in enumerate(PROBES):
        ax = axes[2, 2 + pi]; P = D[(D.probe == probe) & (D.extension != 'N')]
        txt = []
        for coat in hues:
            z = P[P.coating == coat].dropna(subset=['delta_Q_A_deficit_pct_points', 'change_free_probe_median'])
            ax.scatter(z.change_free_probe_median, z.delta_Q_A_deficit_pct_points, s=80, color=pal[coat],
                       edgecolor='black', label=coat, zorder=3)
            fz = z[z.tube_flag]
            ax.scatter(fz.change_free_probe_median, fz.delta_Q_A_deficit_pct_points, s=130, marker='X', color='white',
                       edgecolor='black', zorder=4)
            cb = Cor[(Cor.subset == 'all valid') & (Cor.probe == probe) & (Cor.coating == coat) &
                     (Cor.quantity == 'delta_Q_A_deficit_pct_points') & (Cor.versus == 'change_free_probe_median')]
            co = Cor[(Cor.subset == 'all valid') & (Cor.probe == probe) & (Cor.coating == coat) &
                     (Cor.quantity == 'delta_Q_A_deficit_pct_points') & (Cor.versus == 'log2_origami_rate_ratio')]
            if len(cb):
                txt.append(f'{coat}: r(brightness) {cb.iloc[0].pearson_r:+.2f}, r(origami) {co.iloc[0].pearson_r:+.2f}, '
                           f'n {int(cb.iloc[0].n)}')
        ax.axhline(0, color='0.6', lw=1); ax.axvline(0, color='0.6', lw=1)
        ax.text(0.02, 0.98, '\n'.join(txt), transform=ax.transAxes, va='top', fontsize=10, zorder=5,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
        ax.set_xlabel('Change in free-probe brightness vs N\n(median B525-H, VSSC-1-H < 50)')
        ax.set_ylabel('Δ Q$_A$ vs own N (points)')
        ax.set_title(f"{'GH'[pi]}  {probe}: Δ Q$_A$ vs brightness", fontsize=14); ax.legend(fontsize=10, frameon=False,
                                                                                        loc='lower right')
    fig.suptitle('Two-colour coincidence detection metrics adapted to nFCM (A = origami gate, B = B525-H above the '
                 'trigger), per coating; chance coincidences from a data-derived coincidence window', y=1.01)
    return fig


def main():
    D, C, W, curves = compute()
    By, Cor = summarise(D)
    fig = draw(D, W, curves, Cor)
    M.save(fig, 'fig29_tccd_coincidence', {'_samples': D, '_by_extension': By, '_correlations': Cor,
                                           '_coincidence_window': W, '_controls': C})
    pd.set_option('display.width', 220)
    print(C[['probe', 'control', 'N_A', 'N_B', 'N_coinc', 'f_c', 'Q_A_deficit', 'obs_exp_deficit']].round(4).to_string(index=False))
    lv = D.groupby(['probe', 'coating', 'extension'], sort=False)[['f_c', 'obs_exp_deficit', 'Q_A_deficit',
                                                                   'Q_A_pulse_width']].mean()
    print('\nMean levels:'); print(lv.round(4).to_string())
    show = By[(By.subset == 'excluding flagged tubes') & (By.quantity == 'delta_Q_A_deficit_pct_points')]
    print('\nDelta Q_A vs own N (points), excluding flagged tubes:')
    print(show.pivot_table(index=['probe', 'coating'], columns='extension', values='mean')[['3', '6', '12']].round(2).to_string())
    print('\nFlagged tubes:', D.loc[D.tube_flag, 'sample'].tolist(), '; invalid:', D.loc[~D.valid, 'sample'].tolist())


if __name__ == '__main__':
    main()
