"""Flow-cytometry panels of the proposed Figures 2-5 (nFCM_Figures_1-5_editable.pptx) that are not in the manuscript.

    2b  VSSC-1-H vs V525-H density of buffer and bare origami on identical axes and colour scale (batch 2)
    2c  Survival curve of recorded side scatter (fraction of scatter-triggered events >= x) with power-law fits
    2d  Scatter-triggered event rate per acquisition and the detected fraction, rate / (concentration x flow rate)
    4c  PLL-PEG-coated origami, batch 2: fluorescence-vs-scatter branch and its fitted log-log slope
    4f  PLL-PEG, batch 1: median ATTO 425 per scatter bin with the bare curve shifted 1.8-fold, and a fitted shift k
    5b  B525-H of origami-gate events vs that of low-scatter (free antibody) events, all antibody samples, one fit
    5f  Run-order check: observed/chance ratio and antibody event rate of every antibody acquisition vs acquisition time

Batch 1 = streptavidin session (05/25), batch 2 = antibody session (04/18). Acquisitions above 7,000 events/s are
excluded by load_fcs. Scatter-triggered = VSSC-1-H >= 200.

Out:  figures_svg/proposed_panels/<panel>.svg and <panel>_*.csv
Run:  python fig37_proposed_flow_panels.py [panel ...]      (default: all panels)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
import statsmodels.formula.api as smf
import make_manuscript_figures as M
from shared_utils import acq_time, bs_percentile

OUTD = os.path.join(M.OUT, 'proposed_panels'); os.makedirs(OUTD, exist_ok=True)
PREVIEW = os.environ.get('FIG_PREVIEW_DIR')
SS, F, B = 'VSSC-1-H', 'V525-H', 'B525-H'
TRIG = 200
COL = {'Buffer': '#8C8C8C', 'NoCoating': '#3354D1', 'PLL': '#F28E2B', 'Silica': '#2E9E4F', 'control': '#F7C08A'}
LBL = {'Buffer': 'buffer', 'NoCoating': 'bare', 'PLL': 'PLL-PEG', 'Silica': 'silica'}
B1, B2 = 'Streptavidin session (05/25)', 'Antibody session (04/18)'
BATCH = {B1: 'batch 1', B2: 'batch 2'}
INK, MUTED = '#222222', '#6b6b6b'
NOMINAL_NM = 0.16                                   # origami concentration at the sample port (Methods)

plt.rcdefaults()
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'svg.fonttype': 'none', 'legend.frameon': False, 'lines.linewidth': 1.5})


def save(fig, name, tables):
    fig.savefig(os.path.join(OUTD, f'{name}.svg'), bbox_inches='tight')
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, f'{name}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    for suffix, df in tables.items():
        df.to_csv(os.path.join(OUTD, f'{name}{suffix}.csv'), index=False)
    print(f'  wrote {name}.svg (+{len(tables)} csv)')


def span_edges(arrays, n):
    """Log-spaced bin edges covering every positive value of the given arrays (nothing clipped or dropped)."""
    v = np.concatenate([np.asarray(a, float) for a in arrays]); v = v[v > 0]
    return np.logspace(np.floor(np.log10(v.min()) * 2) / 2, np.ceil(np.log10(v.max() * 1.0001) * 2) / 2, n)


def files(sess, coat, tag=None, reps='ABC'):
    s = M.SESSIONS[sess]; tag = tag or s['noprobe']
    return [(r, d) for r in reps if (d := M.L(s['folder'], f'N_{coat}_{tag}_{r}.fcs')) is not None]


def buffer(sess):
    return M.L(M.SESSIONS[sess]['folder'], 'Buffer.fcs')


# ------------------------------------------------------------------------------------------------ 2b
def panel_2b():
    """Scatter-triggered events only (VSSC-1-H >= 200); x axis from 10^1 so the trigger edge is visible."""
    sess = B2
    samples = [('buffer', buffer(sess), 'Buffer.fcs'),
               ('bare origami (replicate B)', M.L(M.SESSIONS[sess]['folder'], 'N_NoCoating_NoAB_B.fcs'), 'N_NoCoating_NoAB_B.fcs')]
    trig = [(lab, d[d[SS] >= TRIG], d, fn) for lab, d, fn in samples]
    xhi = span_edges([z[SS] for _, z, _, _ in trig], 2)[-1]
    xb = np.logspace(1, np.log10(xhi), 121); yb = span_edges([z[F] for _, z, _, _ in trig], 121)
    hs, rows, summ = [], [], []
    for lab, z, d, fn in trig:
        t = acq_time(d)
        ok = z[F] > 0
        h = np.histogram2d(z[SS][ok], z[F][ok], [xb, yb])[0] / t
        hs.append(h)
        ii, jj = np.nonzero(h)
        rows += [dict(sample=lab, file=fn, ss_bin_lo=xb[i], ss_bin_hi=xb[i + 1], v525_bin_lo=yb[j], v525_bin_hi=yb[j + 1],
                      events_per_s=h[i, j]) for i, j in zip(ii, jj)]
        summ.append(dict(sample=lab, file=fn, acq_time_s=t, recorded_events=len(d), scatter_triggered_events=len(z),
                         scatter_triggered_events_per_s=len(z) / t, events_not_scatter_triggered_excluded=len(d) - len(z),
                         scatter_triggered_events_with_V525_le_0_not_plotted=int((~ok).sum())))
    vmin = min(h[h > 0].min() for h in hs); vmax = max(h.max() for h in hs)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6), sharey=True)
    for ax, (lab, z, d, fn), h, sm in zip(axes, trig, hs, summ):
        pc = ax.pcolormesh(xb, yb, np.where(h > 0, h, np.nan).T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis', rasterized=True)
        ax.axvline(TRIG, color='#D62728', ls='--', lw=1.2)
        ax.text(TRIG / 1.15, yb[-1] / 3, 'trigger', fontsize=7, color='#D62728', ha='right')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(10, xb[-1]); ax.set_ylim(yb[0], yb[-1])
        ax.set_xlabel('VSSC-1-H')
        ax.set_title(f"{lab}\n{sm['scatter_triggered_events_per_s']:.0f} scatter-triggered events/s", fontsize=9,
                     color=COL['Buffer'] if lab == 'buffer' else COL['NoCoating'])
    axes[0].set_ylabel('V525-H (ATTO 425)')
    cb = fig.colorbar(pc, ax=axes, pad=0.02, fraction=0.05); cb.set_label('events per second per bin', fontsize=8)
    save(fig, 'fig2b_buffer_vs_bare_density', {'_bins': pd.DataFrame(rows), '_summary': pd.DataFrame(summ)})


# ------------------------------------------------------------------------------------------------ 2c
def ccdf(v, grid):
    v = np.sort(v)
    return 1 - np.searchsorted(v, grid, side='left') / len(v)


def fit_tail(v):
    p95 = np.percentile(v, 95); grid = np.logspace(np.log10(TRIG), np.log10(p95), 40)
    lr = stats.linregress(np.log10(grid), np.log10(ccdf(v, grid)))
    hill = lambda xm: len(v[v >= xm]) / np.log(v[v >= xm] / xm).sum()
    return dict(n_events=len(v), median=np.median(v), p95=p95, ccdf_slope_200_to_p95=lr.slope, ccdf_slope_r2=lr.rvalue ** 2,
                hill_alpha_xmin200=hill(TRIG), hill_alpha_xmin500=hill(500), hill_alpha_xmin1000=hill(1000))


def panel_2c():
    allmax = max(float(d[SS].max()) for sess in [B1, B2] for c in ['NoCoating', 'PLL', 'Silica'] for _, d in files(sess, c))
    allmax = max(allmax, float(buffer(B1)[SS].max()), float(buffer(B2)[SS].max()))
    grid = np.logspace(np.log10(TRIG), np.log10(allmax), 240)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
    curves, fits = [], []
    for ax, sess in zip(axes, [B1, B2]):
        buf = buffer(sess)
        groups = [('Buffer', [('', buf)])] + [(c, files(sess, c)) for c in ['NoCoating', 'PLL', 'Silica']]
        for c, fl in groups:
            v = np.concatenate([d[SS].values[d[SS].values >= TRIG] for _, d in fl])
            y = ccdf(v, grid)
            ax.plot(grid, np.where(y > 0, y, np.nan), color=COL[c], lw=1.6 if c == 'NoCoating' else 1.2,
                    ls=':' if c == 'Buffer' else '-', label=LBL[c], zorder=3 if c == 'NoCoating' else 2)
            curves += [dict(batch=BATCH[sess], session=sess, sample=LBL[c], replicates='pooled', vssc1h=x, fraction_ge_x=f)
                       for x, f in zip(grid, y)]
            fits.append(dict(batch=BATCH[sess], session=sess, sample=LBL[c], replicate='pooled', **fit_tail(v)))
            for r, d in (fl if c != 'Buffer' else []):
                fits.append(dict(batch=BATCH[sess], session=sess, sample=LBL[c], replicate=r,
                                 **fit_tail(d[SS].values[d[SS].values >= TRIG])))
        fb = [f for f in fits if f['session'] == sess and f['sample'] == 'bare' and f['replicate'] == 'pooled'][0]
        xs = np.logspace(np.log10(TRIG), np.log10(fb['p95'] * 2.5), 20)
        ax.plot(xs, (xs / TRIG) ** fb['ccdf_slope_200_to_p95'], color=INK, ls='--', lw=1, zorder=4)
        for q, lab in [('median', 'median'), ('p95', '95th pct')]:
            x0 = fb[q]; y0 = ccdf(np.concatenate([d[SS].values[d[SS].values >= TRIG] for _, d in files(sess, 'NoCoating')]), [x0])[0]
            ax.plot([x0], [y0], 'o', mfc='white', mec=COL['NoCoating'], ms=5, zorder=5)
            ax.text(x0 * 1.12, y0 * 1.15, f'{lab} {x0:.0f}', fontsize=7, color=COL['NoCoating'])
        ax.text(0.97, 0.80, f"bare: slope {fb['ccdf_slope_200_to_p95']:.1f}\n(fit 200 to 95th pct)", transform=ax.transAxes,
                ha='right', fontsize=7, color=INK)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(3e-6, 1.3); ax.set_xlim(TRIG, allmax * 1.2)
        ax.set_xlabel('VSSC-1-H of recorded events'); ax.set_title(BATCH[sess], fontsize=9)
    axes[0].set_ylabel('fraction of events ≥ x'); axes[1].legend(loc='lower left')
    save(fig, 'fig2c_side_scatter_survival', {'_curves': pd.DataFrame(curves), '_fits': pd.DataFrame(fits)})


# ------------------------------------------------------------------------------------------------ 2d
def panel_2d():
    rows = []
    for sess in [B1, B2]:
        s = M.SESSIONS[sess]; bu = buffer(sess); tb = acq_time(bu); brate = (bu[SS] >= TRIG).sum() / tb
        a = M.acq_record(s['folder'], 'Buffer.fcs')
        rows.append(dict(batch=BATCH[sess], session=sess, sample='buffer', replicate='', file='Buffer.fcs',
                         scatter_events_per_s=brate, flow_ul_per_min=60 * a['Volume'] / a['Duration']))
        for r, d in files(sess, 'NoCoating'):
            a = M.acq_record(s['folder'], f"N_NoCoating_{s['noprobe']}_{r}.fcs"); q = a['Volume'] / a['Duration']   # µL/s
            rate = (d[SS] >= TRIG).sum() / acq_time(d)
            flux = NOMINAL_NM * 1e-9 * 6.022e23 * q * 1e-6
            rows.append(dict(batch=BATCH[sess], session=sess, sample='bare origami', replicate=r,
                             file=f"N_NoCoating_{s['noprobe']}_{r}.fcs", scatter_events_per_s=rate, flow_ul_per_min=60 * q,
                             buffer_scatter_events_per_s=brate, nominal_concentration_nM=NOMINAL_NM,
                             nominal_particles_per_s=flux, detected_fraction_pct=100 * (rate - brate) / flux))
    R = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    mk = {'batch 1': dict(marker='^', mfc=None), 'batch 2': dict(marker='o', mfc='white')}
    ax = axes[0]
    for i, smp in enumerate(['buffer', 'bare origami']):
        z = R[R['sample'] == smp]; colr = COL['Buffer'] if smp == 'buffer' else COL['NoCoating']
        ax.hlines(z.scatter_events_per_s.mean(), i - 0.25, i + 0.25, color=colr, lw=1.5)
        for j, (b, zz) in enumerate(z.groupby('batch')):
            ax.plot(i + (j - 0.5) * 0.18 + np.linspace(-0.04, 0.04, len(zz)), zz.scatter_events_per_s, ls='', ms=6,
                    color=colr, mec=colr, marker=mk[b]['marker'], mfc=mk[b]['mfc'] or colr, label=b if i == 1 else None)
    ax.set_yscale('log'); ax.set_ylim(10, 3000); ax.set_xticks([0, 1]); ax.set_xticklabels(['buffer', 'bare\norigami'])
    ax.set_xlim(-0.6, 1.6); ax.set_ylabel('scatter-triggered events per second'); ax.legend(loc='upper left')
    ax = axes[1]; z = R[R['sample'] == 'bare origami']
    for j, (b, zz) in enumerate(z.groupby('batch')):
        ax.plot(j + np.linspace(-0.08, 0.08, len(zz)), zz.detected_fraction_pct, ls='', ms=6, color=COL['NoCoating'],
                marker=mk[b]['marker'], mfc=mk[b]['mfc'] or COL['NoCoating'])
        ax.hlines(zz.detected_fraction_pct.mean(), j - 0.25, j + 0.25, color=COL['NoCoating'], lw=1.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['batch 1', 'batch 2']); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, None)
    ax.set_ylabel('origami recorded (% of delivered)')
    ax.set_title(f'rate ÷ (c × Q)\nc = {NOMINAL_NM} nM, Q = 1 µL/min', fontsize=8, color=MUTED)
    save(fig, 'fig2d_event_rate_detected_fraction', {'_acquisitions': R})


# ------------------------------------------------------------------------------------------------ 4c
def branch_medians(d, edges, fmin):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d[SS] >= a) & (d[SS] < b); f = d.loc[m, F].values; br = f[f > fmin]
        out.append(dict(ss_bin_lo=a, ss_bin_hi=b, ss_bin_center=np.sqrt(a * b), events=int(m.sum()), branch_events=len(br),
                        branch_fraction=len(br) / max(m.sum(), 1), branch_median_v525=np.median(br) if len(br) >= 30 else np.nan))
    return pd.DataFrame(out)


def slope_fit(T, lo, hi):
    z = T[(T.ss_bin_center >= lo) & (T.ss_bin_center <= hi) & T.branch_median_v525.notna()]
    if len(z) < 3:
        return np.nan, np.nan, len(z)
    lr = stats.linregress(np.log10(z.ss_bin_center), np.log10(z.branch_median_v525))
    return lr.slope, lr.stderr, len(z)


def branch_fraction(d, ss_min=1000, f_min=200):
    x = M.scat(d)
    return 100 * ((x[SS] >= ss_min) & (x[F] > f_min)).mean()


def panel_4c():
    sess = B2; edges = np.logspace(np.log10(TRIG), 5, 16)
    meds, fits, frac = [], [], []
    reps = {c: dict(files(sess, c)) for c in ['PLL', 'NoCoating']}
    for c in reps:
        for r, d in [('pooled', pd.concat([M.scat(v) for v in reps[c].values()]))] + [(r, M.scat(v)) for r, v in reps[c].items()]:
            for fmin in (200, 300):
                T = branch_medians(d, edges, fmin)
                meds.append(T.assign(coating=LBL[c], replicate=r, branch_v525_cut=fmin))
                for lo, hi in [(2400, 7e4), (1500, 7e4), (2400, 2e4)]:
                    sl, se, n = slope_fit(T, lo, hi)
                    fits.append(dict(coating=LBL[c], replicate=r, branch_v525_cut=fmin, fit_ss_lo=lo, fit_ss_hi=hi,
                                     slope=sl, slope_se=se, n_bins=n))
    for sess_ in [B1, B2]:
        frac.append(dict(batch=BATCH[sess_], coating='buffer', replicate='', pct_events_in_branch=branch_fraction(buffer(sess_))))
        for c in ['NoCoating', 'PLL', 'Silica']:
            for r, d in files(sess_, c):
                frac.append(dict(batch=BATCH[sess_], coating=LBL[c], replicate=r, pct_events_in_branch=branch_fraction(d)))
    Mdf, Fdf, Bdf = pd.concat(meds), pd.DataFrame(fits), pd.DataFrame(frac)
    bare = pd.concat([M.scat(v) for v in reps['NoCoating'].values()])
    x0, y0 = np.median(bare[SS]), np.median(bare[F])                                      # single bare origami
    pll = {r: M.scat(v) for r, v in reps['PLL'].items()}
    xb = span_edges([v[SS] for v in pll.values()], 91); yb = span_edges([v[F] for v in pll.values()], 91)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharey=True)
    dens = []
    for ax, (r, d) in zip(axes, pll.items()):
        ok = (d[SS] > 0) & (d[F] > 0)
        h = np.histogram2d(d[SS][ok], d[F][ok], [xb, yb])[0]
        dens += [dict(replicate=r, ss_bin_lo=xb[i], ss_bin_hi=xb[i + 1], v525_bin_lo=yb[j], v525_bin_hi=yb[j + 1], events=h[i, j])
                 for i, j in zip(*np.nonzero(h))]
        ax.pcolormesh(xb, yb, np.where(h > 0, h / h.sum(), np.nan).T, norm=LogNorm(vmin=1e-6, vmax=1e-1), cmap='Blues', rasterized=True)
        ax.axvline(TRIG, color=INK, ls='--', lw=1)
        xs = np.logspace(np.log10(x0), np.log10(xb[-1]), 10)
        for k in (0.5, 1, 1.5):
            ax.plot(xs, y0 * (xs / x0) ** k, color=MUTED, ls=':', lw=1)
            xe = min(xb[-1], x0 * (yb[-1] / y0) ** (1 / k))
            ax.text(xe / 1.6, y0 * (xe / 1.6 / x0) ** k * 1.3, '½' if k == 0.5 else f'{k:g}', fontsize=7, color=MUTED)
        f = Fdf[(Fdf.coating == 'PLL-PEG') & (Fdf.replicate == r) & (Fdf.branch_v525_cut == 200) & (Fdf.fit_ss_lo == 2400)
                & (Fdf.fit_ss_hi == 7e4)].iloc[0]
        bf = Bdf[(Bdf.batch == 'batch 2') & (Bdf.coating == 'PLL-PEG') & (Bdf.replicate == r)].pct_events_in_branch.iloc[0]
        T = Mdf[(Mdf.coating == 'PLL-PEG') & (Mdf.replicate == r) & (Mdf.branch_v525_cut == 200)]
        zz = T[(T.ss_bin_center >= 2400) & (T.ss_bin_center <= 7e4) & T.branch_median_v525.notna()]
        if np.isfinite(f.slope):
            ax.plot(zz.ss_bin_center, zz.branch_median_v525, 'o', color=COL['PLL'], mec='white', ms=4.5, zorder=4)
            a0 = np.log10(zz.branch_median_v525).mean() - f.slope * np.log10(zz.ss_bin_center).mean()
            xf = np.logspace(np.log10(1500), np.log10(8e4), 10)
            ax.plot(xf, 10 ** (a0 + f.slope * np.log10(xf)), color=COL['PLL'], lw=1.5, zorder=3)
            txt = f'branch slope {f.slope:.2f} ± {f.slope_se:.2f}'
        else:
            txt = 'too few branch events to fit'
        ax.text(0.97, 0.05, f'{txt}\n{bf:.2f}% of events in branch', transform=ax.transAxes, ha='right', fontsize=6.5,
                color=COL['PLL'])
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(xb[0], xb[-1]); ax.set_ylim(yb[0], yb[-1])
        ax.set_xlabel('VSSC-1-H'); ax.set_title(f'replicate {r}', fontsize=9)
    axes[0].set_ylabel('V525-H (ATTO 425)')
    fig.suptitle('PLL-PEG-coated origami, batch 2', fontsize=10, y=1.02)
    dens = pd.DataFrame(dens)
    dens = pd.DataFrame([dict(ss_bin_lo=xb[i], ss_bin_hi=xb[i + 1], v525_bin_lo=yb[j], v525_bin_hi=yb[j + 1], events=h[i, j])
                         for i, j in zip(*np.nonzero(h))])
    save(fig, 'fig4c_pllpeg_branch_slope', {'_branch_medians': Mdf, '_slope_fits': Fdf, '_branch_fraction': Bdf,
                                           '_density': dens})


# ------------------------------------------------------------------------------------------------ 4f
def median_curve(d, t, buf, tb, edges, nmin=50):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sv = d[(d[SS] >= a) & (d[SS] < b)][F].values; bv = buf[(buf[SS] >= a) & (buf[SS] < b)][F].values
        out.append(bs_percentile(sv[sv > 0], t, bv[bv > 0], tb, 50) if len(sv) - len(bv) * t / tb >= nmin else np.nan)
    return np.array(out)


K_GRID = np.logspace(np.log10(0.4), np.log10(3), 801)


def shift_rms(xc, m_coat, m_bare, k, free_scale=False, ss_max=np.inf):
    ok = np.isfinite(m_bare); lx, ly = np.log10(xc[ok]), np.log10(m_bare[ok])
    x = np.log10(xc) - np.log10(k)
    use = np.isfinite(m_coat) & (x >= lx.min()) & (x <= lx.max()) & (xc <= ss_max)
    if use.sum() < 4:
        return np.nan, np.nan, int(use.sum())
    res = np.log10(m_coat[use]) - np.interp(x[use], lx, ly); lb = res.mean() if free_scale else 0.0
    return np.sqrt(np.mean((res - lb) ** 2)), 10 ** lb, int(use.sum())


def fit_shift(xc, m_coat, m_bare, free_scale=False, ss_max=np.inf):
    """k (and optionally b) minimizing the mean squared log10 difference between m_coat(S) and b * m_bare(S / k)."""
    ok = np.isfinite(m_bare); lx, ly = np.log10(xc[ok]), np.log10(m_bare[ok])
    best = (np.inf, np.nan, np.nan, 0)
    for k in K_GRID:
        x = np.log10(xc) - np.log10(k)
        use = np.isfinite(m_coat) & (x >= lx.min()) & (x <= lx.max()) & (xc <= ss_max)
        if use.sum() < 4:
            continue
        pred = np.interp(x[use], lx, ly); res = np.log10(m_coat[use]) - pred
        lb = res.mean() if free_scale else 0.0
        rms = np.sqrt(np.mean((res - lb) ** 2))
        if rms < best[0]:
            best = (rms, k, 10 ** lb, int(use.sum()))
    out = dict(k=best[1], fluorescence_scale=best[2], rms_log10=best[0], n_bins=best[3])
    for kk in (1.0, 1.8):
        out[f'rms_log10_at_k{kk:g}'] = shift_rms(xc, m_coat, m_bare, kk, free_scale, ss_max)[0]
    return out


def coverage_from_k(k):
    t = pd.read_csv(os.path.join(M.OUT, 'scattering_vs_pllpeg_coverage.csv'))
    return float(np.interp(k, t.I_ratio_ADDA, t.fraction_phosphates_bound)) if k > 1 else 0.0


def panel_4f():
    sess = B1; s = M.SESSIONS[sess]
    bu = buffer(sess); buf = M.scat(bu); tb = acq_time(bu)
    ctrl = M.scat(M.L('coatings_controls', 'PLL_PEG_NoOrigami.fcs'))
    disp = np.logspace(np.log10(TRIG), 5, 15); xd = np.sqrt(disp[:-1] * disp[1:])          # bins of the current Fig. 4
    fine = np.logspace(np.log10(TRIG), 4, 25); xf = np.sqrt(fine[:-1] * fine[1:])
    fl = {c: files(sess, c) for c in ['NoCoating', 'PLL']}
    pooled = {c: (pd.concat([M.scat(d) for _, d in fl[c]]), sum(acq_time(d) for _, d in fl[c])) for c in fl}
    curves = {c: median_curve(*pooled[c], buf, tb, disp) for c in pooled}
    raw = lambda z, nmin: np.array([z[(z[SS] >= a) & (z[SS] < b)][F].median() if ((z[SS] >= a) & (z[SS] < b)).sum() >= nmin
                                    else np.nan for a, b in zip(disp[:-1], disp[1:])])
    curves['control'], curves['Buffer'] = raw(ctrl, 50), raw(buf, 20)
    fits, prof = [], []
    for r in ['pooled'] + list('ABC'):
        if r == 'pooled':
            mb, mc = median_curve(*pooled['NoCoating'], buf, tb, fine), median_curve(*pooled['PLL'], buf, tb, fine)
        else:
            db = dict(fl['NoCoating'])[r]; dc = dict(fl['PLL'])[r]
            mb = median_curve(M.scat(db), acq_time(db), buf, tb, fine); mc = median_curve(M.scat(dc), acq_time(dc), buf, tb, fine)
        for ss_max, rng in [(np.inf, 'VSSC-1-H 200-10,000'), (1000, 'VSSC-1-H 200-1,000 (single-origami region)')]:
            for free in (False, True):
                f = fit_shift(xf, mc, mb, free, ss_max)
                fits.append(dict(batch='batch 1', replicate=r, fit_range=rng,
                                 model='k and fluorescence scale b' if free else 'k only', **f,
                                 phosphates_bound_fraction_for_k=coverage_from_k(f['k'])))
            prof += [dict(replicate=r, fit_range=rng, k=k, rms_log10_k_only=shift_rms(xf, mc, mb, k, False, ss_max)[0])
                     for k in K_GRID[::8]]
    Fdf = pd.DataFrame(fits)
    ko = Fdf[Fdf.model == 'k only']; kmin, kmax = ko.k.min(), ko.k.max()
    fmax = coverage_from_k(kmax) * 100
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.plot(xd, curves['NoCoating'], 'o-', color=COL['NoCoating'], ms=4, label='bare origami')
    ax.plot(xd, curves['PLL'], 's-', color=COL['PLL'], ms=4, label='PLL-PEG-coated')
    ax.plot(xd, curves['control'], '^--', color=COL['control'], ms=4, lw=1, label='PLL-PEG alone')
    ax.plot(xd, curves['Buffer'], 'v-', color=COL['Buffer'], ms=4, lw=1, label='buffer')
    ax.plot(xd * 1.8, curves['NoCoating'], '--', color=COL['NoCoating'], lw=1.2, label='bare shifted ×1.8')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(30, 400); ax.set_xlim(150, 1e5)
    ax.set_yticks([30, 50, 100, 200, 400]); ax.set_yticklabels(['30', '50', '100', '200', '400'])
    ax.set_xlabel('VSSC-1-H bin'); ax.set_ylabel('median V525-H (ATTO 425)'); ax.set_title('PLL-PEG, batch 1', fontsize=9)
    ax.text(0.98, 0.03, f'best-fit shift k = {kmin:.2f}–{kmax:.2f} (≤ {fmax:.1f}% of phosphates bound)\n'
            'if the ATTO 425 signal is unchanged; ×1.8 = 10% bound',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6, color=INK)
    ax.legend(loc='upper left', fontsize=6.5)
    C = pd.DataFrame([dict(batch='batch 1', sample=n, ss_bin_center=x, median_v525=v)
                      for n, key in [('bare origami (buffer-subtracted)', 'NoCoating'), ('PLL-PEG-coated (buffer-subtracted)', 'PLL'),
                                     ('PLL-PEG alone (raw)', 'control'), ('buffer (raw)', 'Buffer')]
                      for x, v in zip(xd, curves[key])] +
                     [dict(batch='batch 1', sample='bare origami shifted x1.8', ss_bin_center=x * 1.8, median_v525=v)
                      for x, v in zip(xd, curves['NoCoating'])])
    save(fig, 'fig4f_pllpeg_batch1_shifted_bare', {'_curves': C, '_shift_fits': Fdf, '_rms_profile': pd.DataFrame(prof)})


# ------------------------------------------------------------------------------------------------ 5b
def panel_5b():
    q = pd.read_csv(os.path.join(M.OUT, 'fig26_low_scatter_probe_brightness_samples.csv'))
    q = q[(q.session == B2) & q.coating.isin(['Uncoated', 'PLL-PEG'])].copy()
    q['biotin'] = np.where(q.extension.astype(str) == 'N', 'no biotin', 'biotin')
    lr = stats.linregress(q.low_SS_median_B525, q.gated_median_B525)
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    for (c, b), z in q.groupby(['coating', 'biotin']):
        colr = COL['NoCoating'] if c == 'Uncoated' else COL['PLL']
        if b == 'no biotin':
            ax.plot(z.low_SS_median_B525, z.gated_median_B525, 'o', mfc='white', mec=colr, mew=1.4, ms=6,
                    label=f"no biotin, {'bare' if c == 'Uncoated' else 'PLL-PEG'}", zorder=4)
        else:
            ax.plot(z.low_SS_median_B525, z.gated_median_B525, 'o', color='#8C8C8C', ms=5, zorder=3,
                    label='biotin, both coatings' if c == 'Uncoated' else None)
    xs = np.linspace(q.low_SS_median_B525.min() - 1, q.low_SS_median_B525.max() + 1, 10)
    ax.plot(xs, lr.intercept + lr.slope * xs, color=INK, ls='--', lw=1)
    ax.text(0.97, 0.06, f'one line, r = {lr.rvalue:.2f}\nslope {lr.slope:.2f}, n = {len(q)}', transform=ax.transAxes, ha='right',
            fontsize=7)
    ax.set_xlabel('B525-H of low-scatter events (free antibody)'); ax.set_ylabel('B525-H of origami events')
    ax.legend(loc='upper left', fontsize=7)
    q['fit_prediction'] = lr.intercept + lr.slope * q.low_SS_median_B525
    S = pd.DataFrame([dict(fit='all samples', n=len(q), slope=lr.slope, slope_se=lr.stderr, intercept=lr.intercept, r=lr.rvalue,
                           p=lr.pvalue)] +
                     [dict(fit=f'{g} only', n=len(z), **dict(zip(['slope', 'slope_se', 'intercept', 'r', 'p'],
                                                                   [(l := stats.linregress(z.low_SS_median_B525, z.gated_median_B525)).slope,
                                                                    l.stderr, l.intercept, l.rvalue, l.pvalue])))
                      for g, z in q.groupby('biotin')])
    save(fig, 'fig5b_origami_vs_free_antibody_signal',
         {'_samples': q[['session', 'coating', 'replicate', 'extension', 'biotin', 'low_SS_median_B525', 'gated_median_B525',
                         'fit_prediction']], '_fit': S})


# ------------------------------------------------------------------------------------------------ 5f
def panel_5f():
    t = pd.read_csv(os.path.join(M.OUT, 'tccd_acquisition_order.csv'))
    t['start'] = pd.to_datetime(t.start); t0 = t.start.min()
    t['hours_since_first_acquisition'] = (t.start - t0).dt.total_seconds() / 3600
    t['set'] = t.coating + ' ' + t.replicate
    a = t[t.coating.isin(['Uncoated', 'PLL-PEG']) & t.valid].copy()
    stat_rows = []
    fmt = lambda p: 'p < 0.001' if p < 0.001 else f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}'
    for y in ['observed_over_chance', 'R_B']:
        a['ly'] = np.log(a[y])
        bio = a[a.extension.astype(str) != 'N']
        m = smf.ols('ly ~ minutes_since_first_sample_of_set + C(set)', data=bio).fit()
        b = m.params['minutes_since_first_sample_of_set']; pv = m.pvalues['minutes_since_first_sample_of_set']
        stat_rows.append(dict(quantity=y, model="biotin samples within a set: log y ~ minutes since the set's first sample + set",
                              samples='bare and PLL-PEG, 3/6/12 nt, valid', n=int(m.nobs),
                              pct_change_per_10_min=100 * (np.exp(10 * b) - 1), p=pv,
                              note='spacer length and acquisition time are confounded (order 3, 6, 12 nt in every set)'))
        n = a[a.extension.astype(str) == 'N']
        lr = stats.linregress(n.minutes_in_session, np.log(n[y]))
        stat_rows.append(dict(quantity=y, model='no-biotin samples across the session: log y ~ minutes in session',
                              samples='bare and PLL-PEG, no biotin', n=len(n), pct_change_per_10_min=100 * (np.exp(10 * lr.slope) - 1),
                              p=lr.pvalue, note='drift between sets, free of the biotin and spacer effects'))
    S = pd.DataFrame(stat_rows)
    mk = {'N': 'o', '3': 's', '6': '^', '12': 'D'}
    cmap = {'Uncoated': COL['NoCoating'], 'PLL-PEG': COL['PLL'], 'Silica': COL['Silica']}
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.8), sharex=True)
    for ax, y, lab in [(axes[0], 'observed_over_chance', 'observed ÷ chance (ρ)'), (axes[1], 'R_B', 'antibody events per second')]:
        for st, z in t.groupby('set'):
            z = z.sort_values('order_in_set'); colr = cmap[z.coating.iloc[0]]; alpha = 0.45 if z.coating.iloc[0] == 'Silica' else 1
            zv = z[z.valid] if y == 'observed_over_chance' else z
            ax.plot(zv.hours_since_first_acquisition, zv[y], '-', color=colr, lw=1, alpha=alpha)
            for e, zz in zv.groupby(zv.extension.astype(str)):
                ax.plot(zz.hours_since_first_acquisition, zz[y], ls='', marker=mk[e], color=colr, mec='white', mew=0.5, ms=5,
                        alpha=alpha)
        ax.set_ylabel(lab)
    axes[1].set_xlabel(f'hours since the first acquisition ({t0:%H:%M}, 04/18)')
    h = [plt.Line2D([], [], color=cmap[c], lw=1.5, label={'Uncoated': 'bare', 'PLL-PEG': 'PLL-PEG', 'Silica': 'silica (not analysed)'}[c])
         for c in cmap] + [plt.Line2D([], [], ls='', marker=mk[e], color=MUTED, label='no biotin' if e == 'N' else f'{e} nt') for e in mk]
    axes[0].legend(handles=h, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.32), fontsize=7)
    for ax, y in [(axes[0], 'observed_over_chance'), (axes[1], 'R_B')]:
        w = S[(S.quantity == y) & S.model.str.startswith('biotin')].iloc[0]; d0 = S[(S.quantity == y) & S.model.str.startswith('no-biotin')].iloc[0]
        ax.text(0.99, 0.04 if y == 'observed_over_chance' else 0.80,
                f"biotin samples within a set: {w.pct_change_per_10_min:+.1f}% per 10 min ({fmt(w.p)})\n"
                f"no-biotin samples across the session: {d0.pct_change_per_10_min:+.1f}% per 10 min ({fmt(d0.p)})",
                transform=ax.transAxes, ha='right', fontsize=6.5)
    cols = ['sample', 'coating', 'replicate', 'extension', 'start', 'hours_since_first_acquisition', 'minutes_in_session',
            'order_in_set', 'minutes_since_first_sample_of_set', 'observed_over_chance', 'R_B', 'R_A', 'valid']
    out = t[cols].rename(columns={'R_B': 'antibody_events_per_s', 'R_A': 'origami_events_per_s'})
    save(fig, 'fig5f_run_order_check', {'_acquisitions': out, '_trend_statistics': S})


PANELS = {'2b': panel_2b, '2c': panel_2c, '2d': panel_2d, '4c': panel_4c, '4f': panel_4f, '5b': panel_5b, '5f': panel_5f}

if __name__ == '__main__':
    for p in (sys.argv[1:] or PANELS):
        print('panel', p); PANELS[p]()
