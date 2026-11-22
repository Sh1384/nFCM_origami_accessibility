"""Checks behind four questions on the 10/25 draft (Figure 3H caption, excluded acquisitions, statistics, scatter tail).

Q1  Figure 3H. Scatter-triggered events per second (VSSC-1-H >= 200) of every no-probe tube of bare, PLL-PEG- and
    silica-coated origami, raw and minus the session buffer, next to the coating-only controls acquired 02/13
    (coatings_controls/PLL_PEG_NoOrigami.fcs, coatings_controls/SilicaChemicalsOnly.fcs; no buffer that day).
Q2  Every .fcs file in the data folders: all recorded events / acquisition time (the quantity tested by
    shared_utils.load_fcs), flagged if above 7,000 events/s. Coating and probe are parsed from the file name.
Q3  a  Silica vs bare median side scatter of origami-confident events, batch 1 (silica_confident_origami_summary.py,
       background P95): Welch t-test on the three replicate medians per coating.
    b  No-biotin observed-to-chance ratio rho0, bare vs PLL-PEG (antibody session): recomputed here with the same gate,
       buffer subtraction and coincidence window as fig29_tccd_coincidence.py, and without buffer subtraction;
       Welch, Student and paired t-tests on the three samples per coating.
Q4  Bare origami without probe, recorded side scatter (as in Figures 2C and S4B), from the raw events, per tube and pooled:
       slope at 200    -d ln N(>=s) / d ln s of the cumulative count between 200 and 200 x 10^0.05, raw and minus the
                       session buffer (scaled by acquisition time); this sets how the scatter-triggered event rate
                       responds to a small change in scatter
       MLE 200-300     exponent of the density per log unit (N(>=s) ~ s^-alpha for a pure power law): unbinned maximum
                       likelihood for a power law truncated to [200, 300], and a binned Poisson likelihood (100 log bins)
                       with the buffer as a fixed background; also for [200, 95th percentile] and [200, 1,000]
       line fit        least-squares line through ln N(>=s) vs ln s from 200 to the 95th percentile (the earlier fig2c fit)
    Local slope of the cumulative curve in steps of 0.05 decade, where >= 200 events lie above the step (minus buffer:
    only where the scaled buffer is less than half of the events above the step).
Batch 1 = streptavidin session (05/25), batch 2 = antibody session (04/18).

Out:  figures_svg/checks_11_08/fig41{a,b,c,d}_*.svg and *.csv
Run:  python fig41_caption_stat_checks.py [q1 q2 q3 q4]      (default: all)
"""
import sys, os, re, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from fcsparser import parse
import fig37_proposed_flow_panels as P
import make_manuscript_figures as M
import shared_utils as U
from shared_utils import acq_time, compute_monomer_gates, apply_gate

OUTD = os.path.join(M.OUT, 'checks_11_08'); os.makedirs(OUTD, exist_ok=True)
SS, B = 'VSSC-1-H', 'B525-H'
TRIG, AB_T = 200, 250
B1, B2 = P.B1, P.B2
BLAB = {B1: 'batch 1 (05/25)', B2: 'batch 2 (04/18)'}
COL, INK, MUTED = P.COL, P.INK, P.MUTED
LBL = {'NoCoating': 'bare', 'PLL': 'PLL-PEG', 'Silica': 'silica'}
CTRL = [('PLL_PEG_NoOrigami.fcs', 'PLL-PEG only', 'PLL'), ('SilicaChemicalsOnly.fcs', 'silica precursors only', 'Silica')]
rng = np.random.default_rng(3)
STEP = 0.05                                          # log10 step for slopes of the cumulative curve


def save(fig, name, tables):
    fig.savefig(os.path.join(OUTD, f'{name}.svg'), bbox_inches='tight')
    if P.PREVIEW:
        fig.savefig(os.path.join(P.PREVIEW, f'{name}.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    for suffix, df in tables.items():
        df.to_csv(os.path.join(OUTD, f'{name}{suffix}.csv'), index=False)
    print(f'  wrote {name}.svg (+{len(tables)} csv)')


def start_time(folder, fname):
    try:
        return M.acq_record(folder, fname)['StartTime']
    except Exception:
        return ''


# ---------------------------------------------------------------- Q1
def q1():
    rows = []
    for sess in [B1, B2]:
        s = M.SESSIONS[sess]; buf = P.buffer(sess); bt = acq_time(buf); b_ss = (buf[SS] >= TRIG).sum() / bt
        rows.append(dict(batch=BLAB[sess], sample='buffer', coating='Buffer', replicate='', file=f"{s['folder']}/Buffer.fcs",
                         start=start_time(s['folder'], 'Buffer.fcs'), acq_time_s=bt, all_events_per_s=len(buf) / bt,
                         scatter_events_per_s=b_ss, scatter_events_per_s_minus_buffer=np.nan))
        for c in ['NoCoating', 'PLL', 'Silica']:
            for r in 'ABC':
                f = f"N_{c}_{s['noprobe']}_{r}.fcs"; d = M.L(s['folder'], f)
                if d is None:
                    continue
                t = acq_time(d); v = (d[SS] >= TRIG).sum() / t
                rows.append(dict(batch=BLAB[sess], sample=LBL[c], coating=c, replicate=r, file=f"{s['folder']}/{f}",
                                 start=start_time(s['folder'], f), acq_time_s=t, all_events_per_s=len(d) / t,
                                 scatter_events_per_s=v, scatter_events_per_s_minus_buffer=v - b_ss))
    for f, lab, c in CTRL:
        d = M.L('coatings_controls', f); t = acq_time(d)
        rows.append(dict(batch='coating-only controls (02/13)', sample=lab, coating=c, replicate='', file=f'coatings_controls/{f}',
                         start=start_time('coatings_controls', f), acq_time_s=t, all_events_per_s=len(d) / t,
                         scatter_events_per_s=(d[SS] >= TRIG).sum() / t, scatter_events_per_s_minus_buffer=np.nan))
    S = pd.DataFrame(rows)
    T = (S[S.replicate != ''].groupby(['batch', 'sample'], sort=False)
         .agg(n=('scatter_events_per_s', 'size'), scatter_events_per_s_mean=('scatter_events_per_s', 'mean'),
              scatter_events_per_s_sd=('scatter_events_per_s', 'std'),
              minus_buffer_mean=('scatter_events_per_s_minus_buffer', 'mean'),
              minus_buffer_sd=('scatter_events_per_s_minus_buffer', 'std')).reset_index())

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    groups = [(BLAB[B1], c) for c in ['NoCoating', 'PLL', 'Silica']] + [(BLAB[B2], c) for c in ['NoCoating', 'PLL', 'Silica']]
    xt, xl = [], []
    for i, (b, c) in enumerate(groups):
        x = i + (0 if b == BLAB[B1] else 0.6)
        z = S[(S.batch == b) & (S.coating == c)].scatter_events_per_s_minus_buffer.values     # as plotted in Figure 3H
        ax.bar(x, z.mean(), 0.6, color=COL[c], alpha=0.25, edgecolor=COL[c], lw=1)
        ax.errorbar(x, z.mean(), z.std(ddof=1), color=INK, capsize=3, lw=1)
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(z)), z, s=18, color=COL[c], edgecolor='white', lw=0.5, zorder=3)
        xt.append(x); xl.append(f'{LBL[c]}\n{b.split(" (")[0]}')
    for b, x0 in [(BLAB[B1], -0.4), (BLAB[B2], 3.2)]:
        v = S[(S.batch == b) & (S.coating == 'Buffer')].scatter_events_per_s.iloc[0]
        ax.hlines(v, x0, x0 + 2.8, color=COL['Buffer'], ls='--', lw=1)
        ax.text(x0 + 2.8, v * 1.08, f'buffer subtracted, {v:.0f}', color=MUTED, fontsize=7, ha='right')
    for k, (f, lab, c) in enumerate(CTRL):
        x = 7.0 + k * 0.9; v = S[S['sample'] == lab].scatter_events_per_s.iloc[0]
        ax.bar(x, v, 0.6, color='white', edgecolor=COL[c], lw=1.2, hatch='////')
        ax.text(x, v * 1.1, f'{v:.0f}', ha='center', fontsize=7, color=INK)
        xt.append(x); xl.append(lab.replace(' only', '\nonly').replace('precursors', 'prec.'))
    ax.set_yscale('log'); ax.set_ylim(20, 2e4)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylabel('scatter-triggered events/s\n(VSSC-1-H ≥ 200)')
    ax.set_title('Figure 3H data: tubes minus session buffer (bars mean ± SD); 02/13 controls hatched, raw (no buffer that day)',
                 fontsize=8, loc='left')
    save(fig, 'fig41a_fig3h_rates_and_coating_controls', {'_tubes': S, '_conditions': T})
    return S, T


# ---------------------------------------------------------------- Q2
def q2():
    rows = []
    for folder in sorted(os.listdir(U.PROJECT_ROOT)):
        for path in sorted(glob.glob(os.path.join(U.PROJECT_ROOT, folder, '*.fcs'))):
            f = os.path.basename(path)
            _, d = parse(path, reformat_meta=True)
            t = acq_time(d); eps = len(d) / t
            m = re.search(r'_(NoCoating|PLL|Silica|Corona|Chitosan)_', f)
            coat = m.group(1) if m else ('Buffer' if f.startswith('Buffer') else 'control')
            in_ms = (folder in ('Shervin4_18', 'Shervin5_25') and (coat in ('NoCoating', 'PLL', 'Silica', 'Buffer'))) or \
                    (folder, f) in {('probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs'), ('probe_only_controls', 'Buffer__Shervin03_13_2026.fcs'),
                                    ('probe_only_controls', 'Streptavidin_Only__Shervin5_03.fcs'), ('probe_only_controls', 'Buffer__Shervin5_03.fcs'),
                                    ('coatings_controls', 'PLL_PEG_NoOrigami.fcs'), ('coatings_controls', 'SilicaChemicalsOnly.fcs')}
            rows.append(dict(folder=folder, file=f, coating=coat, manuscript_file=in_ms, events=len(d), acq_time_s=t,
                             all_events_per_s=eps, scatter_events_per_s=(d[SS] >= TRIG).sum() / t,
                             excluded_over_7000=eps > U.MAX_EVENTS_PER_SEC))
    A = pd.DataFrame(rows)
    C = (A.groupby(['manuscript_file', 'folder', 'coating']).agg(files=('file', 'size'), excluded=('excluded_over_7000', 'sum'),
                                                                 max_events_per_s=('all_events_per_s', 'max')).reset_index())

    ms = A[A.manuscript_file & A.folder.isin(['Shervin5_25', 'Shervin4_18'])]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, (folder, lab) in zip(axes, [('Shervin5_25', BLAB[B1]), ('Shervin4_18', BLAB[B2])]):
        z = ms[ms.folder == folder]
        for i, c in enumerate(['Buffer', 'NoCoating', 'PLL', 'Silica']):
            q = z[z.coating == c]; x = i + rng.uniform(-0.15, 0.15, len(q))
            ex = q.excluded_over_7000.values
            ax.scatter(x[~ex], q.all_events_per_s[~ex], s=16, color=COL[c], edgecolor='white', lw=0.5, zorder=3)
            ax.scatter(x[ex], q.all_events_per_s[ex], s=34, marker='X', color=COL[c], edgecolor=INK, lw=0.6, zorder=4)
            ax.text(i, 3.2e4, f'{ex.sum()}/{len(q)}', ha='center', fontsize=7, color=INK if ex.sum() else MUTED)
        ex = z[z.excluded_over_7000].sort_values('all_events_per_s', ascending=False)
        ax.text(0.36, 0.03, 'excluded:\n' + '\n'.join(f"{r.file.replace('.fcs', '')}  {r.all_events_per_s:,.0f}/s" for r in ex.itertuples()),
                transform=ax.transAxes, fontsize=6.3, va='bottom', color=INK)
        ax.axhline(U.MAX_EVENTS_PER_SEC, color='#C0392B', ls='--', lw=1)
        ax.text(1.5, U.MAX_EVENTS_PER_SEC * 0.8, '7,000 events/s', color='#C0392B', fontsize=7, ha='center', va='top')
        ax.set_yscale('log'); ax.set_ylim(20, 5e4); ax.set_xlim(-0.5, 3.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(['buffer', 'bare', 'PLL-PEG', 'silica'])
        ax.set_title(f'{lab}: excluded / files', fontsize=8, loc='left')
    axes[0].set_ylabel('all recorded events/s')
    save(fig, 'fig41b_excluded_acquisitions', {'_all_files': A, '_counts': C})
    return A, C


# ---------------------------------------------------------------- Q3
def rho0_recomputed():
    """No-biotin rho0 with the gate, buffer subtraction and 2 tau of fig29_tccd_coincidence.py, and without subtraction."""
    q = pd.read_csv(os.path.join(M.OUT, 'fig29_tccd_coincidence_samples.csv'))
    q = q[q.probe.str.contains('IgG') & (q.extension.astype(str) == 'N')]
    w = q.two_tau_deficit_us.iloc[0] * 1e-6
    s = M.SESSIONS[B2]; dfs = M.session_dfs(s, exts=tuple(U.EXTENSIONS))
    gates = compute_monomer_gates(dfs, ctl=s['noprobe'])
    buf = dfs['Buffer']; tb = acq_time(buf)
    rows = []
    for c in ['NoCoating', 'PLL', 'Silica']:
        gb = apply_gate(buf, c, 'A', gates)
        for r in 'ABC':
            d = dfs.get(('N', c, s['probe'], r))
            if d is None:
                continue
            t = acq_time(d); g = apply_gate(d, c, r, gates)
            raw = (len(g), (d[B] >= AB_T).sum(), (g[B] >= AB_T).sum())
            bg = (len(gb) * t / tb, (buf[B] >= AB_T).sum() * t / tb, (gb[B] >= AB_T).sum() * t / tb)
            for lab, (a, b, cc) in [('buffer-subtracted (as in paper)', tuple(x - y for x, y in zip(raw, bg))), ('no buffer subtraction', raw)]:
                rows.append(dict(coating=M.COAT_LBL[c], replicate=r, sample=f"N_{c}_{s['probe']}_{r}", variant=lab,
                                 R_O=a / t, R_Ab=b / t, R_c=cc / t, two_tau_us=w * 1e6, rho0=cc / ((a / t) * (b / t) * w * t)))
    R = pd.DataFrame(rows)
    chk = R[R.variant.str.startswith('buffer')].merge(q[['sample', 'obs_exp_deficit']], on='sample')
    assert np.allclose(chk.rho0, chk.obs_exp_deficit, rtol=1e-6), 'recomputed rho0 differs from fig29 output'
    return R


def q3():
    # a  silica vs bare origami-confident median side scatter
    rep = pd.read_csv(os.path.join(M.OUT, 'silica_confident_origami_replicates.csv'))
    rep = rep[rep.threshold == 'P95']
    tests = []
    for day, z in rep.groupby('day', sort=False):
        si = z[z.coating == 'Silica'].median_SS_origami_confident.values; ba = z[z.coating == 'Uncoated'].median_SS_origami_confident.values
        tests.append(dict(quantity='median VSSC-1-H of origami-confident events, silica vs bare', group=day, n=f'{len(si)} vs {len(ba)}',
                          mean_1=si.mean(), sd_1=si.std(ddof=1), mean_2=ba.mean(), sd_2=ba.std(ddof=1),
                          welch_p=stats.ttest_ind(si, ba, equal_var=False).pvalue, student_p=stats.ttest_ind(si, ba).pvalue,
                          paired_p=stats.ttest_rel(si, ba).pvalue, mann_whitney_p=stats.mannwhitneyu(si, ba).pvalue))
    # b  rho0
    R = rho0_recomputed()
    for v, z in R[R.coating != 'Silica'].groupby('variant', sort=False):
        a = z[z.coating == 'Uncoated'].sort_values('replicate').rho0.values; b = z[z.coating == 'PLL-PEG'].sort_values('replicate').rho0.values
        both = np.r_[a, b]
        tests.append(dict(quantity=f'no-biotin rho0, bare vs PLL-PEG, {v}', group=BLAB[B2], n=f'{len(a)} vs {len(b)}',
                          mean_1=a.mean(), sd_1=a.std(ddof=1), mean_2=b.mean(), sd_2=b.std(ddof=1),
                          welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue, student_p=stats.ttest_ind(a, b).pvalue,
                          paired_p=stats.ttest_rel(a, b).pvalue, mann_whitney_p=stats.mannwhitneyu(a, b).pvalue,
                          cv_pct_6_samples=100 * both.std(ddof=1) / both.mean()))
    Tst = pd.DataFrame(tests)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    for i, (day, z) in enumerate(rep.groupby('day', sort=False)):
        for k, c in enumerate(['Uncoated', 'Silica']):
            v = z[z.coating == c].median_SS_origami_confident.values; x = i + (k - 0.5) * 0.36
            colr = COL['NoCoating' if c == 'Uncoated' else 'Silica']
            ax.bar(x, v.mean(), 0.32, color=colr, alpha=0.25, edgecolor=colr)
            ax.errorbar(x, v.mean(), v.std(ddof=1), color=INK, capsize=3, lw=1)
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=16, color=colr, edgecolor='white', lw=0.5, zorder=3)
        p = Tst[(Tst.group == day) & Tst.quantity.str.startswith('median')].welch_p.iloc[0]
        ax.text(i, 700, f'Welch p = {p:.2f}', ha='center', fontsize=7, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['batch 1 (05/25)', 'batch 2 (04/18)']); ax.set_ylim(0, 760)
    ax.set_ylabel('median VSSC-1-H,\norigami-confident events')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[c], alpha=0.5, label=l) for c, l in [('NoCoating', 'bare'), ('Silica', 'silica')]],
              fontsize=7, loc='upper right', bbox_to_anchor=(1, 0.9))
    ax.set_title('Q3a: p = 0.07 (n = 3 tubes per coating)', fontsize=8, loc='left')
    ax = axes[1]
    for k, (v, mk, fc) in enumerate([('buffer-subtracted (as in paper)', 'o', None), ('no buffer subtraction', 'o', 'white')]):
        z = R[(R.variant == v) & (R.coating != 'Silica')]
        for i, c in enumerate(['Uncoated', 'PLL-PEG']):
            q = z[z.coating == c].rho0.values; x = i + (k - 0.5) * 0.36
            colr = COL['NoCoating' if c == 'Uncoated' else 'PLL']
            ax.errorbar(x, q.mean(), q.std(ddof=1), fmt='_', ms=10, color=INK, capsize=3, lw=1)
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(q)), q, s=18, marker=mk, color=fc or colr, edgecolor=colr, lw=1, zorder=3)
    for k, v in enumerate(['buffer-subtracted (as in paper)', 'no buffer subtraction']):
        r = Tst[Tst.quantity.str.endswith(v)].iloc[0]
        ax.text(0.03, 0.96 - 0.09 * k, f"{'filled' if k == 0 else 'open'}: {r.mean_1:.1f} ± {r.sd_1:.1f} vs {r.mean_2:.1f} ± {r.sd_2:.1f}, Welch p = {r.welch_p:.2f}",
                transform=ax.transAxes, fontsize=6.5, va='top', color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['bare', 'PLL-PEG']); ax.set_ylabel('no-biotin observed / chance (ρ⁰)')
    ax.set_ylim(9.4, 12.2); ax.set_title('Q3b: ρ⁰ by coating (n = 3 each)', fontsize=8, loc='left')
    save(fig, 'fig41c_statistics', {'_tests': Tst, '_rho0_samples': R, '_silica_confident_replicates': rep})
    return Tst, R


# ---------------------------------------------------------------- Q4
def mle_raw(v, lo, hi):
    """Unbinned MLE of alpha for N(>s) ~ s^-alpha truncated to [lo, hi]; SE from the observed information."""
    x = v[(v >= lo) & (v <= hi)]; n = len(x); sl = np.log(x).sum()
    if n < 50:
        return np.nan, np.nan, n
    nll = lambda a: -(n * np.log(a) - (a + 1) * sl - n * np.log(lo ** -a - hi ** -a))
    a = optimize.minimize_scalar(nll, bounds=(0.01, 20), method='bounded').x
    h = 1e-4 * a; curv = (nll(a + h) - 2 * nll(a) + nll(a - h)) / h ** 2
    return a, (1 / np.sqrt(curv) if curv > 0 else np.nan), n


def mle_background(v, bgv, scale, lo, hi, nbins=100):
    """Binned Poisson MLE: counts = A (lo^-a - ...) differences + fixed buffer background (buffer counts x scale)."""
    e = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    n = np.histogram(v, e)[0].astype(float); b = np.histogram(bgv, e)[0] * scale
    if n.sum() - b.sum() < 50:
        return np.nan, np.nan

    def nll(p):
        a, logA = p
        mu = np.exp(logA) * (e[:-1] ** -a - e[1:] ** -a) / (lo ** -a - hi ** -a) + b
        return -(n * np.log(np.maximum(mu, 1e-300)) - mu).sum()
    res = optimize.minimize(nll, [3.0, np.log(max(n.sum() - b.sum(), 1))], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
    a = res.x[0]; h = 1e-4 * a
    f = lambda aa: optimize.minimize_scalar(lambda la: nll([aa, la]), bracket=(res.x[1] - 1, res.x[1] + 1)).fun   # profile likelihood
    curv = (f(a + h) - 2 * f(a) + f(a - h)) / h ** 2
    return a, (1 / np.sqrt(curv) if curv > 0 else np.nan)


def ccdf_counts(v, grid):
    v = np.sort(v)
    return len(v) - np.searchsorted(v, grid, side='left')


def threshold_values(v, t, bv, bt, p95):
    """Exponents at the threshold from one set of raw events (v, acquisition time t; buffer bv, bt)."""
    s2 = TRIG * 10 ** STEP
    n1, n2 = ccdf_counts(v, [TRIG, s2]); b1, b2 = ccdf_counts(bv, [TRIG, s2]) * t / bt
    g = np.logspace(np.log10(TRIG), np.log10(p95), 60)
    chord = -np.polyfit(np.log(g), np.log(ccdf_counts(v, g)), 1)[0]
    a, se, n = mle_raw(v, TRIG, 300.0); ab, seb = mle_background(v, bv, t / bt, TRIG, 300.0)
    return dict(n_events_ge_200=int(n1), buffer_share_pct=100 * b1 / n1,
                cumulative_slope_at_200_raw=np.log(n1 / n2) / np.log(s2 / TRIG),
                cumulative_slope_at_200_minus_buffer=np.log((n1 - b1) / (n2 - b2)) / np.log(s2 / TRIG),
                density_exponent_200_300_raw=a, density_exponent_200_300_raw_se=se,
                density_exponent_200_300_buffer_background=ab, density_exponent_200_300_buffer_background_se=seb,
                straight_line_fit_200_to_p95_raw=chord, p95=p95)


def q4():
    fits, thr, local, curves = [], [], [], []
    for sess in [B1, B2]:
        buf = P.buffer(sess); bt = acq_time(buf); bv = buf[SS].values
        reps = P.files(sess, 'NoCoating')
        pooled = np.concatenate([d[SS].values for _, d in reps]); pt = sum(acq_time(d) for _, d in reps)
        p95 = float(np.percentile(pooled[pooled >= TRIG], 95))
        for rep, v, t in [(r, d[SS].values, acq_time(d)) for r, d in reps] + [('pooled', pooled, pt)]:
            v = v[v >= TRIG]
            thr.append(dict(batch=BLAB[sess], replicate=rep,
                            **threshold_values(v, t, bv, bt, p95 if rep == 'pooled' else float(np.percentile(v, 95)))))
            for smax_lab, smax in [('300', 300.0), ('95th percentile', p95), ('1000', 1000.0)]:
                a, se, n = mle_raw(v, TRIG, smax); ab, seb = mle_background(v, bv, t / bt, TRIG, smax)
                fits.append(dict(batch=BLAB[sess], replicate=rep, s_min=TRIG, s_max_label=smax_lab, s_max=smax, n_events=n,
                                 density_exponent_raw=a, density_exponent_raw_se=se,
                                 density_exponent_buffer_background=ab, density_exponent_buffer_background_se=seb))
        grid = TRIG * 10 ** np.arange(0, np.log10(pooled.max() / TRIG), STEP)
        n_s = ccdf_counts(pooled, grid); n_b = ccdf_counts(bv, grid) * pt / bt
        curves += [dict(batch=BLAB[sess], s=g, events_per_s_ge_s_bare=a / pt, events_per_s_ge_s_buffer=b / pt)
                   for g, a, b in zip(grid, n_s, n_b)]
        for i in range(1, len(grid) - 1):
            lo, hi = grid[i - 1], grid[i + 1]
            raw = np.log(n_s[i - 1] / n_s[i + 1]) / np.log(hi / lo) if n_s[i + 1] >= 200 else np.nan
            net0, net1 = n_s[i - 1] - n_b[i - 1], n_s[i + 1] - n_b[i + 1]
            net = np.log(net0 / net1) / np.log(hi / lo) if n_b[i + 1] < 0.5 * n_s[i + 1] and net0 > net1 else np.nan   # buffer < half
            local.append(dict(batch=BLAB[sess], s=grid[i], cumulative_slope_raw=raw, cumulative_slope_minus_buffer=net,
                              events_ge_s=int(n_s[i]), buffer_events_ge_s_scaled=n_b[i]))
    Fits, Thr, Loc, Cur = pd.DataFrame(fits), pd.DataFrame(thr), pd.DataFrame(local), pd.DataFrame(curves)
    cols = ['cumulative_slope_at_200_raw', 'cumulative_slope_at_200_minus_buffer', 'density_exponent_200_300_raw',
            'density_exponent_200_300_buffer_background', 'straight_line_fit_200_to_p95_raw']
    Rep = Thr[Thr.replicate != 'pooled'].groupby('batch', sort=False)[cols].agg(['mean', 'std']).round(3)
    Rep.columns = ['_'.join(c) for c in Rep.columns]; Rep = Rep.reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
    ax = axes[0]
    for sess, ls in [(B1, '-'), (B2, '--')]:
        z = Cur[Cur.batch == BLAB[sess]]
        ax.plot(z.s, z.events_per_s_ge_s_bare, color=COL['NoCoating'], ls=ls, lw=1.4, label=f'bare, {BLAB[sess]}')
        ax.plot(z.s, z.events_per_s_ge_s_buffer, color=COL['Buffer'], ls=ls, lw=1.2, label=f'buffer, {BLAB[sess]}')
    f = Thr[(Thr.batch == BLAB[B1]) & (Thr.replicate == 'pooled')].iloc[0]; z = Cur[Cur.batch == BLAB[B1]]
    xs = np.array([TRIG, f.p95]); y0 = z.events_per_s_ge_s_bare.iloc[0]
    ax.plot(xs, y0 * (xs / TRIG) ** -f.straight_line_fit_200_to_p95_raw, color='#C0392B', lw=1, ls=':',
            label='line fit 200 to 95th pct (batch 1)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(TRIG, 3e4); ax.set_ylim(1e-2, 3e3)
    ax.set_xlabel('VSSC-1-H, s'); ax.set_ylabel('events/s with VSSC-1-H ≥ s')
    ax.legend(fontsize=6.3, loc='upper right'); ax.set_title('A  bare origami without probe', fontsize=8, loc='left')
    ax = axes[1]
    for sess, mk in [(B1, 'o'), (B2, 's')]:
        z = Loc[Loc.batch == BLAB[sess]]
        ax.plot(z.s, z.cumulative_slope_raw, marker=mk, ms=3, color=COL['NoCoating'], lw=1, label=f'{BLAB[sess]}, raw')
        ax.plot(z.s, z.cumulative_slope_minus_buffer, marker=mk, ms=3, mfc='white', color=COL['Buffer'], lw=1,
                label=f'{BLAB[sess]}, minus buffer')
    ax.axhline(3.7, color='#C0392B', ls='--', lw=0.8); ax.text(4e3, 3.85, 'value used (3.7)', color='#C0392B', fontsize=6.5, ha='right')
    ax.set_xscale('log'); ax.set_xlim(TRIG, 5e3); ax.set_ylim(0, 7.5); ax.set_xlabel('VSSC-1-H, s')
    ax.set_ylabel('local slope, −d ln N(≥s) / d ln s'); ax.legend(fontsize=6.3, loc='upper right')
    ax.set_title('B  local slope of the cumulative curve', fontsize=8, loc='left')
    ax = axes[2]
    meth = [('cumulative_slope_at_200_raw', 'slope at\n200'), ('density_exponent_200_300_raw', 'MLE\n200–300'),
            ('straight_line_fit_200_to_p95_raw', 'line fit\n200–p95')]
    for j, (sess, dx) in enumerate([(B1, -0.17), (B2, 0.17)]):
        z = Thr[Thr.batch == BLAB[sess]]
        for i, (c, _) in enumerate(meth):
            r = z[z.replicate != 'pooled'][c].values; pv = z[z.replicate == 'pooled'][c].iloc[0]
            ax.scatter(i + dx + rng.uniform(-0.04, 0.04, len(r)), r, s=14, marker='os'[j], color=COL['NoCoating'],
                       edgecolor='white', lw=0.5, zorder=3, label=f'{BLAB[sess]} tubes' if i == 0 else None)
            ax.hlines(pv, i + dx - 0.1, i + dx + 0.1, color=INK, lw=1.4, label='pooled' if i == 0 and j == 0 else None)
    ax.axhline(3.7, color='#C0392B', ls='--', lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in meth], fontsize=7); ax.set_ylim(0, 7.5)
    ax.set_ylabel('exponent'); ax.legend(fontsize=6.3, loc='lower left')
    ax.set_title('C  at the threshold, per tube (raw)', fontsize=8, loc='left')
    save(fig, 'fig41d_scatter_tail_exponent', {'_threshold': Thr, '_threshold_replicate_summary': Rep, '_window_fits': Fits,
                                                '_local_slope': Loc, '_cumulative_curves': Cur})
    return Thr, Rep, Fits


def main(parts):
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)
    if 'q1' in parts:
        print('== Q1'); S, T = q1(); print(T.round(1).to_string(index=False))
        print(S[S.replicate == ''][['batch', 'sample', 'file', 'start', 'acq_time_s', 'all_events_per_s', 'scatter_events_per_s']].round(1).to_string(index=False))
    if 'q2' in parts:
        print('== Q2'); A, C = q2(); print(C.to_string(index=False))
        print(A[A.excluded_over_7000][['folder', 'file', 'manuscript_file', 'all_events_per_s', 'scatter_events_per_s']].round(0).to_string(index=False))
    if 'q3' in parts:
        print('== Q3'); Tst, R = q3(); print(Tst.round(4).T.to_string())
    if 'q4' not in parts:
        return
    print('== Q4'); Thr, Rep, Fits = q4()
    print(Thr.round(3).T.to_string()); print(Rep.T.to_string())


if __name__ == '__main__':
    main(sys.argv[1:] or ['q1', 'q2', 'q3', 'q4'])
