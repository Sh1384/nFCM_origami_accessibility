"""Figure 5 — probe accessibility on coated origami, computed from the raw FCS files.

Panels
  A  Total probe association: log2(with-probe / no-probe) on N-extension monomers
  B  Method 1: log2(biotin ext / N-ext), both with probe
  C  Method 2: % monomers above the same-coating, same-replicate no-biotin P99
  D  Pooled B525-H distributions per coating, biotin extensions overlaid

Two gating modes (same metrics, same layout):

  new  Shared origami gate (shared_utils): SS >= 200 (scatter trigger; removes
       free probe), SS <= per-coating P95 of the pooled no-probe control
       (aggregates), spillover-compensated V525 (V525 - k*B525) <= pooled
       NoCoating no-probe P99 (bright outliers); no V525 lower limit. The session
       buffer, gated identically and scaled by acquisition time, is subtracted
       before every median / percentile / fraction / density. Corona excluded.
       Panel E adds antibody-bound origami counted over the full scatter range
       (quantify_bound_origami_antibody.py: SS >= 200 plus V525 unmixing below
       the scatter trigger).

  old  Manuscript two-component monomer gate: side-scatter window = P25-P75 of
       the coating's pooled no-probe control (after an SS >= 300 floor), plus the
       per-coating V525-H lower bound from the manuscript's peak-enrichment sweep
       (Table 1). No buffer subtraction. Includes FBS corona, as published.
       Printed metrics are checked against manuscript Tables 2 and 3.

Inputs
  Data/Shervin5_25 (streptavidin) and Data/Shervin4_18 (antibody), incl. Buffer.fcs

Outputs (in figures_rerun/)
  Fig5_Probe_Accessibility_{new,old}gates.{pdf,png}
  Fig5_PanelA_only_{new,old}gates.{pdf,png}
  Fig5_metrics_{new,old}gates.csv

Usage
  python make_figure5.py            # both modes
  python make_figure5.py new        # or: old
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, f_oneway

from shared_utils import (
    CH_SS, CH_ATTO, CH_S, COATINGS, FIGURE_DIR, MAX_EVENTS_PER_SEC, load_fcs,
    compute_monomer_gates, apply_gate, describe_gate, acq_time,
    bs_percentile, bs_pct_above, bs_density,
)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.linewidth': 0.9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'legend.frameon': False,
})

OLD_COAT  = ['NoCoating', 'PLL', 'Silica', 'Corona']
COAT_LBL  = {'NoCoating':'Uncoated','PLL':'PLL-PEG','Silica':'Silica','Corona':'FBS corona'}
COAT_COL  = {'NoCoating':'#1F77B4','PLL':'#2CA02C','Silica':'#FF7F0E','Corona':'#9467BD'}
EXT       = ['N', '3', '6', '12']
EXT_LBL   = {'N':'N (no biotin)','3':'3 nt','6':'6 nt','12':'12 nt'}
EXT_COL   = {'N':'#4D4D4D','3':'#377EB8','6':'#4DAF4A','12':'#E41A1C'}
REPS      = ['A', 'B', 'C']

# (key, label, data folder, probe tag, no-probe tag)
SESSIONS = [
    ('strep', 'Streptavidin',    'Shervin5_25', 'S',  'NoS'),
    ('ab',    'Anti-biotin IgG', 'Shervin4_18', 'AB', 'NoAB'),
]

# Manuscript Table 1: side-scatter window and V525-H lower bound per coating.
# The SS window is recomputed from the data and checked against these values
# (the gate itself uses the rounded table values, as the published analysis did);
# the V525-H lower bound (peak-enrichment sweep) is taken from the table.
OLD_GATE_TABLE = {
    'strep': {'NoCoating': (327, 644, 119.5), 'PLL': (326, 593, 145.4),
              'Silica': (484, 1311, 66.7),   'Corona': (497, 2191, 134.9)},
    'ab':    {'NoCoating': (328, 527, 120.2), 'PLL': (341, 768, 148.8),
              'Silica': (399, 4710, 181.1),  'Corona': (438, 1897, 157.7)},
}

# Manuscript Tables 2 and 3 (mean across replicates), for checking the old mode.
MANUSCRIPT = {
    ('strep', 'NoCoating', 'N'): (2.84, None, None), ('ab', 'NoCoating', 'N'): (0.98, None, None),
    ('strep', 'PLL', 'N'):       (3.05, None, None), ('ab', 'PLL', 'N'):       (0.70, None, None),
    ('strep', 'Silica', 'N'):    (0.04, None, None), ('ab', 'Silica', 'N'):    (0.67, None, None),
    ('strep', 'NoCoating', '3'):  (None, -0.872, 0.72), ('ab', 'NoCoating', '3'):  (None, 0.198, 2.16),
    ('strep', 'NoCoating', '6'):  (None, 0.216, 1.89),  ('ab', 'NoCoating', '6'):  (None, 0.180, 3.10),
    ('strep', 'NoCoating', '12'): (None, -0.004, 1.04), ('ab', 'NoCoating', '12'): (None, -0.045, 1.08),
    ('strep', 'PLL', '3'):        (None, -0.067, 0.83), ('ab', 'PLL', '3'):        (None, 0.231, 2.37),
    ('strep', 'PLL', '6'):        (None, -0.044, 0.93), ('ab', 'PLL', '6'):        (None, 0.310, 2.40),
    ('strep', 'PLL', '12'):       (None, -0.016, 0.86), ('ab', 'PLL', '12'):       (None, 0.197, 2.72),
    ('strep', 'Silica', '3'):     (None, 0.024, 2.39),  ('ab', 'Silica', '3'):     (None, 0.142, 1.05),
    ('strep', 'Silica', '6'):     (None, 0.456, 14.27), ('ab', 'Silica', '6'):     (None, 0.080, 0.46),
    ('strep', 'Silica', '12'):    (None, 0.061, 1.00),  ('ab', 'Silica', '12'):    (None, 0.016, 1.36),
}

# ---------------------------------------------------------------- data
_cache = {}
def fcs(folder, fname):
    """Cached FCS load; None if the file does not exist."""
    key = (folder, fname)
    if key not in _cache:
        try:
            _cache[key] = load_fcs(folder, fname)
        except FileNotFoundError:
            _cache[key] = None
    return _cache[key]


class Sample:
    """Gated, positive B525-H values of one or more pooled files, plus their
    total acquisition time (used to scale the buffer in the new mode)."""
    def __init__(self, vals, t, buf=None):
        self.vals, self.t, self.buf = vals, t, buf   # buf: gated buffer Sample (new mode)

    @staticmethod
    def pool(samples):
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        return Sample(np.concatenate([s.vals for s in samples]), sum(s.t for s in samples),
                      samples[0].buf)


class Analysis:
    """Gate + statistics for one mode ('new' or 'old')."""

    def __init__(self, mode):
        self.mode  = mode
        self.coats = COATINGS if mode == 'new' else OLD_COAT
        self.gates, self.buf = {}, {}
        for sess, _, folder, _, nptag in SESSIONS:
            if mode == 'new':
                buffer = fcs(folder, 'Buffer.fcs')
                dfs = {('N', c, nptag, r): d for c in self.coats for r in REPS
                       if (d := fcs(folder, f'N_{c}_{nptag}_{r}.fcs')) is not None}
                g = compute_monomer_gates(dfs, ctl=nptag)
                self.gates[sess] = {c: g[(c, 'A')] for c in self.coats if (c, 'A') in g}
                self.buf[sess] = {}
                for c, gc in self.gates[sess].items():
                    b = apply_gate(buffer, c, 'A', g)[CH_S].values
                    self.buf[sess][c] = Sample(b[b > 0], acq_time(buffer))
                    print(f'[new] {sess} {c:9s}: {describe_gate(gc)}')
            else:
                self.gates[sess] = {}
                for coat in self.coats:
                    pool = np.concatenate([d[CH_SS].values for r in REPS
                                           if (d := fcs(folder, f'N_{coat}_{nptag}_{r}.fcs')) is not None])
                    lo, hi = np.percentile(pool[pool >= 300], [25, 75])
                    t_lo, t_hi, v_lo = OLD_GATE_TABLE[sess][coat]
                    # The published analysis used the rounded Table 1 window
                    self.gates[sess][coat] = (t_lo, t_hi, v_lo)
                    flag = '' if (round(lo) == t_lo and round(hi) == t_hi) else '  <-- differs from Table 1'
                    print(f'[old] {sess} {coat:9s}: SS [{lo:.0f}, {hi:.0f}] (Table 1 [{t_lo}, {t_hi}]), '
                          f'V525 >= {v_lo}{flag}')

    def sample(self, sess, folder, fname, coat):
        d = fcs(folder, fname)
        if d is None:
            return None
        if self.mode == 'new':
            v = apply_gate(d, coat, 'A', {(coat, 'A'): self.gates[sess][coat]})[CH_S].values
        else:
            lo, hi, v_lo = self.gates[sess][coat]
            v = d[(d[CH_SS] >= lo) & (d[CH_SS] <= hi) & (d[CH_ATTO] >= v_lo)][CH_S].values
        v = v[v > 0]
        if not len(v):
            print(f'   [{self.mode}] no gated events in {folder}/{fname}; skipped')
            return None
        return Sample(v, acq_time(d), self.buf[sess][coat] if self.mode == 'new' else None)

    # statistics (buffer-subtracted in the new mode)
    def pct(self, sess, s, q):
        if self.mode == 'new':
            b = s.buf
            return bs_percentile(s.vals, s.t, b.vals, b.t, q)
        return float(np.percentile(s.vals, q))

    def frac_above(self, sess, s, thr):
        if self.mode == 'new':
            b = s.buf
            return bs_pct_above(s.vals, s.t, b.vals, b.t, thr)
        return float((s.vals > thr).mean() * 100)

    def density(self, sess, s, xx):
        """Density of log10(B525-H) on grid xx."""
        if self.mode == 'new':
            b = s.buf
            x, d = bs_density(s.vals, s.t, b.vals, b.t, lo=10 ** xx[0], hi=10 ** xx[-1],
                              per_decade=50, smooth=3.0)
            return np.interp(xx, np.log10(x), d)
        return gaussian_kde(np.log10(s.vals), bw_method=0.15)(xx)

    # metrics table
    def metrics(self):
        rows = []
        for sess, _, folder, ptag, nptag in SESSIONS:
            m_probe_uncoated = {}
            for coat in self.coats:
                for rep in REPS:
                    noprobe = self.sample(sess, folder, f'N_{coat}_{nptag}_{rep}.fcs', coat)
                    nprobe  = self.sample(sess, folder, f'N_{coat}_{ptag}_{rep}.fcs', coat)
                    if nprobe is None or noprobe is None:
                        continue
                    m_probe = self.pct(sess, nprobe, 50)
                    thr     = self.pct(sess, nprobe, 99)
                    if coat == 'NoCoating':
                        m_probe_uncoated[rep] = m_probe
                    for ext in EXT:
                        s = nprobe if ext == 'N' else self.sample(
                            sess, folder, f'{ext}_{coat}_{ptag}_{rep}.fcs', coat)
                        if s is None:
                            continue
                        rows.append({
                            'session': sess, 'coating': coat, 'extension': ext, 'replicate': rep,
                            'log2fc_total': (np.log2(m_probe / self.pct(sess, noprobe, 50))
                                             if ext == 'N' and noprobe is not None else np.nan),
                            'M1_biotin_MFI_shift': np.log2(self.pct(sess, s, 50) / m_probe),
                            'M2_biotin_specific_fraction': self.frac_above(sess, s, thr),
                            'median_B525': self.pct(sess, s, 50),
                            'N_probe_P99_threshold': thr,
                            'n_gated_events': len(s.vals),
                        })
            # Non-specific binding vs same-replicate uncoated: log2(Mprobe(X,r) / Mprobe(Uncoated,r))
            for r in rows:
                if r['session'] == sess and r['extension'] == 'N' and r['replicate'] in m_probe_uncoated:
                    r['log2_nsb_vs_uncoated'] = np.log2(r['median_B525'] / m_probe_uncoated[r['replicate']])
        return pd.DataFrame(rows)


def mean_sem(series):
    v = series.dropna().values
    if len(v) == 0:
        return np.nan, 0.0, v
    sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), sem, v

# ---------------------------------------------------------------- panels
def draw_panel_a(ax, an, df, sess, annotate_anova=True):
    """Total probe association, one bar per coating, coloured by coating."""
    coats = an.coats
    x = np.arange(len(coats))
    for xi, coat in enumerate(coats):
        m, s, pts = mean_sem(
            df[(df.session == sess) & (df.coating == coat) &
               (df.extension == 'N')]['log2fc_total'])
        ax.bar(xi, m, 0.62, yerr=s, capsize=5, color=COAT_COL[coat],
               edgecolor='black', linewidth=0.7, alpha=0.88)
        for v in pts:
            ax.scatter(xi + np.random.uniform(-0.05, 0.05), v, s=30, color='black',
                       edgecolor='white', linewidth=0.6, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([COAT_LBL[c] for c in coats], fontsize=12)
    ax.axhline(0, color='#666', linewidth=0.8, linestyle='--')
    ax.grid(True, axis='y', alpha=0.15, linewidth=0.5)
    ax.tick_params(axis='y', labelsize=11)
    if annotate_anova:
        groups = [df[(df.session == sess) & (df.coating == c) &
                     (df.extension == 'N')]['log2fc_total'].dropna().values for c in coats]
        F, p = f_oneway(*groups)
        dfb, dfw = len(groups) - 1, sum(len(g) for g in groups) - len(groups)
        ax.text(0.03, 0.95, f'F({dfb},{dfw}) = {F:.1f}\np = {p:.1e}', transform=ax.transAxes,
                fontsize=11, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#BBB', alpha=0.95))

def draw_grouped(ax, an, df, sess, col, log_y=False, null=None):
    """Grouped bars by biotin extension (used for Method 1 and Method 2)."""
    coats = an.coats
    x = np.arange(len(coats)); w = 0.20
    for ei, ext in enumerate(EXT):
        vals, errs = [], []
        for coat in coats:
            m, s, _ = mean_sem(df[(df.session == sess) & (df.coating == coat) &
                                  (df.extension == ext)][col])
            vals.append(m); errs.append(s)
        ax.bar(x + (ei - 1.5) * w, vals, w, yerr=errs, capsize=3,
               error_kw={'linewidth': 0.8}, color=EXT_COL[ext],
               edgecolor='black', linewidth=0.6)
        for xi, coat in enumerate(coats):
            _, _, pts = mean_sem(df[(df.session == sess) & (df.coating == coat) &
                                    (df.extension == ext)][col])
            for v in pts:
                ax.scatter(xi + (ei - 1.5) * w + np.random.uniform(-0.015, 0.015), v,
                           s=14, color='black', edgecolor='white',
                           linewidth=0.4, zorder=5, alpha=0.85)
    if null is not None:
        ax.axhline(null, color='red', linestyle='--', linewidth=1.3, alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels([COAT_LBL[c] for c in coats], fontsize=12)
    if log_y:
        ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.15, linewidth=0.5)
    ax.tick_params(axis='y', labelsize=11)

def draw_kde(ax, an, sess, folder, coat, ptag, nptag):
    """Pooled B525-H density per biotin extension, with no-probe reference."""
    xx = np.linspace(1, 4, 400)
    def pooled(ext, tag):
        return Sample.pool([s for r in REPS
                            if (s := an.sample(sess, folder, f'{ext}_{coat}_{tag}_{r}.fcs', coat))
                            is not None and len(s.vals) >= 50])
    np_pool = pooled('N', nptag)
    if np_pool is not None:
        ax.fill_between(xx, an.density(sess, np_pool, xx),
                        color='#B0B0B0', alpha=0.35, edgecolor='#666',
                        linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
    for ext in EXT:
        pool = pooled(ext, ptag)
        if pool is None:
            continue
        ax.plot(xx, an.density(sess, pool, xx),
                color=EXT_COL[ext], linewidth=2.2, alpha=0.95, zorder=3)
    ax.set_xlim(1, 4); ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.10, linewidth=0.3)

def gate_note(an):
    if an.mode == 'new':
        ss = lambda sess: ', '.join(f"{COAT_LBL[c]} ≤{g['ss_high']:.0f}"
                                    for c, g in an.gates[sess].items())
        gs, ga = an.gates['strep']['NoCoating'], an.gates['ab']['NoCoating']
        return ('Origami gate: SS ≥ 200; SS upper (P95) streptavidin ' + ss('strep') +
                '; antibody ' + ss('ab') +
                f"; V525−k·B525 ≤ {gs['v525c_high']:.0f} (k={gs['k']:.2f}) / "
                f"{ga['v525c_high']:.0f} (k={ga['k']:.2f}); no V525 lower limit; buffer-subtracted")
    return 'Manuscript monomer gate: SS P25–P75 window + V525-H lower bound (Table 1), no buffer subtraction'

# ---------------------------------------------------------------- full figure
def build_full_figure(an, df, bound=None):
    coats = an.coats
    n_rows = 5 if bound is not None else 4
    fig = plt.figure(figsize=(16, 18 if n_rows == 4 else 22.5))
    gs  = fig.add_gridspec(n_rows, 1, height_ratios=[1.0, 1.0, 1.0, 1.45, 1.0][:n_rows],
                           hspace=0.55, top=0.925 if n_rows == 4 else 0.94,
                           bottom=0.04, left=0.07, right=0.98)

    def probe_pair(slot, share_y):
        sub  = slot.subgridspec(1, 2, wspace=0.10 if share_y else 0.22)
        ax_s = fig.add_subplot(sub[0, 0])
        ax_a = fig.add_subplot(sub[0, 1], sharey=ax_s if share_y else None)
        if share_y:
            plt.setp(ax_a.get_yticklabels(), visible=False)
        return ax_s, ax_a

    # A - shared y so the streptavidin/antibody magnitudes compare directly
    a_s, a_a = probe_pair(gs[0], share_y=True)
    draw_panel_a(a_s, an, df, 'strep', annotate_anova=False)
    draw_panel_a(a_a, an, df, 'ab',    annotate_anova=False)
    a_s.set_ylabel('log$_2$(with-probe / no-probe)', fontsize=13)
    a_s.set_title('Streptavidin', fontsize=13, pad=8)
    a_a.set_title('Anti-biotin IgG', fontsize=13, pad=8)

    # B - independent y; the antibody effects are ~10x smaller than streptavidin
    b_s, b_a = probe_pair(gs[1], share_y=False)
    draw_grouped(b_s, an, df, 'strep', 'M1_biotin_MFI_shift', null=0)
    draw_grouped(b_a, an, df, 'ab',    'M1_biotin_MFI_shift', null=0)
    for ax in (b_s, b_a):
        ax.set_ylabel('log$_2$(biotin ext / N-ext)', fontsize=13)
    b_s.set_title('Streptavidin', fontsize=13, pad=8)
    b_a.set_title('Anti-biotin IgG', fontsize=13, pad=8)

    # C - shared log y
    c_s, c_a = probe_pair(gs[2], share_y=True)
    draw_grouped(c_s, an, df, 'strep', 'M2_biotin_specific_fraction', log_y=True, null=1)
    draw_grouped(c_a, an, df, 'ab',    'M2_biotin_specific_fraction', log_y=True, null=1)
    c_s.set_ylim(0.1, 50)
    c_s.set_ylabel('% monomers > N-ext P99', fontsize=13)
    c_s.set_title('Streptavidin', fontsize=13, pad=8)
    c_a.set_title('Anti-biotin IgG', fontsize=13, pad=8)

    # D - 2 probe rows x coatings, plus legend column
    bot = gs[3].subgridspec(2, len(coats) + 1, width_ratios=[1] * len(coats) + [0.35],
                            hspace=0.22, wspace=0.12)
    d_top_left = None
    for row, (sess, label, folder, ptag, nptag) in enumerate(SESSIONS):
        for col, coat in enumerate(coats):
            ax = fig.add_subplot(bot[row, col])
            if row == 0 and col == 0:
                d_top_left = ax
            draw_kde(ax, an, sess, folder, coat, ptag, nptag)
            if row == 0:
                ax.set_title(COAT_LBL[coat], fontsize=12, fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(label, fontsize=12, fontweight='bold')
            if row == 1:
                ax.set_xlabel('log$_{10}$(B525-H)', fontsize=11)
            ax.tick_params(axis='x', labelsize=10)

    leg = fig.add_subplot(bot[:, len(coats)]); leg.axis('off')
    kde_handles = [Patch(facecolor='#B0B0B0', alpha=0.35, edgecolor='#666',
                         linestyle='--', label='No-probe')]
    kde_handles += [Line2D([0], [0], color=EXT_COL[e], linewidth=2.2, label=EXT_LBL[e])
                    for e in EXT]
    leg.legend(handles=kde_handles, loc='center left', fontsize=11,
               title='Biotin extension', title_fontsize=12, handlelength=1.8)

    headers = [
        (a_s, 'A  Total probe association: log$_2$(with-probe / no-probe)'),
        (b_s, 'B  Method 1 — Biotin B525-H median shift'),
        (c_s, 'C  Method 2 — Biotin-specific event fraction (log scale)'),
        (d_top_left, 'D  Biotin-length dose-response: B525-H probe distributions per coating'),
    ]

    # E - antibody-bound origami over the full scatter range (unmixing below trigger)
    if bound is not None:
        e_l, e_r = probe_pair(gs[4], share_y=False)
        draw_grouped(e_l, an, bound, 'ab', 'pct_bound')
        draw_grouped(e_r, an, bound, 'ab', 'delta_vs_N', null=0)
        e_l.set_ylabel('% of origami bound', fontsize=13)
        e_r.set_ylabel('Δ % bound (ext − N)', fontsize=13)
        e_l.set_title('Anti-biotin IgG: % bound', fontsize=13, pad=8)
        e_r.set_title('Anti-biotin IgG: biotin-specific change', fontsize=13, pad=8)
        headers.append((e_l, 'E  Antibody-bound origami, both scatter regions '
                             '(gated SS ≥ 200 + V525 unmixing below the scatter trigger)'))

    # Panel headers placed from the actual axes geometry so they never collide
    # with the per-subplot titles, whatever the gridspec spacing.
    HEADER_OFFSET = 0.021 if n_rows == 4 else 0.017
    for ax, txt in headers:
        y = ax.get_position().y1 + HEADER_OFFSET
        fig.text(0.070, y, txt, ha='left', fontsize=14, fontweight='bold', va='bottom')
    fig.text(0.070, a_s.get_position().y1 + HEADER_OFFSET + 0.018, gate_note(an), ha='left',
             va='bottom', fontsize=10, style='italic', color='#444', wrap=True)

    # Extension legend sits at subplot-title height, in the gap between the
    # two panel-B subplots.
    ext_handles = [Patch(facecolor=EXT_COL[e], edgecolor='black', linewidth=0.5,
                         label=EXT_LBL[e]) for e in EXT]
    ext_handles.append(Line2D([0], [0], color='red', linestyle='--',
                              linewidth=1.2, label='Null'))
    fig.legend(handles=ext_handles, loc='lower center',
               bbox_to_anchor=(0.53, b_s.get_position().y1 + 0.004),
               ncol=5, fontsize=11, handlelength=1.5, columnspacing=1.5,
               frameon=False, title='Biotin extension (panels B, C, D):',
               title_fontsize=11)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURE_DIR, f'Fig5_Probe_Accessibility_{an.mode}gates.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote Fig5_Probe_Accessibility_{an.mode}gates.{{pdf,png}}')

# ---------------------------------------------------------------- panel A alone
def build_panel_a_standalone(an, df):
    """Aspect ratio matched to panel B of the full figure. No ANOVA box."""
    fig = plt.figure(figsize=(16, 4.5))
    gs  = fig.add_gridspec(1, 2, wspace=0.10, top=0.80, bottom=0.16, left=0.07, right=0.98)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1], sharey=ax_s)
    plt.setp(ax_a.get_yticklabels(), visible=False)
    draw_panel_a(ax_s, an, df, 'strep', annotate_anova=False)
    draw_panel_a(ax_a, an, df, 'ab',    annotate_anova=False)
    ax_s.set_ylabel('log$_2$(with-probe / no-probe)', fontsize=13)
    ax_s.set_title('Streptavidin', fontsize=13, pad=8)
    ax_a.set_title('Anti-biotin IgG', fontsize=13, pad=8)
    fig.text(0.07, 0.95, 'A  Total probe association: log₂(with-probe / no-probe)',
             ha='left', fontsize=14, fontweight='bold', va='top')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIGURE_DIR, f'Fig5_PanelA_only_{an.mode}gates.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote Fig5_PanelA_only_{an.mode}gates.{{pdf,png}}')

# ---------------------------------------------------------------- summaries
def summary(df):
    """Mean across replicates per (session, coating, extension)."""
    return df.groupby(['session', 'coating', 'extension'])[
        ['log2fc_total', 'M1_biotin_MFI_shift', 'M2_biotin_specific_fraction']].mean()

def check_against_manuscript(df):
    s = summary(df)
    worst = 0.0
    print('\n[old] check against manuscript Tables 2 and 3 (mean across replicates):')
    if MAX_EVENTS_PER_SEC is not None:
        print(f'   note: files above {MAX_EVENTS_PER_SEC:,} events/s are excluded, so values '
              'that used those files will not match the manuscript')
    for (sess, coat, ext), (a, m1, m2) in MANUSCRIPT.items():
        if (sess, coat, ext) not in s.index:
            print(f'   NO DATA  {sess} {coat} {ext} (all replicates excluded)')
            worst = max(worst, 1.0)
            continue
        row = s.loc[(sess, coat, ext)]
        for name, ref, val, tol in [('A', a, row.log2fc_total, 0.006),
                                    ('M1', m1, row.M1_biotin_MFI_shift, 0.0006),
                                    ('M2', m2, row.M2_biotin_specific_fraction, 0.006)]:
            if ref is None:
                continue
            worst = max(worst, abs(val - ref) - tol)
            if abs(val - ref) > tol:
                print(f'   MISMATCH {sess} {coat} {ext} {name}: computed {val:.3f}, manuscript {ref}')
    print('   all values match the manuscript to its reported precision' if worst <= 0
          else '   (see mismatches above)')

def compare_modes(old, new):
    s_old, s_new = summary(old), summary(new)
    both = s_old.join(s_new, lsuffix='_old', rsuffix='_new', how='inner')
    cols = []
    for m in ['log2fc_total', 'M1_biotin_MFI_shift', 'M2_biotin_specific_fraction']:
        cols += [f'{m}_old', f'{m}_new']
    print('\n=== Old vs new gates (mean across replicates; Corona only in old) ===')
    print(both[cols].rename(columns=lambda c: c.replace('_biotin_MFI_shift', '')
                            .replace('_biotin_specific_fraction', '')
                            .replace('log2fc_total', 'A')).to_string(float_format='%.3f'))

def run(mode):
    np.random.seed(0)          # fixes the scatter jitter so reruns are identical
    an = Analysis(mode)
    df = an.metrics()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    out_csv = os.path.join(FIGURE_DIR, f'Fig5_metrics_{mode}gates.csv')
    df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')
    bound = None
    if mode == 'new':
        from quantify_bound_origami_antibody import quantify, with_delta
        b = with_delta(quantify(verbose=False))
        bound = b.rename(columns={'coating': 'coating', 'ext': 'extension'}).assign(session='ab')
        bound = bound[bound.coating.isin(an.coats)]
        bound.to_csv(os.path.join(FIGURE_DIR, 'Fig5_panelE_bound_newgates.csv'), index=False)
    build_full_figure(an, df, bound)
    build_panel_a_standalone(an, df)
    return df

if __name__ == '__main__':
    modes = sys.argv[1:] or ['new', 'old']
    results = {m: run(m) for m in modes}
    if 'old' in results:
        check_against_manuscript(results['old'])
    if {'old', 'new'} <= results.keys():
        compare_modes(results['old'], results['new'])
