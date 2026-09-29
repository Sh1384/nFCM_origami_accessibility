"""Figure 5: probe accessibility on coated origami.

The central result figure. Separates how much probe sticks non-specifically from
how much the biotin handle actually captures, for both probes.

Panels
  A  Total probe association on no-biotin monomers, with probe against without
  B  Method 1, log2 shift in median probe signal from adding a biotin handle
  C  Method 2, percent of monomers above the no-biotin threshold, 1 percent null
  D  Pooled probe-channel distributions per coating, extensions overlaid

Panels A and C share a y-axis between probes so the two are directly comparable;
panel B is scaled independently because its range is much narrower. Panel A is
also written standalone at the same aspect ratio, as the manuscript uses it
separately.

Inputs   raw FCS; config/gates.json; results/main/method1_method2_metrics.csv
Outputs  figures/main/Fig5_Probe_Accessibility_manuscript.{pdf,png}
         figures/main/Fig5_PanelA_only.{pdf,png}
Usage    python scripts/05_figure5_accessibility.py
"""
import warnings; warnings.filterwarnings('ignore')
import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from fcsparser import parse
from scipy.stats import gaussian_kde, f_oneway


# --- repository bootstrap -------------------------------------------------
# Resolve `nfcm` whether or not the package has been pip-installed, and take
# every path from nfcm.paths so this script holds no absolute path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from nfcm.gating import load_gates
from nfcm.paths import (
    CACHE_DIR, FIGURES_MAIN, RESULTS_MAIN,
    data_root, require_data_root,
)

#: Root of the raw FCS archive. See README.md for the download link and
#: nfcm.paths.data_root for how the location is resolved.
PROJ = str(require_data_root())
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG  = str(FIGURES_MAIN)
OUT  = str(FIGURES_MAIN)
RES  = str(RESULTS_MAIN)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.linewidth': 0.9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'legend.frameon': False,
})

COAT      = ['NoCoating', 'PLL', 'Silica']
COAT_LBL  = {'NoCoating':'Uncoated','PLL':'PLL-PEG','Silica':'Silica'}
COAT_COL  = {'NoCoating':'#1F77B4','PLL':'#2CA02C','Silica':'#FF7F0E'}
EXT       = ['N', '3', '6', '12']
EXT_LBL   = {'N':'N (no biotin)','3':'3 nt','6':'6 nt','12':'12 nt'}
EXT_COL   = {'N':'#4D4D4D','3':'#377EB8','6':'#4DAF4A','12':'#E41A1C'}
REPS      = ['A', 'B', 'C']

SESSIONS = [
    ('strep', 'Streptavidin',    'Shervin5_25',              'S',  'NoS',  'SS_S', 'V525_S_LO'),
    ('ab',    'Anti-biotin IgG', 'Shervin4_18/Shervin4_18',  'AB', 'NoAB', 'SS_A', 'V525_A_LO'),
]

# ---------------------------------------------------------------- data
gates = load_gates()
df    = pd.read_csv(f'{RES}/method1_method2_metrics.csv')

def monomer_b525(session_dir, ext, coat, tag, rep, ss_key, v_key):
    """B525-H of events inside the two-component monomer gate."""
    fp = f'{PROJ}/{session_dir}/{ext}_{coat}_{tag}_{rep}.fcs'
    if not os.path.exists(fp):
        return None
    d = parse(fp, reformat_meta=True)[1]
    lo, hi = gates[ss_key][coat]
    vlo    = gates[v_key][coat]
    arr = d[(d['VSSC-1-H'] >= lo) & (d['VSSC-1-H'] <= hi) &
            (d['V525-H']   >= vlo)]['B525-H'].values
    return arr[arr > 0]

def mean_sem(series):
    v = series.dropna().values
    if len(v) == 0:
        return np.nan, 0.0, v
    sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), sem, v

# ---------------------------------------------------------------- panels
def draw_panel_a(ax, sess, annotate_anova=True):
    """Total probe association, one bar per coating, coloured by coating."""
    x = np.arange(len(COAT))
    for xi, coat in enumerate(COAT):
        m, s, pts = mean_sem(
            df[(df.session == sess) & (df.coating == coat) &
               (df.extension == 'N')]['log2fc_total'])
        ax.bar(xi, m, 0.62, yerr=s, capsize=5, color=COAT_COL[coat],
               edgecolor='black', linewidth=0.7, alpha=0.88)
        for v in pts:
            ax.scatter(xi + np.random.uniform(-0.05, 0.05), v, s=30, color='black',
                       edgecolor='white', linewidth=0.6, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([COAT_LBL[c] for c in COAT], fontsize=12)
    ax.axhline(0, color='#666', linewidth=0.8, linestyle='--')
    ax.grid(True, axis='y', alpha=0.15, linewidth=0.5)
    ax.tick_params(axis='y', labelsize=11)
    if annotate_anova:
        groups = [df[(df.session == sess) & (df.coating == c) &
                     (df.extension == 'N')]['log2fc_total'].values for c in COAT]
        F, p = f_oneway(*groups)
        ax.text(0.03, 0.95, f'F(2,6) = {F:.1f}\np = {p:.1e}', transform=ax.transAxes,
                fontsize=11, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#BBB', alpha=0.95))

def draw_grouped(ax, sess, col, log_y=False, null=None):
    """Grouped bars by biotin extension (used for Method 1 and Method 2)."""
    x = np.arange(len(COAT)); w = 0.20
    for ei, ext in enumerate(EXT):
        vals, errs = [], []
        for coat in COAT:
            m, s, _ = mean_sem(df[(df.session == sess) & (df.coating == coat) &
                                  (df.extension == ext)][col])
            vals.append(m); errs.append(s)
        ax.bar(x + (ei - 1.5) * w, vals, w, yerr=errs, capsize=3,
               error_kw={'linewidth': 0.8}, color=EXT_COL[ext],
               edgecolor='black', linewidth=0.6)
        for xi, coat in enumerate(COAT):
            _, _, pts = mean_sem(df[(df.session == sess) & (df.coating == coat) &
                                    (df.extension == ext)][col])
            for v in pts:
                ax.scatter(xi + (ei - 1.5) * w + np.random.uniform(-0.015, 0.015), v,
                           s=14, color='black', edgecolor='white',
                           linewidth=0.4, zorder=5, alpha=0.85)
    if null is not None:
        ax.axhline(null, color='red', linestyle='--', linewidth=1.3, alpha=0.75)
    ax.set_xticks(x); ax.set_xticklabels([COAT_LBL[c] for c in COAT], fontsize=12)
    if log_y:
        ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.15, linewidth=0.5)
    ax.tick_params(axis='y', labelsize=11)

def draw_kde(ax, session_dir, coat, ptag, nptag, ss_key, v_key):
    """Pooled B525-H density per biotin extension, with no-probe reference."""
    xx = np.linspace(1, 4, 400)
    np_pool = [a for r in REPS
               if (a := monomer_b525(session_dir, 'N', coat, nptag, r, ss_key, v_key))
               is not None and len(a) >= 50]
    if np_pool:
        pooled = np.concatenate(np_pool)
        ax.fill_between(xx, gaussian_kde(np.log10(pooled), bw_method=0.15)(xx),
                        color='#B0B0B0', alpha=0.35, edgecolor='#666',
                        linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
    for ext in EXT:
        pool = [a for r in REPS
                if (a := monomer_b525(session_dir, ext, coat, ptag, r, ss_key, v_key))
                is not None and len(a) >= 50]
        if not pool:
            continue
        arr = np.concatenate(pool)
        ax.plot(xx, gaussian_kde(np.log10(arr), bw_method=0.15)(xx),
                color=EXT_COL[ext], linewidth=2.2, alpha=0.95, zorder=3)
    ax.set_xlim(1, 4); ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.10, linewidth=0.3)

# ---------------------------------------------------------------- full figure
def build_full_figure():
    fig = plt.figure(figsize=(16, 18))
    gs  = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 1.45],
                           hspace=0.55, top=0.925, bottom=0.04, left=0.07, right=0.98)

    def probe_pair(slot, share_y):
        sub  = slot.subgridspec(1, 2, wspace=0.10 if share_y else 0.22)
        ax_s = fig.add_subplot(sub[0, 0])
        ax_a = fig.add_subplot(sub[0, 1], sharey=ax_s if share_y else None)
        if share_y:
            plt.setp(ax_a.get_yticklabels(), visible=False)
        return ax_s, ax_a

    # A - shared y so the streptavidin/antibody magnitudes compare directly
    a_s, a_a = probe_pair(gs[0], share_y=True)
    draw_panel_a(a_s, 'strep', annotate_anova=False)
    draw_panel_a(a_a, 'ab',    annotate_anova=False)
    a_s.set_ylabel('log$_2$(with-probe / no-probe)', fontsize=13)
    a_s.set_title('Streptavidin', fontsize=13, pad=8)
    a_a.set_title('Anti-biotin IgG', fontsize=13, pad=8)

    # B - independent y; the antibody effects are ~10x smaller than streptavidin
    b_s, b_a = probe_pair(gs[1], share_y=False)
    draw_grouped(b_s, 'strep', 'M1_biotin_MFI_shift', null=0)
    draw_grouped(b_a, 'ab',    'M1_biotin_MFI_shift', null=0)
    for ax in (b_s, b_a):
        ax.set_ylabel('log$_2$(biotin ext / N-ext)', fontsize=13)
    b_s.set_title('Streptavidin', fontsize=13, pad=8)
    b_a.set_title('Anti-biotin IgG', fontsize=13, pad=8)

    # C - shared log y
    c_s, c_a = probe_pair(gs[2], share_y=True)
    draw_grouped(c_s, 'strep', 'M2_biotin_specific_fraction', log_y=True, null=1)
    draw_grouped(c_a, 'ab',    'M2_biotin_specific_fraction', log_y=True, null=1)
    c_s.set_ylim(0.1, 50)
    c_s.set_ylabel('% monomers > N-ext P99', fontsize=13)
    c_s.set_title('Streptavidin', fontsize=13, pad=8)
    c_a.set_title('Anti-biotin IgG', fontsize=13, pad=8)

    # D - 2 probe rows x 4 coatings, plus legend column
    bot = gs[3].subgridspec(2, 4, width_ratios=[1, 1, 1, 0.42],
                            hspace=0.22, wspace=0.12)
    d_top_left = None
    for row, (_, label, sdir, ptag, nptag, ss_key, v_key) in enumerate(SESSIONS):
        for col, coat in enumerate(COAT):
            ax = fig.add_subplot(bot[row, col])
            if row == 0 and col == 0:
                d_top_left = ax
            draw_kde(ax, sdir, coat, ptag, nptag, ss_key, v_key)
            if row == 0:
                ax.set_title(COAT_LBL[coat], fontsize=12, fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(label, fontsize=12, fontweight='bold')
            if row == 1:
                ax.set_xlabel('log$_{10}$(B525-H)', fontsize=11)
            ax.tick_params(axis='x', labelsize=10)

    leg = fig.add_subplot(bot[:, 3]); leg.axis('off')
    kde_handles = [Patch(facecolor='#B0B0B0', alpha=0.35, edgecolor='#666',
                         linestyle='--', label='No-probe')]
    kde_handles += [Line2D([0], [0], color=EXT_COL[e], linewidth=2.2, label=EXT_LBL[e])
                    for e in EXT]
    leg.legend(handles=kde_handles, loc='center left', fontsize=11,
               title='Biotin extension', title_fontsize=12, handlelength=1.8)

    # Panel headers placed from the actual axes geometry so they never collide
    # with the per-subplot titles, whatever the gridspec spacing.
    HEADER_OFFSET = 0.021
    for ax, txt in [
        (a_s, 'A  Total probe association: log$_2$(with-probe / no-probe)'),
        (b_s, 'B  Method 1 \u2014 Biotin B525-H median shift'),
        (c_s, 'C  Method 2 \u2014 Biotin-specific event fraction (log scale)'),
        (d_top_left, 'D  Biotin-length dose-response: B525-H probe distributions per coating'),
    ]:
        y = ax.get_position().y1 + HEADER_OFFSET
        fig.text(0.070, y, txt, ha='left', fontsize=14, fontweight='bold', va='bottom')

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

    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/Fig5_Probe_Accessibility_manuscript.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote Fig5_Probe_Accessibility.{pdf,png}')

# ---------------------------------------------------------------- panel A alone
def build_panel_a_standalone():
    """Aspect ratio matched to panel B of the full figure. No ANOVA box."""
    fig = plt.figure(figsize=(16, 4.5))
    gs  = fig.add_gridspec(1, 2, wspace=0.10, top=0.80, bottom=0.16, left=0.07, right=0.98)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1], sharey=ax_s)
    plt.setp(ax_a.get_yticklabels(), visible=False)
    draw_panel_a(ax_s, 'strep', annotate_anova=False)
    draw_panel_a(ax_a, 'ab',    annotate_anova=False)
    ax_s.set_ylabel('log$_2$(with-probe / no-probe)', fontsize=13)
    ax_s.set_title('Streptavidin', fontsize=13, pad=8)
    ax_a.set_title('Anti-biotin IgG', fontsize=13, pad=8)
    fig.text(0.07, 0.95, 'A  Total probe association: log₂(with-probe / no-probe)',
             ha='left', fontsize=14, fontweight='bold', va='top')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/Fig5_PanelA_only.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote Fig5_PanelA_only.{pdf,png}')

if __name__ == '__main__':
    np.random.seed(0)          # fixes the scatter jitter so reruns are identical
    build_full_figure()
    build_panel_a_standalone()
