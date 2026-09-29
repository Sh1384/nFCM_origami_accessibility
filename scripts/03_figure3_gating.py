"""Figure 3: instrumentation characterisation and the monomer gate.

Shows how each coating moves the measurement window and how the two-component
gate is derived in response.

Rows
  A  Median side scatter, median ATTO 425, and side-scatter CV per coating
  B  Spectral separation of the two fluorescence channels, from buffer,
     ATTO-only, and AF488-only controls
  C  Side scatter against ATTO 425 per coating, with the derived gate drawn on
  D  The V525-H peak-enrichment sweep behind each coating's lower bound

Row C is the clearest view of why the gate cannot be universal: the three gates
sit in visibly different places because the coatings scatter differently.

Inputs   raw FCS; config/gates.json
Outputs  figures/main/Fig3_Instrumentation_Gating_manuscript.{pdf,png}
Usage    python scripts/03_figure3_gating.py
"""
import warnings; warnings.filterwarnings('ignore')
from fcsparser import parse
import numpy as np, os, json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,'axes.linewidth':0.9,
    'axes.spines.top':False,'axes.spines.right':False,
    'xtick.major.width':0.9,'ytick.major.width':0.9,'legend.frameon':False,
})


# --- repository bootstrap -------------------------------------------------
# Resolve `nfcm` whether or not the package has been pip-installed, and take
# every path from nfcm.paths so this script holds no absolute path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from nfcm.gating import load_gates
from nfcm.paths import (
    CACHE_DIR, CONFIG_DIR, FIGURES_MAIN, FIGURES_SI,
    RESULTS_INTERMEDIATE, RESULTS_MAIN, RESULTS_SI, require_data_root,
)

#: Root of the raw FCS archive. See README.md for the download link and
#: nfcm.paths.data_root for how the location is resolved.
PROJ = str(require_data_root())
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT  = str(FIGURES_MAIN)
COATINGS    = ['NoCoating','PLL','Silica']
COAT_LABELS = {'NoCoating':'Uncoated','PLL':'PLL-PEG','Silica':'Silica'}
COAT_COLORS = {'NoCoating':'#1F77B4','PLL':'#2CA02C','Silica':'#FF7F0E'}
REPS = ['A','B','C']

g = load_gates()
SS_S, SS_A = {k:tuple(v) for k,v in g['SS_S'].items()}, {k:tuple(v) for k,v in g['SS_A'].items()}
VLO_S, VLO_A = g['V525_S_LO'], g['V525_A_LO']

SESS = {
    'strep': dict(root='Shervin5_25',             np_tag='NoS', p_tag='S',
                  SS=SS_S, VLO=VLO_S, label='Streptavidin'),
    'ab':    dict(root='Shervin4_18/Shervin4_18', np_tag='NoAB', p_tag='AB',
                  SS=SS_A, VLO=VLO_A, label='Anti-biotin IgG'),
}
DEBRIS = {'NoCoating': f'{PROJ}/Shervin5_25/Buffer.fcs',
          'PLL':       f'{PROJ}/Shervin2_13_2026/PLL_PEG_NoOrigami.fcs',
          'Silica':    f'{PROJ}/Shervin2_13_2026/SilicaChemicalsOnly.fcs'}

def load(fp): return parse(fp, reformat_meta=True)[1]
def load_v(fp, ssr):
    d = load(fp); lo, hi = ssr
    v = d[(d['VSSC-1-H']>=lo)&(d['VSSC-1-H']<=hi)]['V525-H'].values
    return v[v>0]

fig = plt.figure(figsize=(15.5, 14.5))
gs = fig.add_gridspec(4, 1, height_ratios=[0.80, 0.80, 1.00, 1.85],
                      hspace=0.52, top=0.950, bottom=0.045, left=0.070, right=0.985)

# ---------------------------------------------------------------- Row A
rowA = gs[0].subgridspec(1, 3, wspace=0.42)
axA = [fig.add_subplot(rowA[0,i]) for i in range(3)]
ss_med, v_med, ss_cv = ({c:[] for c in COATINGS} for _ in range(3))
for coat in COATINGS:
    for rep in REPS:
        fp = f'{PROJ}/Shervin5_25/N_{coat}_NoS_{rep}.fcs'
        if not os.path.exists(fp): continue
        d = load(fp); gg = d[d['VSSC-1-H']>=100]
        ss = gg['VSSC-1-H'].values; v = gg['V525-H'].values
        ss_med[coat].append(float(np.median(ss)))
        v_med[coat].append(float(np.median(v[v>0])))
        ss_cv[coat].append(100*float(np.std(ss)/np.mean(ss)))

rng = np.random.default_rng(7)
for ax, (ylab, data) in zip(axA, [
        ('Median VSSC-1-H (a.u.)', ss_med),
        ('Median V525-H (a.u.)',   v_med),
        ('VSSC-1-H CV (%)',        ss_cv)]):
    x = np.arange(len(COATINGS))
    for xi, c in zip(x, COATINGS):
        vals = data[c]; m = np.mean(vals)
        s = np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0
        ax.bar(xi, m, 0.6, yerr=s, capsize=4, color=COAT_COLORS[c],
               edgecolor='black', linewidth=0.6, alpha=0.88)
        for v_ in vals:
            ax.scatter(xi+rng.uniform(-0.05,0.05), v_, s=22, color='black',
                       edgecolor='white', linewidth=0.5, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([COAT_LABELS[c] for c in COATINGS],
                                         fontsize=10, rotation=15)
    ax.set_ylabel(ylab, fontsize=11)
    ax.grid(True, axis='y', alpha=0.18, linewidth=0.5)

# ---------------------------------------------------------------- Row B
rowB = gs[1].subgridspec(1, 2, wspace=0.28)
axB1, axB2 = fig.add_subplot(rowB[0,0]), fig.add_subplot(rowB[0,1])
buf   = load(f'{PROJ}/Shervin5_25/Buffer.fcs')
atto  = load(f'{PROJ}/Shervin5_25/N_NoCoating_NoS_A.fcs')
af488 = load(f'{PROJ}/Shervin5_25/N_NoCoating_S_A.fcs')
def kde_plot(ax, arr, ch, color, label):
    v = arr[ch].values; v = v[v>0]
    kde = gaussian_kde(np.log10(v), bw_method=0.10)
    xs = np.linspace(-0.5, 4, 400)
    ax.plot(10**xs, kde(xs), color=color, linewidth=2.2, label=label)
for ax, xlab, ch, order in [
        (axB1, 'V525-H (ATTO 425 channel)', 'V525-H',
         [('Buffer', buf, '#888'), ('AF488 only', af488, '#FF7F0E'), ('ATTO only', atto, '#1F77B4')]),
        (axB2, 'B525-H (AF488 channel)',    'B525-H',
         [('Buffer', buf, '#888'), ('ATTO only', atto, '#1F77B4'), ('AF488 only', af488, '#FF7F0E')])]:
    for label, d, color in order: kde_plot(ax, d, ch, color, label)
    ax.set_xscale('log'); ax.set_xlim(1, 1e4)
    ax.set_xlabel(xlab, fontsize=11); ax.set_ylabel('Density', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.1, linewidth=0.4)

# ---------------------------------------------------------------- Row C
rowC = gs[2].subgridspec(1, 3, wspace=0.22)
for i, coat in enumerate(COATINGS):
    ax = fig.add_subplot(rowC[0,i])
    d = load(f'{PROJ}/Shervin5_25/N_{coat}_NoS_A.fcs')
    ax.scatter(d['VSSC-1-H'], d['V525-H'], s=1.0, alpha=0.35,
               color=COAT_COLORS[coat], edgecolor='none', rasterized=True)
    lo, hi = SS_S[coat]; vlo = VLO_S[coat]
    ax.add_patch(Rectangle((lo, vlo), hi-lo, 1e4-vlo, facecolor='red', alpha=0.18,
                           edgecolor='red', linewidth=1.7, zorder=4))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e2, 1e6); ax.set_ylim(1e1, 1e4)
    ax.set_xlabel('VSSC-1-H (Side Scatter)', fontsize=10.5)
    if i == 0: ax.set_ylabel('V525-H (ATTO 425)', fontsize=10.5)
    ax.set_title(COAT_LABELS[coat], fontsize=12, fontweight='bold', pad=5)
    ax.text(0.04, 0.96, f'SS $\\in$ [{lo:.0f}, {hi:.0f}]\nV525 $\\geq$ {vlo:.1f}',
            transform=ax.transAxes, fontsize=9.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BBB', alpha=0.92), zorder=6)

# ---------------------------------------------------------------- Row D
rowD = gs[3].subgridspec(2, 3, wspace=0.30, hspace=0.55)
thr_grid = np.logspace(np.log10(50), np.log10(500), 400)
for r_i, skey in enumerate(['strep','ab']):
    S = SESS[skey]
    for i, coat in enumerate(COATINGS):
        ax = fig.add_subplot(rowD[r_i, i])
        ssr = S['SS'][coat]; vlo = S['VLO'][coat]
        sig = np.concatenate([load_v(f'{PROJ}/{S["root"]}/N_{coat}_{S["p_tag"]}_{rep}.fcs', ssr)
                              for rep in REPS
                              if os.path.exists(f'{PROJ}/{S["root"]}/N_{coat}_{S["p_tag"]}_{rep}.fcs')])
        deb = load_v(DEBRIS[coat], ssr)
        sp = np.array([(sig>t).mean() for t in thr_grid])*100
        dp = np.array([(deb>t).mean() for t in thr_grid])*100
        sp_lo = (sig>vlo).mean()*100; dp_lo = (deb>vlo).mean()*100
        enr = sp_lo/max(dp_lo, 1e-4)
        ax.plot(thr_grid, sp, color='#1F77B4', linewidth=2.2)
        ax.plot(thr_grid, dp, color='#666', linewidth=2.2, linestyle='--')
        ax.axhline(5, color='#D62728', linewidth=1.3, linestyle=':', alpha=0.75)
        ax.axvline(vlo, color='black', linewidth=1.6, alpha=0.85)
        ax.axvspan(vlo, thr_grid[-1], alpha=0.10, color='#2CA02C')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(thr_grid[0], thr_grid[-1]); ax.set_ylim(0.01, 200)
        ax.set_xticks([60, 100, 200, 300, 500])
        ax.set_xticklabels(['60','100','200','300','500'], fontsize=9.5)
        ax.minorticks_off()
        ax.set_xlabel('V525-H threshold (a.u.)', fontsize=10.5)
        if i == 0: ax.set_ylabel(f'Pass-rate (%)\n({S["label"]})', fontsize=10.5)
        ax.set_title(COAT_LABELS[coat], fontsize=12, fontweight='bold', pad=5)
        ax.text(0.97, 0.97,
                f'V525 $\\geq$ {vlo:.1f}\nsig {sp_lo:.1f}%\ndeb {dp_lo:.2f}%\nenr {enr:,.0f}$\\times$',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BBB', alpha=0.92))
        print(f'  D {skey:<6}{coat:<10} V525>={vlo:6.1f}  sig {sp_lo:5.1f}%  deb {dp_lo:6.2f}%  enr {enr:,.0f}x')

# ---------------------------------------------------------------- headers
fig.canvas.draw()
def hdr(ax, txt, off=0.016, x=0.070, fs=13.5):
    fig.text(x, ax.get_position().y1+off, txt, ha='left', fontsize=fs,
             fontweight='bold', va='bottom')
hdr(axA[0], 'A  Coating characterization (bare origami, no probe)')
hdr(axB1,   'B  Spectral crosstalk between V525 and B525 channels')
hdr(fig.axes[5], 'C  Monomer gate region (side-scatter window and V525 lower bound)')
hdr(fig.axes[8], 'D  V525-H peak-enrichment sweep with 5% retention floor (both probe sessions)')
# shared legend for row D, on the D header line
yD = fig.axes[8].get_position().y1 + 0.016
fig.legend(handles=[Line2D([],[],color='#1F77B4',lw=2.2,label='Signal retained'),
                    Line2D([],[],color='#666',lw=2.2,ls='--',label='Debris pass'),
                    Line2D([],[],color='#D62728',lw=1.3,ls=':',label='5% floor')],
           loc='lower right', bbox_to_anchor=(0.985, yD), ncol=3, fontsize=10.5,
           frameon=False, handlelength=1.8, columnspacing=1.4)

for e in ['pdf','png']:
    fig.savefig(f'{OUT}/Fig3_Instrumentation_Gating_manuscript.{e}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('\nwrote Fig3_Instrumentation_Gating_manuscript (3 coatings)')
