"""Shared utilities for paper figure generation.

Centralizes:
- channel names, coating/extension/replicate lists
- color palettes
- FCS loader
- origami gate: SS >= 200 (scatter trigger), per-coating SS upper limit
  (aggregates), spillover-compensated V525-H upper limit; no V525 lower limit
- gate-application helper
- buffer-background subtraction (buffer scaled by acquisition time)
- per-coating B525 threshold computation
- matplotlib style setup
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fcsparser import parse

warnings.filterwarnings('ignore')

# ── Channels and structure ───────────────────────────────────────────────────
CH_SS   = 'VSSC-1-H'
CH_ATTO = 'V525-H'
CH_S    = 'B525-H'   # AF488 (streptavidin or anti-biotin)
CH_AB   = 'B525-H'   # alias for legacy antibody scripts

COATINGS    = ['NoCoating', 'PLL', 'Silica']
EXTENSIONS  = ['N', '3', '6', '12']
REPLICATES  = ['A', 'B', 'C']

# Log-spaced bins used for buffer-subtracted histograms (100 bins/decade)
BS_BINS = np.logspace(-1, 7, 801)

# ── Color palette (matches the notebooks) ────────────────────────────────────
COAT_COLORS = {
    'NoCoating': '#1F77B4',
    'PLL':       '#2CA02C',
    'Silica':    '#FF7F0E',
}
EXT_COLORS = {
    'N':  '#333333',
    '3':  '#1F77B4',
    '6':  '#2CA02C',
    '12': '#D62728',
}
EXT_STYLES = {
    'N':  {'ls': ':',  'lw': 1.2},
    '3':  {'ls': '-',  'lw': 2.0},
    '6':  {'ls': '--', 'lw': 2.0},
    '12': {'ls': '-.', 'lw': 2.0},
}
REP_MARKERS = {'A': 'o', 'B': 's', 'C': '^'}

CH_TITLES = {
    CH_SS:   'VSSC-1-H (Side Scatter)',
    CH_ATTO: 'V525-H (ATTO 425)',
    CH_S:    'B525-H (AF488)',
}

# Paths (relative to this scripts/ folder, so they work on any machine)
MANUSCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PROJECT_ROOT   = os.path.join(MANUSCRIPT_DIR, 'Data')
FIGURE_DIR     = os.path.join(MANUSCRIPT_DIR, 'figures_rerun')


# ── Plot style ───────────────────────────────────────────────────────────────
def setup_plot_style():
    """Apply consistent publication-quality matplotlib style."""
    plt.rcParams.update({
        'figure.dpi':        150,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'axes.titlesize':    11,
        'axes.labelsize':    10,
        'axes.titleweight':  'bold',
        'legend.fontsize':   8,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype':      42,  # embed TrueType for editable PDFs
        'ps.fonttype':       42,
    })


# ── FCS loading ──────────────────────────────────────────────────────────────
# Files acquired above this event rate are excluded from every analysis
# (set to None to disable).
MAX_EVENTS_PER_SEC = 7000


class ExcludedFCS(FileNotFoundError):
    """Raised for FCS files excluded by QC; subclasses FileNotFoundError so
    callers that skip missing files skip these too."""


_excluded_reported = set()


def load_fcs(folder, fname):
    """Parse an FCS file and return its DataFrame.

    Raises FileNotFoundError if missing, and ExcludedFCS if the file was acquired
    above MAX_EVENTS_PER_SEC.
    """
    fpath = os.path.join(PROJECT_ROOT, folder, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"FCS file not found: {fpath}")
    _, df = parse(fpath, reformat_meta=True)
    if MAX_EVENTS_PER_SEC is not None:
        eps = len(df) / acq_time(df)
        if eps > MAX_EVENTS_PER_SEC:
            if fpath not in _excluded_reported:
                _excluded_reported.add(fpath)
                print(f"  QC: excluded {folder}/{fname} ({eps:,.0f} events/s > {MAX_EVENTS_PER_SEC:,})")
            raise ExcludedFCS(f"{fpath} excluded: {eps:,.0f} events/s")
    return df


def load_shervin5_25():
    """Load all FCS files from Shervin5_25 (3 replicates × 4 coatings × {4 exts + NoS}).
    Returns dict keyed by (ext, coat, ab_or_nos, rep) or 'Buffer'.
    """
    dfs = {}
    dfs['Buffer'] = load_fcs('Shervin5_25', 'Buffer.fcs')
    for coat in COATINGS:
        for rep in REPLICATES:
            try:
                dfs[('N', coat, 'NoS', rep)] = load_fcs(
                    'Shervin5_25', f'N_{coat}_NoS_{rep}.fcs')
            except FileNotFoundError:
                pass
    for ext in EXTENSIONS:
        for coat in COATINGS:
            for rep in REPLICATES:
                try:
                    dfs[(ext, coat, 'S', rep)] = load_fcs(
                        'Shervin5_25', f'{ext}_{coat}_S_{rep}.fcs')
                except FileNotFoundError:
                    pass
    return dfs


# ── Gating ───────────────────────────────────────────────────────────────────
# Origami gate, per session and coating:
#   SS >= SS_TRIGGER            scatter-triggered events only. The instrument also
#                               triggered on B525, so events below the scatter
#                               trigger are free probe (or noise at the B525 trigger).
#   SS <= ss_high               aggregate limit = SS_HIGH_PCT of the coating's pooled
#                               no-probe control (SS >= SS_TRIGGER). P95, not P99:
#                               the top 1% is dominated by aggregates.
#   V525c <= v525c_high         bright-outlier limit on spillover-compensated V525
#                               (V525 - k * B525), = V525C_HIGH_PCT of pooled NoCoating
#                               no-probe (SS >= SS_TRIGGER). Compensation keeps
#                               probe-bound origami (AF488 bleeds into V525) in the gate.
#   no V525 lower limit         background is removed by buffer subtraction instead.
SS_TRIGGER      = 200
SS_HIGH_PCT     = 95
V525C_HIGH_PCT  = 99

# Probe-only controls used to measure AF488 -> V525 spillover, keyed by no-probe tag
PROBE_ONLY = {
    'NoS':  ('probe_only_controls', 'Streptavidin_Only__Shervin5_03.fcs'),
    'NoAB': ('probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs'),
}
_SPILL_EDGES = [0, 100, 150, 250, 400, 700, 1200, 2500, 10000]
_k_cache = {}


def spillover_k(ctl):
    """AF488 -> V525 spillover k for the probe matching no-probe tag `ctl`.

    Slope of median V525-H vs median B525-H across B525 bins of the probe-only
    control (no ATTO present), using bins with > 30 events.
    """
    if ctl not in _k_cache:
        d = load_fcs(*PROBE_ONLY[ctl])
        xs, ys = [], []
        for lo, hi in zip(_SPILL_EDGES[:-1], _SPILL_EDGES[1:]):
            m = (d[CH_S] >= lo) & (d[CH_S] < hi)
            if m.sum() > 30:
                xs.append(d[CH_S][m].median()); ys.append(d[CH_ATTO][m].median())
        _k_cache[ctl] = max(float(np.polyfit(xs, ys, 1)[0]), 0.0)
    return _k_cache[ctl]


def v525c(df, k):
    """Spillover-compensated V525-H."""
    return df[CH_ATTO] - k * df[CH_S]


def compute_session_gates(noprobe, k):
    """Gates for one session from its no-probe controls.

    noprobe: {coating: [DataFrames of that coating's no-probe replicates]}
    Returns {coating: {'ss_low', 'ss_high', 'v525c_high', 'k'}}.
    """
    trig = lambda df: df[df[CH_SS] >= SS_TRIGGER]
    nc = pd.concat([trig(d) for d in noprobe['NoCoating']])
    v_high = float(np.percentile(v525c(nc, k), V525C_HIGH_PCT))
    gates = {}
    for coat, dlist in noprobe.items():
        if not dlist:
            continue
        ss = np.concatenate([trig(d)[CH_SS].values for d in dlist])
        gates[coat] = {'ss_low': SS_TRIGGER, 'ss_high': float(np.percentile(ss, SS_HIGH_PCT)),
                       'v525c_high': v_high, 'k': k}
    return gates


def compute_monomer_gates(dfs, ctl='NoS', buffer_key='Buffer'):
    """Origami gates for the session in `dfs` (keys (ext, coat, tag, rep)).

    Returned per (coating, replicate) so callers can use apply_gate(df, coat, rep, gates);
    the gate depends on the coating only.
    """
    noprobe = {coat: [dfs[('N', coat, ctl, rep)] for rep in REPLICATES
                      if ('N', coat, ctl, rep) in dfs] for coat in COATINGS}
    sg = compute_session_gates(noprobe, spillover_k(ctl))
    return {(coat, rep): sg[coat] for coat in sg for rep in REPLICATES}


def gate_events(df, g):
    """Keep events inside gate g (SS window + compensated-V525 upper limit)."""
    return df[(df[CH_SS] >= g['ss_low']) & (df[CH_SS] <= g['ss_high']) &
              (v525c(df, g['k']) <= g['v525c_high'])]


def apply_gate(df, coat, rep, gates):
    """Apply the gate for (coating, replicate) to a DataFrame."""
    return gate_events(df, gates[(coat, rep)])


def describe_gate(g):
    return (f"SS [{g['ss_low']:.0f}, {g['ss_high']:.0f}], "
            f"V525-{g['k']:.2f}·B525 ≤ {g['v525c_high']:.0f}")


# ── Buffer-background subtraction ────────────────────────────────────────────
# The buffer is scaled by (sample acquisition time / buffer acquisition time),
# assuming equal flow rate, and its events are subtracted from the sample.
# Pass the *gated* sample and buffer values plus the acquisition times of the
# *ungated* files.
def acq_time(df):
    """Acquisition time from the Time channel (max - min)."""
    return float(df['Time'].max() - df['Time'].min())


def bs_hist(s_vals, s_time, b_vals, b_time, bins=BS_BINS):
    """Buffer-subtracted histogram counts (negative bins clipped to 0)."""
    s = np.histogram(np.clip(s_vals, bins[0], None), bins)[0].astype(float)
    b = np.histogram(np.clip(b_vals, bins[0], None), bins)[0] * (s_time / b_time)
    return np.clip(s - b, 0, None)


def bs_percentile(s_vals, s_time, b_vals, b_time, q, bins=BS_BINS):
    """Percentile q (0-100) of the buffer-subtracted distribution (log-interpolated)."""
    counts = bs_hist(s_vals, s_time, b_vals, b_time, bins)
    cdf = np.cumsum(counts)
    if cdf[-1] <= 0:
        return np.nan
    target = cdf[-1] * q / 100
    i = min(np.searchsorted(cdf, target), len(counts) - 1)
    prev = cdf[i - 1] if i > 0 else 0
    frac = (target - prev) / counts[i] if counts[i] > 0 else 0
    lo, hi = np.log10(bins[i]), np.log10(bins[i + 1])
    return float(10 ** (lo + frac * (hi - lo)))


def bs_mean_std(s_vals, s_time, b_vals, b_time, bins=BS_BINS):
    """Mean and SD of the buffer-subtracted distribution (geometric bin centres)."""
    counts = bs_hist(s_vals, s_time, b_vals, b_time, bins)
    if counts.sum() <= 0:
        return np.nan, np.nan
    x = np.sqrt(bins[:-1] * bins[1:])
    mean = np.average(x, weights=counts)
    return float(mean), float(np.sqrt(np.average((x - mean) ** 2, weights=counts)))


def bs_counts(s_vals, s_time, b_vals, b_time, thr=None):
    """Net (buffer-subtracted) event count, and net count above `thr` if given."""
    scale = s_time / b_time
    net = len(s_vals) - len(b_vals) * scale
    if thr is None:
        return float(net)
    above = (s_vals > thr).sum() - (b_vals > thr).sum() * scale
    return float(net), float(max(above, 0))


def bs_pct_above(s_vals, s_time, b_vals, b_time, thr):
    """% of net (buffer-subtracted) events above threshold."""
    net, above = bs_counts(s_vals, s_time, b_vals, b_time, thr)
    return 100 * above / net if net > 0 else np.nan


def bs_density(s_vals, s_time, b_vals, b_time, lo=1, hi=1e4, per_decade=50, smooth=2.0):
    """Smoothed buffer-subtracted density on a log10 axis (replaces a KDE).

    Returns (x, density) with density normalised per log10 unit.
    """
    bins = np.logspace(np.log10(lo), np.log10(hi), int(per_decade * np.log10(hi / lo)) + 1)
    counts = bs_hist(s_vals, s_time, b_vals, b_time, bins)
    if smooth:
        k = np.arange(-int(4 * smooth), int(4 * smooth) + 1)
        kern = np.exp(-0.5 * (k / smooth) ** 2)
        counts = np.convolve(counts, kern / kern.sum(), mode='same')
    total = counts.sum()
    dens = counts / total / (1 / per_decade) if total > 0 else counts
    return np.sqrt(bins[:-1] * bins[1:]), dens


def compute_thresholds(dfs, gates, channel=CH_S, percentile=99, ctl='NoS', buffer_key='Buffer'):
    """Per-(coating, replicate) B525 threshold = P99 of the buffer-subtracted,
    V525-gated no-probe control."""
    buf = dfs[buffer_key]
    thrs = {}
    for coat in COATINGS:
        for rep in REPLICATES:
            key = ('N', coat, ctl, rep)
            if key not in dfs:
                continue
            mono = apply_gate(dfs[key], coat, rep, gates)
            buf_g = apply_gate(buf, coat, rep, gates)
            if len(mono) > 0:
                thrs[(coat, rep)] = bs_percentile(
                    mono[channel].values, acq_time(dfs[key]),
                    buf_g[channel].values, acq_time(buf), percentile)
    return thrs


# ── Output helpers ───────────────────────────────────────────────────────────
def savefig(fig, basename):
    """Save a figure to paper/figures/ as both PNG (300 DPI) and PDF."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    png = os.path.join(FIGURE_DIR, basename + '.png')
    pdf = os.path.join(FIGURE_DIR, basename + '.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")
