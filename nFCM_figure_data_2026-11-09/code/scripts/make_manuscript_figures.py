"""Manuscript figures (SVG, 600 dpi for rasterized layers) with their data points.

Every figure is computed the same way regardless of outcome: titles are neutral,
directions are reported from the data, and statistics are computed rather than
assumed. Both acquisition sessions are shown wherever the data exist:
  streptavidin session  Data/Shervin5_25  (no-probe tag NoS, probe tag S)
  antibody session      Data/Shervin4_18  (no-probe tag NoAB, probe tag AB)
Files above MAX_EVENTS_PER_SEC are excluded by shared_utils.load_fcs.

Figures (numbering follows the manuscript plan)
  fig00  V525-H separation between uncoated origami and buffer (distributions,
         medians, AUC)                                  [scatter-triggered events]
  fig01  Events per second, uncoated origami vs buffer  [all events and scatter-triggered]
  fig02  Median VSSC-1-H, uncoated origami vs buffer    [scatter-triggered events]
  fig03  VSSC-1-H distribution relative to the scatter trigger, and V525-H histogram
  fig06  Silica: buffer-subtracted scatter vs V525-H maps, median V525-H vs scatter,
         and the high-scatter / low-V525 fraction
  fig07  Median VSSC-1-H per coating (buffer-subtracted, scatter-triggered)
  fig08  Median B525-H on origami without vs with antibody (origami gate, buffer-subtracted)
  fig09  Median B525-H on origami without vs with streptavidin (same)
  fig10  Free antibody: events per second vs its buffer, and scatter vs B525-H maps
  fig11  Free streptavidin: same
  fig12  Fluorescence-triggered events below the scatter trigger in antibody samples
  fig13  Change from no biotin vs biotin extension, antibody (M1, M2, % bound)
  fig14  Change from no biotin vs biotin extension, streptavidin (M1, M2)
  fig15  Events classified as origami-antibody conjugates: no-biotin origami + antibody
         vs origami only (no antibody) vs antibody only, per coating
  fig16  Non-specific probe association by coating (no-biotin origami), streptavidin and
         antibody: median shift and % above no-probe P99; Silica vs other coatings
  fig16b Same, excluding high-scatter / low-V525 events, with probe-only controls and
         the rate of events above threshold
  fig17  Non-specific adsorption summary (median shift vs 0, free-probe clump share), both probes
  fig18  Scatter-triggered event rate: PLL-PEG-coated origami vs PLL-PEG only (trigger settings checked)
  fig19  Scatter-triggered event rate: silica-coated origami vs silica chemicals only and uncoated origami
         (trigger settings checked)
  fig20  Silica side scatter of all events and of fluorescence-confident origami (background threshold at
         matched side scatter), with expected false positives
  fig21  Same as fig20 for PLL-PEG (background = buffer + PLL-PEG only)
  fig06, fig07, fig19 and fig20 are written to figures_svg/silica_side_scatter/
  fig22  2x2 quadrants (high/low side scatter x high/low ATTO 425) per day for uncoated, PLL-PEG,
         silica origami and the coating-only controls (in both side-scatter folders)
  fig23  Side scatter of particles with ATTO 425 above buffer but below aggregation, per coating and day,
         with the expected contribution of coating-derived particles (in both side-scatter folders)
  fig24  Antibody solution effect on B525-H: non-origami particles in antibody-only vs buffer, quantile
         shifts, and correlation of the no-biotin shift with free-antibody event rate
  fig25  Antibody session: biotin-extension signal vs number/brightness of fluorescence-triggered events,
         ATTO among them, dimmest-origami brightening, and acquisition-order/session-time confound
  fig26  Free-probe brightness at very low side scatter (no origami) vs the origami signal, per sample,
         antibody and streptavidin sessions
  fig27  Percent conjugation vs free-antibody brightness, and a brightness-calibrated conjugation threshold
         (median B525-H of the same sample's low-scatter free antibody)
  fig27b fig27 repeated with the fig13 corrections (unbound tail, clump correction, unmixing below the
         scatter trigger) at the calibrated threshold, with a step-by-step breakdown
  fig27c Threshold = P99 of no-biotin + antibody (per replicate), adjusted per extension by its own
         replicate's P25: shifted (+ difference) and scaled (x ratio)
  fig27d Mixture model: stretched N+AB + (N+AB + one antibody), gain from the sample's free antibody
  fig07, fig18 and fig21 are written to figures_svg/pll_peg_side_scatter/ (fig07 copied)

Outputs: figures_svg/figNN_<name>.svg, figures_svg/figNN_<name>*.csv,
         figures_svg/stats_summary.csv

Usage: python make_manuscript_figures.py [fig00 fig07 ...]   (default: all)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_1samp, ttest_rel, ttest_ind, f_oneway, linregress, spearmanr

from shared_utils import (
    CH_SS, CH_ATTO, CH_S, COATINGS, EXTENSIONS, REPLICATES, COAT_COLORS, MANUSCRIPT_DIR, PROJECT_ROOT, SS_TRIGGER,
    load_fcs, acq_time, compute_monomer_gates, apply_gate, bs_percentile, bs_counts,
)

OUT = os.path.join(MANUSCRIPT_DIR, 'figures_svg')
os.makedirs(OUT, exist_ok=True)

sns.set_theme(context='talk', style='ticks')
plt.rcParams.update({'savefig.dpi': 600, 'svg.fonttype': 'none', 'figure.constrained_layout.use': True,
                     'axes.spines.top': False, 'axes.spines.right': False})

COAT_LBL = {'NoCoating': 'Uncoated', 'PLL': 'PLL-PEG', 'Silica': 'Silica'}
SESSIONS = {
    'Streptavidin session (05/25)': dict(folder='Shervin5_25', noprobe='NoS', probe='S',
                                         probe_name='streptavidin'),
    'Antibody session (04/18)':     dict(folder='Shervin4_18', noprobe='NoAB', probe='AB',
                                         probe_name='anti-biotin IgG'),
}
BUF_COL, ORI_COL = '0.45', COAT_COLORS['NoCoating']
STATS = []
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers
def L(folder, fname):
    try:
        return load_fcs(folder, fname)
    except FileNotFoundError:
        return None


def scat(df):
    """Scatter-triggered events (recorded the same way with or without probe)."""
    return df[df[CH_SS] >= SS_TRIGGER]


def rate(n, t):
    return n / t, np.sqrt(n) / t          # events/s and Poisson SE


def acq_trigger_settings(folder, fname):
    """Trigger settings recorded by the instrument (NANOFLOWINFO JSON in the FCS ANALYSIS segment)."""
    import re, json
    from fcsparser import parse
    path = os.path.join(PROJECT_ROOT, folder, fname)
    meta, _ = parse(path, reformat_meta=True)
    raw = open(path, 'rb').read()[int(meta['$BEGINANALYSIS']):int(meta['$ENDANALYSIS']) + 1].decode('latin1')
    info = json.loads(bytes.fromhex(re.search(r'NANOFLOWINFO\n([0-9A-Fa-f]+)', raw).group(1)).decode('utf8'))
    a = info['Acquisition']
    ch = a['ExtraProperties']['Channels']
    ch = json.loads(ch) if isinstance(ch, str) else ch
    return {'primary_channel': a['PrimaryChannel'], 'primary_threshold': a['PrimaryThreshold'],
            'secondary_channel': a['SecondaryChannel'], 'secondary_threshold': a['SecondaryThreshold'],
            'trigger_logic': a['ThresholdTrigLogic'], 'trigger_type': a['ThresholdTrigType'],
            'trigger_mode': a['ThresholdTrigMode'],
            'channel_trig_levels': ';'.join(f"{c['ChannelName']}={c.get('Trig')}" for c in ch)}


def json_key(d):
    return ', '.join(f'{k}={v}' for k, v in d.items())


def b525_trigger(df):
    """Empirical B525-H trigger level: 0.5th percentile of B525 among SS < 200 events."""
    lo = df[df[CH_SS] < SS_TRIGGER]
    return float(np.percentile(lo[CH_S], 0.5)) if len(lo) >= 5 else np.nan


def log_density(v, bins):
    v = v[(v >= bins[0]) & (v <= bins[-1])]
    h, _ = np.histogram(v, bins)
    return h / max(h.sum(), 1) / np.diff(np.log10(bins))


def auc(a, b, n=50000):
    """P(random a > random b), from the Mann-Whitney U statistic."""
    a = rng.choice(a, min(n, len(a)), replace=False); b = rng.choice(b, min(n, len(b)), replace=False)
    return mannwhitneyu(a, b).statistic / (len(a) * len(b))


def stat(fig, test, comparison, value, p=np.nan, n=np.nan, note=''):
    STATS.append(dict(figure=fig, test=test, comparison=comparison, value=value, p=p, n=n, note=note))


SILICA_DIR = 'silica_side_scatter'   # figures supporting the silica side-scatter paragraph
PLL_DIR = 'pll_peg_side_scatter'     # figures supporting the PLL-PEG side-scatter paragraph


def save(fig, name, tables, subdir=None):
    out = os.path.join(OUT, subdir) if subdir else OUT
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f'{name}.svg'), dpi=600, bbox_inches='tight')
    if os.environ.get('FIG_PREVIEW_DIR'):      # optional low-res PNG previews for checking layout
        fig.savefig(os.path.join(os.environ['FIG_PREVIEW_DIR'], f'{name}.png'), dpi=60, bbox_inches='tight')
    plt.close(fig)
    for suffix, df in tables.items():
        path = os.path.join(out, f'{name}{suffix}.csv')
        try:
            df.to_csv(path, index=False)
        except PermissionError:
            print(f'  WARNING: could not write {os.path.basename(path)} (file open elsewhere?); not updated')
    print(f'  wrote {name}.svg (+{len(tables)} csv)')


def write_raw(name, suffix, cols):
    """Unbinned per-event values behind a 1D histogram: one column per population
    (columns padded with blanks to the longest population)."""
    df = pd.DataFrame({k: pd.Series(np.asarray(v, dtype=float)) for k, v in cols.items()})
    path = os.path.join(OUT, f'{name}{suffix}.csv')
    try:
        df.to_csv(path, index=False, float_format='%.6g')
        print(f'  wrote {os.path.basename(path)} ({len(cols)} columns, up to {len(df):,} events)')
    except PermissionError:
        print(f'  WARNING: could not write {os.path.basename(path)} (file open elsewhere?); not updated')


def uncoated_noprobe(s):
    return [(r, d) for r in REPLICATES if (d := L(s['folder'], f"N_NoCoating_{s['noprobe']}_{r}.fcs")) is not None]


def session_dfs(s, exts=('N',), probe=True):
    dfs = {'Buffer': L(s['folder'], 'Buffer.fcs')}
    for c in COATINGS:
        for r in REPLICATES:
            if (d := L(s['folder'], f"N_{c}_{s['noprobe']}_{r}.fcs")) is not None:
                dfs[('N', c, s['noprobe'], r)] = d
            if probe:
                for e in exts:
                    if (d := L(s['folder'], f"{e}_{c}_{s['probe']}_{r}.fcs")) is not None:
                        dfs[(e, c, s['probe'], r)] = d
    return dfs


def rep_points(ax, x, vals, color='black', s=40):
    ax.scatter(x + rng.uniform(-0.06, 0.06, len(vals)), vals, s=s, color=color,
               edgecolor='white', linewidth=0.6, zorder=5)


# ---------------------------------------------------------------- fig00
def fig00():
    bins = np.logspace(0, 3.5, 141)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    summ, hist, raw = [], [], {}
    for ax, (sname, s) in zip(axes, SESSIONS.items()):
        buf = L(s['folder'], 'Buffer.fcs'); vb = scat(buf)[CH_ATTO].values
        raw[f'{sname} | Buffer | V525-H'] = vb
        vb = vb[vb > 0]
        x = np.sqrt(bins[:-1] * bins[1:])
        hb = log_density(vb, bins)
        ax.fill_between(x, hb, step='mid', color=BUF_COL, alpha=0.35, label='Buffer')
        ax.axvline(np.median(vb), color=BUF_COL, ls='--', lw=2)
        summ.append(dict(session=sname, sample='Buffer', n_events=len(vb), median_V525=np.median(vb), auc_vs_buffer=0.5))
        hist += [dict(session=sname, sample='Buffer', bin_center=xc, density=h) for xc, h in zip(x, hb)]
        aucs = []
        for i, (r, d) in enumerate(uncoated_noprobe(s)):
            vo = scat(d)[CH_ATTO].values
            raw[f'{sname} | Uncoated {r} | V525-H'] = vo
            vo = vo[vo > 0]
            ho = log_density(vo, bins); a = auc(vo, vb); aucs.append(a)
            ax.plot(x, ho, color=sns.light_palette(ORI_COL, 5)[i + 2], lw=2.5,
                    label=f'Uncoated origami {r} (AUC {a:.2f})')
            ax.axvline(np.median(vo), color=sns.light_palette(ORI_COL, 5)[i + 2], ls='--', lw=2)
            summ.append(dict(session=sname, sample=f'Uncoated {r}', n_events=len(vo),
                             median_V525=np.median(vo), auc_vs_buffer=a))
            hist += [dict(session=sname, sample=f'Uncoated {r}', bin_center=xc, density=h) for xc, h in zip(x, ho)]
        stat('fig00', 'AUC origami vs buffer (V525-H, scatter-triggered)', sname, np.mean(aucs),
             n=len(aucs), note='0.5 = no separation, 1 = complete separation')
        ax.set_xscale('log'); ax.set_xlabel('V525-H (a.u.)'); ax.set_title(sname, fontsize=16)
        ax.legend(fontsize=11, frameon=False, loc='upper right')
    axes[0].set_ylabel('Density (per log$_{10}$ unit)')
    fig.suptitle('V525-H of uncoated origami vs buffer (scatter-triggered events; dashed = medians)', y=1.02)
    save(fig, 'fig00_v525_origami_vs_buffer', {'_summary': pd.DataFrame(summ), '_histograms': pd.DataFrame(hist)})
    write_raw('fig00_v525_origami_vs_buffer', '_raw_V525_scatter_triggered', raw)
    # Same data with replicates A-C pooled into one column, one file per session
    for sname, tag in [('Antibody session (04/18)', 'antibody'), ('Streptavidin session (05/25)', 'streptavidin')]:
        reps = [v for k, v in raw.items() if k.startswith(sname) and '| Uncoated ' in k]
        write_raw('fig00_v525_origami_vs_buffer', f'_raw_V525_scatter_triggered_pooled_{tag}',
                  {f'{sname} | Buffer | V525-H': raw[f'{sname} | Buffer | V525-H'],
                   f'{sname} | Uncoated A-C pooled | V525-H': np.concatenate(reps)})


# ---------------------------------------------------------------- fig01
def fig01():
    rows = []
    for sname, s in SESSIONS.items():
        buf = L(s['folder'], 'Buffer.fcs'); tb = acq_time(buf)
        for kind, sel in [('All events', lambda d: d), ('Scatter-triggered (SS ≥ 200)', scat)]:
            rb, eb = rate(len(sel(buf)), tb)
            rows.append(dict(session=sname, events=kind, sample='Buffer', replicate='', eps=rb, poisson_se=eb))
            vals = []
            for r, d in uncoated_noprobe(s):
                ro, eo = rate(len(sel(d)), acq_time(d)); vals.append(ro)
                rows.append(dict(session=sname, events=kind, sample='Uncoated origami', replicate=r, eps=ro, poisson_se=eo))
            t = ttest_1samp(np.log(vals), np.log(rb))
            stat('fig01', 'one-sample t-test, log(events/s) origami reps vs buffer', f'{sname}, {kind}',
                 np.mean(vals) / rb, t.pvalue, len(vals), 'value = mean fold change vs buffer')
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, kind in zip(axes, df.events.unique()):
        d = df[df.events == kind]
        for xi, sname in enumerate(SESSIONS):
            q = d[d.session == sname]
            b = q[q['sample'] == 'Buffer']; o = q[q['sample'] != 'Buffer']
            ax.bar(xi - 0.2, b.eps.iloc[0], 0.38, yerr=b.poisson_se.iloc[0], color=BUF_COL, alpha=0.6,
                   edgecolor='black', label='Buffer' if xi == 0 else None)
            ax.bar(xi + 0.2, o.eps.mean(), 0.38, yerr=o.eps.sem(), color=ORI_COL, edgecolor='black',
                   label='Uncoated origami (mean ± SEM)' if xi == 0 else None, capsize=6)
            rep_points(ax, xi + 0.2, o.eps.values)
        ax.set_xticks(range(len(SESSIONS))); ax.set_xticklabels([k.split(' (')[0].replace(' session', '') + '\nsession' for k in SESSIONS])
        ax.set_ylabel('Events per second'); ax.set_title(kind, fontsize=16)
    axes[0].legend(fontsize=12, frameon=False)
    fig.suptitle('Event rate: uncoated origami (no probe) vs buffer', y=1.02)
    save(fig, 'fig01_event_rate_origami_vs_buffer', {'': df})


# ---------------------------------------------------------------- fig02
def fig02():
    rows = []
    for sname, s in SESSIONS.items():
        buf = L(s['folder'], 'Buffer.fcs'); bs = scat(buf); tb = acq_time(buf)
        mb = float(np.median(bs[CH_SS]))
        rows.append(dict(session=sname, sample='Buffer', replicate='', median_SS=mb, median_SS_buffer_subtracted=np.nan))
        vals = []
        for r, d in uncoated_noprobe(s):
            ds = scat(d); m = float(np.median(ds[CH_SS])); vals.append(m)
            mbs = bs_percentile(ds[CH_SS].values, acq_time(d), bs[CH_SS].values, tb, 50)
            rows.append(dict(session=sname, sample='Uncoated origami', replicate=r, median_SS=m,
                             median_SS_buffer_subtracted=mbs))
        t = ttest_1samp(vals, mb)
        stat('fig02', 'one-sample t-test, median SS origami reps vs buffer', sname,
             np.mean(vals) - mb, t.pvalue, len(vals), 'value = mean difference (origami - buffer), a.u.')
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 6))
    for xi, sname in enumerate(SESSIONS):
        q = df[df.session == sname]; b = q[q['sample'] == 'Buffer']; o = q[q['sample'] != 'Buffer']
        ax.bar(xi - 0.2, b.median_SS.iloc[0], 0.38, color=BUF_COL, alpha=0.6, edgecolor='black',
               label='Buffer' if xi == 0 else None)
        ax.bar(xi + 0.2, o.median_SS.mean(), 0.38, yerr=o.median_SS.sem(), capsize=6, color=ORI_COL,
               edgecolor='black', label='Uncoated origami (mean ± SEM)' if xi == 0 else None)
        rep_points(ax, xi + 0.2, o.median_SS.values)
    ax.set_xticks(range(len(SESSIONS))); ax.set_xticklabels([k.split(' (')[0].replace(' session', '') + '\nsession' for k in SESSIONS])
    ax.set_ylabel('Median VSSC-1-H (a.u.)')
    ax.legend(fontsize=12, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.set_title('Median violet side scatter (scatter-triggered events)', fontsize=16)
    save(fig, 'fig02_median_ss_origami_vs_buffer', {'': df})


# ---------------------------------------------------------------- fig03
def fig03():
    ss_bins = np.logspace(1, 4.5, 141); v_bins = np.logspace(0, 3.5, 141)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    hist, summ = [], []
    raw_ss, raw_v, raw_vt = {}, {}, {}
    for col, (sname, s) in enumerate(SESSIONS.items()):
        buf = L(s['folder'], 'Buffer.fcs')
        for lbl, d in [('Buffer', buf)] + [(f'Uncoated {r}', d) for r, d in uncoated_noprobe(s)]:
            raw_ss[f'{sname} | {lbl} | VSSC-1-H'] = d[CH_SS].values
            raw_v[f'{sname} | {lbl} | V525-H'] = d[CH_ATTO].values
            raw_vt[f'{sname} | {lbl} | V525-H'] = scat(d)[CH_ATTO].values
        ax_ss, ax_v = axes[0, col], axes[1, col]
        for label, dfs_, color in [('Buffer', [buf], BUF_COL),
                                   ('Uncoated origami (pooled A-C)', [d for _, d in uncoated_noprobe(s)], ORI_COL)]:
            pool = pd.concat(dfs_)
            x = np.sqrt(ss_bins[:-1] * ss_bins[1:]); h = log_density(pool[CH_SS].clip(lower=ss_bins[0]).values, ss_bins)
            ax_ss.step(x, h, where='mid', color=color, lw=2.5, label=label)
            hist += [dict(session=sname, sample=label, channel='VSSC-1-H', bin_center=a, density=b) for a, b in zip(x, h)]
            ps = scat(pool)
            xv = np.sqrt(v_bins[:-1] * v_bins[1:])
            for sub, lbl, ls in [(pool, 'all events', '-'), (ps, 'scatter-triggered', '--')]:
                v = sub[CH_ATTO].values; hv = log_density(v[v > 0], v_bins)
                ax_v.step(xv, hv, where='mid', color=color, lw=2.2, ls=ls, label=f'{label}, {lbl}')
                hist += [dict(session=sname, sample=f'{label}, {lbl}', channel='V525-H', bin_center=a, density=b)
                         for a, b in zip(xv, hv)]
            n = len(pool); n_trig = len(ps)
            summ.append(dict(session=sname, sample=label, n_events=n,
                             pct_below_SS_trigger=(1 - n_trig / n) * 100,
                             median_SS_scatter_triggered=float(np.median(ps[CH_SS])) if n_trig else np.nan,
                             pct_of_triggered_within_200_250=float(((ps[CH_SS] < 250).mean()) * 100) if n_trig else np.nan,
                             median_V525_all=float(pool[CH_ATTO].median()),
                             median_V525_scatter_triggered=float(ps[CH_ATTO].median()) if n_trig else np.nan))
        ax_ss.axvline(SS_TRIGGER, color='red', ls='--', lw=2, label='Scatter trigger (200)')
        ax_ss.set_xscale('log'); ax_ss.set_xlabel('VSSC-1-H (a.u.)'); ax_ss.set_title(sname, fontsize=16)
        ax_ss.legend(fontsize=11, frameon=False)
        ax_v.set_xscale('log'); ax_v.set_xlabel('V525-H (a.u.)'); ax_v.legend(fontsize=11, frameon=False, loc='upper left')
    axes[0, 0].set_ylabel('Density (per log$_{10}$ unit)'); axes[1, 0].set_ylabel('Density (per log$_{10}$ unit)')
    fig.suptitle('Uncoated origami (no probe): side scatter relative to the scatter trigger, and V525-H', y=1.01)
    save(fig, 'fig03_ss_distribution_and_v525', {'_summary': pd.DataFrame(summ), '_histograms': pd.DataFrame(hist)})
    write_raw('fig03_ss_distribution_and_v525', '_raw_SS_all_events', raw_ss)
    write_raw('fig03_ss_distribution_and_v525', '_raw_V525_all_events', raw_v)
    write_raw('fig03_ss_distribution_and_v525', '_raw_V525_scatter_triggered', raw_vt)
    # Replicates A-C pooled into one column, one file per session (all three histograms)
    for sname, tag in [('Antibody session (04/18)', 'antibody'), ('Streptavidin session (05/25)', 'streptavidin')]:
        cols = {}
        for raw_d, what in [(raw_ss, 'VSSC-1-H, all events'), (raw_v, 'V525-H, all events'),
                            (raw_vt, 'V525-H, scatter-triggered')]:
            cols[f'{sname} | Buffer | {what}'] = next(v for k, v in raw_d.items()
                                                       if k.startswith(sname) and '| Buffer |' in k)
            cols[f'{sname} | Uncoated A-C pooled | {what}'] = np.concatenate(
                [v for k, v in raw_d.items() if k.startswith(sname) and '| Uncoated ' in k])
        write_raw('fig03_ss_distribution_and_v525', f'_raw_pooled_{tag}', cols)


# ---------------------------------------------------------------- fig06
def fig06():
    xb = np.logspace(np.log10(SS_TRIGGER), 5, 61); yb = np.logspace(0.5, 3.5, 61)
    ss_bins = np.logspace(np.log10(SS_TRIGGER), 5, 15)
    chem = L('coatings_controls', 'SilicaChemicalsOnly.fcs')
    pll_only = L('coatings_controls', 'PLL_PEG_NoOrigami.fcs')
    fig, axes = plt.subplots(2, 3, figsize=(27, 14))
    maps, curves, summ = [], [], []
    for row, (sname, s) in enumerate(SESSIONS.items()):
        buf = scat(L(s['folder'], 'Buffer.fcs')); tb = acq_time(L(s['folder'], 'Buffer.fcs'))
        pooled = {}
        for c in ['NoCoating', 'Silica']:
            ds = [d for r in REPLICATES if (d := L(s['folder'], f"N_{c}_{s['noprobe']}_{r}.fcs")) is not None]
            pooled[c] = (pd.concat([scat(d) for d in ds]), sum(acq_time(d) for d in ds))
        uo = pooled['NoCoating'][0]
        ss_hi = float(np.percentile(uo[CH_SS], 95)); v_lo = float(np.median(uo[CH_ATTO]))
        h2 = lambda d: np.histogram2d(d[CH_SS].clip(lower=xb[0]), d[CH_ATTO].clip(lower=yb[0]), [xb, yb])[0]
        for col, c in enumerate(['NoCoating', 'Silica']):
            d, t = pooled[c]
            net = h2(d) / t - h2(buf) / tb
            ax = axes[row, col]
            m = np.where(net > 0, net / net[net > 0].sum(), np.nan)
            pc = ax.pcolormesh(xb, yb, m.T, norm=LogNorm(vmin=1e-5, vmax=np.nanmax(m)), cmap='viridis', rasterized=True)
            ax.axvline(ss_hi, color='white', ls='--', lw=1.5); ax.axhline(v_lo, color='white', ls='--', lw=1.5)
            q = lambda z: ((z[CH_SS] > ss_hi) & (z[CH_ATTO] < v_lo)).sum()
            frac = (q(d) / t - q(buf) / tb) / (len(d) / t - len(buf) / tb) * 100
            summ.append(dict(session=sname, sample=COAT_LBL[c], ss_high_cut=ss_hi, v525_low_cut=v_lo,
                             pct_net_events_high_SS_low_V525=frac, buffer_subtracted=True))
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('VSSC-1-H (a.u.)')
            ax.set_ylabel('V525-H (a.u.)')
            ax.set_title(f'{COAT_LBL[c]} (buffer-subtracted)\n{frac:.1f}% of net events: SS > {ss_hi:.0f}, V525 < {v_lo:.0f}',
                         fontsize=14)
            fig.colorbar(pc, ax=ax, label='Fraction of net events')
            ii, jj = np.nonzero(np.isfinite(m))
            maps += [dict(session=sname, sample=COAT_LBL[c], ss_bin_lo=xb[i], ss_bin_hi=xb[i + 1],
                          v525_bin_lo=yb[j], v525_bin_hi=yb[j + 1], net_events_per_s=net[i, j]) for i, j in zip(ii, jj)]
        # median V525 per SS bin
        ax = axes[row, 2]
        xc = np.sqrt(ss_bins[:-1] * ss_bins[1:])
        for c in ['NoCoating', 'Silica']:
            d, t = pooled[c]; med = []
            for a, b in zip(ss_bins[:-1], ss_bins[1:]):
                sv = d[(d[CH_SS] >= a) & (d[CH_SS] < b)][CH_ATTO].values
                bv = buf[(buf[CH_SS] >= a) & (buf[CH_SS] < b)][CH_ATTO].values
                net_n = len(sv) - len(bv) * t / tb
                med.append(bs_percentile(sv[sv > 0], t, bv[bv > 0], tb, 50) if net_n >= 50 else np.nan)
            ax.plot(xc, med, 'o-', color=COAT_COLORS[c], lw=2.5, label=f'{COAT_LBL[c]} origami (buffer-subtracted)')
            curves += [dict(session=sname, sample=COAT_LBL[c], ss_bin_center=a, median_V525=b) for a, b in zip(xc, med)]
        cs = scat(chem); med = [cs[(cs[CH_SS] >= a) & (cs[CH_SS] < b)][CH_ATTO].median()
                                if ((cs[CH_SS] >= a) & (cs[CH_SS] < b)).sum() >= 50 else np.nan
                                for a, b in zip(ss_bins[:-1], ss_bins[1:])]
        ax.plot(xc, med, 's--', color='0.3', lw=2, label='Silica chemicals only (no origami, raw)')
        curves += [dict(session=sname, sample='Silica chemicals only (raw)', ss_bin_center=a, median_V525=b)
                   for a, b in zip(xc, med)]
        binmed = lambda z, nmin: [z[(z[CH_SS] >= a) & (z[CH_SS] < b)][CH_ATTO].median()
                                  if ((z[CH_SS] >= a) & (z[CH_SS] < b)).sum() >= nmin else np.nan
                                  for a, b in zip(ss_bins[:-1], ss_bins[1:])]
        for lbl_, z, style, colr, nmin in [('PLL-PEG only (no origami, raw)', scat(pll_only), '^--', '#2CA02C', 50),
                                           (f'Buffer ({sname.split("(")[-1][:-1]}, raw)', buf, 'v:', BUF_COL, 20)]:
            med = binmed(z, nmin)
            ax.plot(xc, med, style, color=colr, lw=2, label=lbl_)
            curves += [dict(session=sname, sample=lbl_, ss_bin_center=a, median_V525=b) for a, b in zip(xc, med)]
        ax.set_xscale('log'); ax.set_xlabel('VSSC-1-H (a.u.)'); ax.set_ylabel('Median V525-H (a.u.)')
        ax.set_title(sname, fontsize=14); ax.legend(fontsize=10, frameon=False)
    chem_frac = [((scat(chem)[CH_SS] > r['ss_high_cut']) & (scat(chem)[CH_ATTO] < r['v525_low_cut'])).mean() * 100
                 for r in summ if r['sample'] == 'Silica']
    for r, f in zip([r for r in summ if r['sample'] == 'Silica'], chem_frac):
        summ.append(dict(session=r['session'], sample='Silica chemicals only', ss_high_cut=r['ss_high_cut'],
                         v525_low_cut=r['v525_low_cut'], pct_net_events_high_SS_low_V525=f, buffer_subtracted=False))
    for r in summ:
        stat('fig06', '% of net scatter-triggered events with SS > uncoated P95 and V525 < uncoated median',
             f"{r['session']}, {r['sample']}", r['pct_net_events_high_SS_low_V525'],
             note='buffer-subtracted' if r['buffer_subtracted'] else 'raw, no buffer available')
    fig.suptitle('Scatter vs V525-H of no-probe origami (scatter-triggered); dashed lines = uncoated P95 SS and median V525', y=1.01)
    save(fig, 'fig06_silica_scatter_vs_v525', {'_summary': pd.DataFrame(summ), '_maps': pd.DataFrame(maps),
                                               '_median_v525_by_ss': pd.DataFrame(curves)}, subdir=SILICA_DIR)


# ---------------------------------------------------------------- fig07
def fig07():
    rows = []
    for sname, s in SESSIONS.items():
        bufall = L(s['folder'], 'Buffer.fcs'); buf = scat(bufall); tb = acq_time(bufall)
        groups = []
        for c in COATINGS:
            v = []
            for r in REPLICATES:
                d = L(s['folder'], f"N_{c}_{s['noprobe']}_{r}.fcs")
                if d is None:
                    continue
                m = bs_percentile(scat(d)[CH_SS].values, acq_time(d), buf[CH_SS].values, tb, 50)
                v.append(m); rows.append(dict(session=sname, coating=COAT_LBL[c], replicate=r, median_SS_buffer_subtracted=m))
            groups.append(v)
        a = f_oneway(*groups)
        stat('fig07', 'one-way ANOVA, median SS across coatings', sname, a.statistic, a.pvalue,
             sum(len(g) for g in groups))
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, sname in zip(axes, SESSIONS):
        q = df[df.session == sname]
        sns.barplot(data=q, x='coating', y='median_SS_buffer_subtracted', hue='coating', order=[COAT_LBL[c] for c in COATINGS],
                    palette={COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}, errorbar='se', capsize=0.15,
                    edgecolor='black', legend=False, ax=ax)
        sns.stripplot(data=q, x='coating', y='median_SS_buffer_subtracted', order=[COAT_LBL[c] for c in COATINGS],
                      color='black', size=7, ax=ax)
        ax.set_xlabel(''); ax.set_ylabel('Median VSSC-1-H (a.u.)'); ax.set_title(sname, fontsize=16)
    fig.suptitle('Median side scatter of no-probe origami by coating (scatter-triggered, buffer-subtracted)', y=1.02)
    save(fig, 'fig07_median_ss_by_coating', {'': df}, subdir=SILICA_DIR)
    copy_into(SILICA_DIR, PLL_DIR, 'fig07_median_ss_by_coating')


# ---------------------------------------------------------------- fig08 / fig09
def probe_median_fig(sname, figname, connect=True, refs=None):
    """refs: optional [(label, folder, file, color)] reference samples passed through the uncoated origami
    gate and plotted as raw medians (no buffer subtraction) to the right of the origami bars."""
    s = SESSIONS[sname]
    dfs = session_dfs(s)
    gates = compute_monomer_gates(dfs, ctl=s['noprobe'])
    buf = dfs['Buffer']; tb = acq_time(buf)
    rows = []
    for c in COATINGS:
        b = apply_gate(buf, c, 'A', gates)[CH_S].values; b = b[b > 0]
        for r in REPLICATES:
            for tag, lbl in [(s['noprobe'], 'No probe'), (s['probe'], f"With {s['probe_name']}")]:
                d = dfs.get(('N', c, tag, r))
                if d is None:
                    continue
                v = apply_gate(d, c, r, gates)[CH_S].values; v = v[v > 0]
                rows.append(dict(coating=COAT_LBL[c], replicate=r, condition=lbl,
                                 median_B525_buffer_subtracted=bs_percentile(v, acq_time(d), b, tb, 50),
                                 median_B525_raw=float(np.median(v)), n_gated=len(v)))
    df = pd.DataFrame(rows)
    ref_rows = []
    for label, folder, fname, colr in (refs or []):
        d = L(folder, fname)
        v = apply_gate(d, 'NoCoating', 'A', gates)[CH_S].values; v = v[v > 0]
        ref_rows.append(dict(coating=label, replicate='', condition='Reference (raw, uncoated origami gate)',
                             median_B525_buffer_subtracted=np.nan, median_B525_raw=float(np.median(v)),
                             n_gated=len(v), color=colr))
    fig, ax = plt.subplots(figsize=(11 + 2.2 * len(ref_rows), 6.5))
    order = [COAT_LBL[c] for c in COATINGS]; conds = ['No probe', f"With {s['probe_name']}"]
    sns.barplot(data=df, x='coating', y='median_B525_buffer_subtracted', hue='condition', order=order,
                hue_order=conds, palette=['0.75', ORI_COL], errorbar='se', capsize=0.1, edgecolor='black', ax=ax)
    for xi, c in enumerate(order):
        q = df[df.coating == c].pivot(index='replicate', columns='condition', values='median_B525_buffer_subtracted')
        for _, rr in q.iterrows():
            ax.plot([xi - 0.2, xi + 0.2], [rr.get(conds[0]), rr.get(conds[1])], 'o-' if connect else 'o',
                    color='black', ms=6, lw=1)
        if q.shape[1] == 2 and q.dropna().shape[0] >= 2:
            qq = q.dropna(); t = ttest_rel(np.log2(qq[conds[1]]), np.log2(qq[conds[0]]))
            stat(figname, 'paired t-test, log2 median B525 with vs without probe', f'{sname}, {c}',
                 float(np.mean(np.log2(qq[conds[1]] / qq[conds[0]]))), t.pvalue, len(qq),
                 'value = mean log2(with/without)')
    ax.set_xlabel(''); ax.set_ylabel('Median B525-H (a.u.)'); ax.legend(fontsize=12, frameon=False, title='')
    ax.set_title(f'{sname}: N-extension origami, origami gate, buffer-subtracted '
                 f"({'lines' if connect else 'points'} = replicates)", fontsize=14)
    if ref_rows:
        n0 = len(order)
        for k, rr in enumerate(ref_rows):
            ax.bar(n0 + 0.3 + k * 0.7, rr['median_B525_raw'], 0.5, color=rr['color'], edgecolor='black',
                   hatch='//', alpha=0.85)
            ax.text(n0 + 0.3 + k * 0.7, rr['median_B525_raw'] + 2, f"{rr['median_B525_raw']:.0f}", ha='center',
                    va='bottom', fontsize=11)
        ax.axvline(n0 - 0.45, color='0.6', ls=':', lw=1.2)
        ax.set_xticks(list(range(n0)) + [n0 + 0.3 + k * 0.7 for k in range(len(ref_rows))])
        ax.set_xticklabels(order + [rr['coating'].replace(' (', '\n(') for rr in ref_rows], fontsize=11)
        ax.set_xlim(-0.6, n0 + 0.3 + (len(ref_rows) - 1) * 0.7 + 0.5)
        ax.text(n0 + 0.3 + (len(ref_rows) - 1) * 0.35, ax.get_ylim()[1] * 0.97,
                'References: raw medians,\nuncoated origami gate', ha='center', va='top', fontsize=10, color='0.3')
        df = pd.concat([df, pd.DataFrame(ref_rows).drop(columns='color')], ignore_index=True)
    save(fig, figname, {'': df})


def fig08():
    probe_median_fig('Antibody session (04/18)', 'fig08_median_b525_with_without_antibody', connect=False,
                     refs=[('Buffer (04/18)', 'Shervin4_18', 'Buffer.fcs', BUF_COL),
                           ('Buffer (03/13)', 'probe_only_controls', 'Buffer__Shervin03_13_2026.fcs', '0.6'),
                           ('Antibody only (03/13)', 'probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs', '#E6862E')])


def fig09():
    probe_median_fig('Streptavidin session (05/25)', 'fig09_median_b525_with_without_streptavidin', connect=False)


# ---------------------------------------------------------------- fig10 / fig11
def free_probe_fig(probe_file, buf_file, label, figname, fluor_only=None, common_mode='or', extra=None):
    """fluor_only: if set, also plot fluorescence-triggered events (SS < 200 and B525 >= fluor_only);
    'common' uses the higher of the two files' empirical B525 trigger levels.
    common_mode: 'or' counts events with SS >= 200 or B525 >= common trigger; 'and' requires both.
    extra: optional (name, folder, [files]) replicate samples added to the event-rate panel. They are
    added only if their recorded trigger settings match the probe-only and buffer files exactly.
    No buffer subtraction is applied to any event rate."""
    pr = L('probe_only_controls', probe_file); bu = L('probe_only_controls', buf_file)
    trig = {'probe-only': b525_trigger(pr), 'buffer': b525_trigger(bu)}
    common = np.nanmax(list(trig.values()))
    samples = [('Buffer', '', 'probe_only_controls', buf_file, bu),
               (f'{label} only', '', 'probe_only_controls', probe_file, pr)]
    settings = {f: acq_trigger_settings(fo, f) for _, _, fo, f, _ in samples}
    if extra is not None:
        ename, efolder, efiles = extra
        esettings = {f: acq_trigger_settings(efolder, f) for f in efiles}
        ref = settings[probe_file]
        if all(v == ref for v in list(settings.values()) + list(esettings.values())):
            settings.update(esettings)
            samples += [(ename, f.rsplit('_', 1)[-1][:-4], efolder, f, d) for f in efiles
                        if (d := L(efolder, f)) is not None]          # runs above 7,000 events/s are skipped
        else:
            print(f'  {ename}: trigger settings differ from {probe_file}; not added')
    stat(figname, 'recorded trigger settings identical across plotted files', label,
         float(len({json_key(v) for v in settings.values()}) == 1),
         note=json_key(next(iter(settings.values()))))
    rows = []
    for name, rep, folder, fname, d in samples:
        if d is None:
            continue
        t = acq_time(d)
        kinds = [('All recorded events', np.ones(len(d), bool)),
                 (f'Common trigger (SS ≥ 200 or B525 ≥ {common:.0f})',
                  ((d[CH_SS] >= SS_TRIGGER) | (d[CH_S] >= common)).values)
                 if common_mode == 'or' else
                 (f'SS ≥ 200 and B525 ≥ {common:.0f}',
                  ((d[CH_SS] >= SS_TRIGGER) & (d[CH_S] >= common)).values),
                 ('Scatter-triggered (SS ≥ 200)', (d[CH_SS] >= SS_TRIGGER).values)]
        if fluor_only == 'common':
            fluor_only = common
        if fluor_only is not None:
            kinds.append((f'Fluorescence-triggered (SS < 200 and B525 ≥ {fluor_only:.0f})',
                          ((d[CH_SS] < SS_TRIGGER) & (d[CH_S] >= fluor_only)).values))
        for kind, m in kinds:
            e, se = rate(int(m.sum()), t)
            rows.append(dict(sample=name, replicate=rep, file=f'{folder}/{fname}', events=kind, eps=e,
                             poisson_se=se, n=int(m.sum()), acq_time_s=t, empirical_B525_trigger=b525_trigger(d),
                             **{f'recorded_{k}': v for k, v in settings[fname].items()}))
    df = pd.DataFrame(rows)
    for kind in df.events.unique():
        q = df[df.events == kind].groupby('sample', sort=False).eps.mean()
        b = q.get('Buffer', np.nan)
        for other in [x for x in q.index if x != 'Buffer']:
            stat(figname, f'rate ratio (Poisson counts), {other} / buffer', f'{label}, {kind}',
                 q[other] / b if b > 0 else np.nan,
                 note='no buffer subtraction; see poisson_se' + ('' if b > 0 else '; buffer has 0 events'))
    stat(figname, 'empirical B525-H trigger levels', label, np.nan,
         note=f"probe-only {trig['probe-only']:.0f}, buffer {trig['buffer']:.0f}")
    fig = plt.figure(figsize=(24, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])
    ax = fig.add_subplot(gs[0])
    hue_order = list(dict.fromkeys(df['sample']))
    palette = [BUF_COL, '#6A3D9A', ORI_COL][:len(hue_order)]
    sns.barplot(data=df, x='events', y='eps', hue='sample', hue_order=hue_order, palette=palette,
                errorbar='se', capsize=0.08, edgecolor='black', ax=ax)
    if (df.replicate != '').any():
        sns.stripplot(data=df, x='events', y='eps', hue='sample', hue_order=hue_order,
                      dodge=True, color='black', size=5, legend=False, ax=ax)
    ax.set_xlabel(''); ax.set_ylabel('Events per second'); ax.tick_params(axis='x', labelsize=11)
    ax.set_xticklabels([t.get_text().replace(' (', '\n(').replace(' and B525', '\nand B525')
                        for t in ax.get_xticklabels()])
    ax.legend(fontsize=12, frameon=False, title='')
    ax.set_title('Event rate (Poisson SE in CSV)', fontsize=15)
    xb = np.logspace(0, 5, 81); yb = np.logspace(1, 4.5, 71)
    maps = []
    for k, (name, d) in enumerate([('Buffer', bu), (f'{label} only', pr)]):
        ax = fig.add_subplot(gs[k + 1])
        h = np.histogram2d(d[CH_SS].clip(lower=xb[0]), d[CH_S].clip(lower=yb[0]), [xb, yb])[0] / acq_time(d)
        pc = ax.pcolormesh(xb, yb, np.where(h > 0, h, np.nan).T, norm=LogNorm(vmin=1e-2, vmax=max(np.nanmax(h), 1)),
                           cmap='magma', rasterized=True)
        ax.axvline(SS_TRIGGER, color='cyan', ls='--', lw=1.5)
        ax.axhline(trig['buffer' if name == 'Buffer' else 'probe-only'], color='lime', ls='--', lw=1.5)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('VSSC-1-H (a.u.)'); ax.set_ylabel('B525-H (a.u.)')
        ax.set_title(f'{name}\n(cyan = SS trigger, green = B525 trigger)', fontsize=14)
        fig.colorbar(pc, ax=ax, label='Events/s per bin')
        ii, jj = np.nonzero(h)
        maps += [dict(sample=name, ss_bin_lo=xb[i], ss_bin_hi=xb[i + 1], b525_bin_lo=yb[j], b525_bin_hi=yb[j + 1],
                      events_per_s=h[i, j]) for i, j in zip(ii, jj)]
    fig.suptitle(f'{label}-only control vs buffer ({probe_file.split("__")[-1][:-4]})', y=1.03)
    save(fig, figname, {'_rates': df, '_maps': pd.DataFrame(maps)})


def fig10():
    free_probe_fig('AB_Only__Shervin03_13_2026.fcs', 'Buffer__Shervin03_13_2026.fcs', 'Antibody',
                   'fig10_free_antibody', fluor_only=250, common_mode='and',
                   extra=('Uncoated origami, no antibody (04/18, A-C)', 'Shervin4_18',
                          ['N_NoCoating_NoAB_A.fcs', 'N_NoCoating_NoAB_B.fcs', 'N_NoCoating_NoAB_C.fcs']))


def fig11():
    free_probe_fig('Streptavidin_Only__Shervin5_03.fcs', 'Buffer__Shervin5_03.fcs', 'Streptavidin',
                   'fig11_free_streptavidin', fluor_only='common', common_mode='and')


# ---------------------------------------------------------------- fig12
def fig12():
    s = SESSIONS['Antibody session (04/18)']
    ab = L('probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs')
    ab_sub = ab[ab[CH_SS] < SS_TRIGGER]
    ss_free95 = float(np.percentile(ab_sub[CH_SS], 95))
    raw = {'Antibody only (03/13) | VSSC-1-H, SS < 200': ab_sub[CH_SS].values}
    rows, hist, vcurve = [], [], []
    bins = np.linspace(0, SS_TRIGGER, 41); xc = (bins[:-1] + bins[1:]) / 2
    ss_bins = [0, 50, 100, 150, SS_TRIGGER]
    h = np.histogram(ab_sub[CH_SS], bins)[0]; h = h / h.sum()
    hist += [dict(sample='Antibody only (03/13)', ss_bin_center=a, fraction=b) for a, b in zip(xc, h)]
    vcurve += [dict(sample='Antibody only (03/13)', ss_bin=f'{a}-{b}',
                    median_V525=float(ab_sub[(ab_sub[CH_SS] >= a) & (ab_sub[CH_SS] < b)][CH_ATTO].median()))
               for a, b in zip(ss_bins[:-1], ss_bins[1:])]
    trig_ab = b525_trigger(ab)
    for c in COATINGS:
        pooled = []
        for r in REPLICATES:
            d = L(s['folder'], f"N_{c}_AB_{r}.fcs"); d0 = L(s['folder'], f"N_{c}_NoAB_{r}.fcs")
            if d is None:
                continue
            t = acq_time(d); sub = d[d[CH_SS] < SS_TRIGGER]; pooled.append(sub)
            raw[f'{COAT_LBL[c]} N + antibody {r} (04/18) | VSSC-1-H, SS < 200'] = sub[CH_SS].values
            above = sub[CH_SS] >= ss_free95
            rows.append(dict(coating=COAT_LBL[c], replicate=r, sample_trigger_B525=b525_trigger(d),
                             antibody_only_trigger_B525=trig_ab,
                             pct_subtrigger_events_above_free_ab_P95_SS=float(above.mean() * 100),
                             eps_subtrigger_above_free_ab_P95_SS=float(above.sum() / t),
                             median_V525_above_free_ab_P95_SS=float(sub[above][CH_ATTO].median()),
                             eps_subtrigger_in_no_antibody_control=(len(d0[d0[CH_SS] < SS_TRIGGER]) / acq_time(d0)
                                                                    if d0 is not None else np.nan)))
        p = pd.concat(pooled); hh = np.histogram(p[CH_SS], bins)[0]; hh = hh / hh.sum()
        hist += [dict(sample=f'{COAT_LBL[c]} N + antibody (04/18, pooled)', ss_bin_center=a, fraction=b) for a, b in zip(xc, hh)]
        vcurve += [dict(sample=f'{COAT_LBL[c]} N + antibody', ss_bin=f'{a}-{b}',
                        median_V525=float(p[(p[CH_SS] >= a) & (p[CH_SS] < b)][CH_ATTO].median()))
                   for a, b in zip(ss_bins[:-1], ss_bins[1:])]
    df = pd.DataFrame(rows)
    for c in df.coating.unique():
        v = df[df.coating == c].pct_subtrigger_events_above_free_ab_P95_SS
        t = ttest_1samp(v, 5.0)
        stat('fig12', 'one-sample t-test, % sub-trigger events above free-antibody P95 SS vs 5%', c,
             float(v.mean()), t.pvalue, len(v), 'antibody-only control from a different session (03/13)')
    try:
        from quantify_bound_origami_antibody import quantify
        qb = quantify(verbose=False); qb = qb[qb.qc_pass]
        qb = qb.assign(coating=qb.coating.map(COAT_LBL))
        unmix = qb[['coating', 'ext', 'rep', 'bound_loSS', 'bound_hiSS', 'frac_bound_from_loSS']]
    except Exception as e:  # keep the figure even if the unmixing fails
        print('  unmixing skipped:', e); unmix = pd.DataFrame()
    fig, axes = plt.subplots(1, 4, figsize=(30, 6.5))
    H = pd.DataFrame(hist)
    for smp, q in H.groupby('sample', sort=False):
        col = 'black' if 'only' in smp else COAT_COLORS[[c for c in COATINGS if COAT_LBL[c] in smp][0]]
        axes[0].step(q.ss_bin_center, q.fraction, where='mid', lw=2.5, color=col, label=smp)
    axes[0].axvline(ss_free95, color='black', ls=':', lw=2, label=f'Free-antibody P95 SS ({ss_free95:.0f})')
    axes[0].set_xlabel('VSSC-1-H (a.u.), events below scatter trigger'); axes[0].set_ylabel('Fraction of events')
    axes[0].legend(fontsize=10, frameon=False); axes[0].set_title('Side scatter of fluorescence-triggered events', fontsize=14)
    sns.stripplot(data=df, x='coating', y='pct_subtrigger_events_above_free_ab_P95_SS', hue='coating',
                  palette={COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}, size=10, ax=axes[1], legend=False)
    axes[1].axhline(5, color='black', ls=':', lw=2, label='Free antibody (5% by definition)')
    axes[1].set_xlabel(''); axes[1].set_ylabel(f'% of sub-trigger events with SS ≥ {ss_free95:.0f}')
    axes[1].legend(fontsize=11, frameon=False); axes[1].set_title('Events scattering more than free antibody', fontsize=14)
    V = pd.DataFrame(vcurve)
    for smp, q in V.groupby('sample', sort=False):
        col = 'black' if 'only' in smp else COAT_COLORS[[c for c in COATINGS if COAT_LBL[c] in smp][0]]
        axes[2].plot(range(len(q)), q.median_V525, 'o-', lw=2.5, color=col, label=smp)
    axes[2].set_xticks(range(len(ss_bins) - 1)); axes[2].set_xticklabels([f'{a}-{b}' for a, b in zip(ss_bins[:-1], ss_bins[1:])])
    axes[2].set_xlabel('VSSC-1-H bin (below trigger)'); axes[2].set_ylabel('Median V525-H (a.u.)')
    axes[2].legend(fontsize=10, frameon=False); axes[2].set_title('ATTO signal of sub-trigger events', fontsize=14)
    if len(unmix):
        sns.barplot(data=unmix, x='coating', y='frac_bound_from_loSS', hue='ext', hue_order=['N', '3', '6', '12'],
                    errorbar='se', capsize=0.1, edgecolor='black', ax=axes[3], palette='Greys')
        axes[3].set_ylabel('% of bound origami below trigger'); axes[3].set_xlabel('')
        axes[3].legend(title='Extension', fontsize=10, frameon=False)
        axes[3].set_title('V525 unmixing (quantify_bound_origami_antibody.py)', fontsize=14)
    fig.suptitle('Antibody session: fluorescence-triggered events below the scatter trigger', y=1.03)
    save(fig, 'fig12_subtrigger_antibody_events', {'_per_sample': df, '_ss_histograms': H, '_v525_by_ss': V,
                                                   '_unmixing': unmix})
    write_raw('fig12_subtrigger_antibody_events', '_raw_SS_subtrigger', raw)


# ---------------------------------------------------------------- fig13 / fig14
_metrics = {}


def fig5_metrics():
    if 'df' not in _metrics:
        from make_figure5 import Analysis
        _metrics['df'] = Analysis('new').metrics()
    return _metrics['df']


def extension_fig(sess_key, sname, figname, extra=None):
    m = fig5_metrics(); m = m[m.session == sess_key]
    n2 = m[m.extension == 'N'].set_index(['coating', 'replicate']).M2_biotin_specific_fraction
    m = m[m.extension != 'N'].copy()
    m['delta_M2'] = [v - n2.get((c, r), np.nan) for v, c, r in zip(m.M2_biotin_specific_fraction, m.coating, m.replicate)]
    m['length_nt'] = m.extension.astype(int)
    panels = [('M1_biotin_MFI_shift', 'M1: log$_2$(median ext / median N)'),
              ('delta_M2', 'ΔM2: % above N P99 (ext − N)')]
    tables = {'': m[['coating', 'extension', 'replicate', 'length_nt', 'M1_biotin_MFI_shift',
                     'M2_biotin_specific_fraction', 'delta_M2']]}
    if extra is not None:
        e = extra.rename(columns={'ext': 'extension', 'rep': 'replicate'})
        e = e[e.extension != 'N'].assign(length_nt=lambda z: z.extension.astype(int))
        m = m.merge(e[['coating', 'extension', 'replicate', 'delta_vs_N']], on=['coating', 'extension', 'replicate'], how='left')
        panels.append(('delta_vs_N', 'Δ % bound, both scatter regions (ext − N)'))
        tables[''] = m[['coating', 'extension', 'replicate', 'length_nt', 'M1_biotin_MFI_shift',
                        'M2_biotin_specific_fraction', 'delta_M2', 'delta_vs_N']]
    fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 6.5))
    trend = []
    for ax, (col, ylabel) in zip(np.atleast_1d(axes), panels):
        for k, c in enumerate(COATINGS):
            q = m[(m.coating == c) & m[col].notna()]
            if q.empty:
                continue
            xs = q.length_nt.values; ys = q[col].values
            off = (k - 1) * 0.35
            g = q.groupby('length_nt')[col]
            ax.errorbar(g.mean().index + off, g.mean().values, yerr=g.sem().fillna(0).values, fmt='o-',
                        color=COAT_COLORS[c], lw=2.5, capsize=5, ms=9)
            ax.scatter(xs + off, ys, color=COAT_COLORS[c], s=25, alpha=0.6, zorder=4)
            if len(np.unique(xs)) >= 2 and len(xs) >= 3:
                lr = linregress(xs, ys); sp = spearmanr(xs, ys)
                trend.append(dict(metric=col, coating=COAT_LBL[c], slope_per_nt=lr.slope, slope_p=lr.pvalue,
                                  spearman_rho=sp.statistic, spearman_p=sp.pvalue, n=len(xs)))
                stat(figname, f'linear trend of {col} vs extension length (3, 6, 12 nt)', f'{sname}, {COAT_LBL[c]}',
                     lr.slope, lr.pvalue, len(xs), f'Spearman rho {sp.statistic:.2f}, p {sp.pvalue:.2f}')
                lbl = f'{COAT_LBL[c]}: slope {lr.slope:+.3f}/nt, p = {lr.pvalue:.2f}'
            else:
                lbl = f'{COAT_LBL[c]}: too few points for a trend'
            ax.plot([], [], 'o-', color=COAT_COLORS[c], label=lbl)
        ax.axhline(0, color='red', ls='--', lw=1.5)
        ax.set_xticks([3, 6, 12]); ax.set_xlabel('Biotin extension (nt)'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=11, frameon=False)
    tables['_trend_stats'] = pd.DataFrame(trend)
    fig.suptitle(f'{sname}: change from no-biotin control vs biotin extension (origami gate, buffer-subtracted)', y=1.03)
    save(fig, figname, tables)


def fig13():
    from quantify_bound_origami_antibody import quantify, with_delta
    extension_fig('ab', 'Antibody session (04/18)', 'fig13_extension_dependence_antibody',
                  extra=with_delta(quantify(verbose=False)))


def fig14():
    extension_fig('strep', 'Streptavidin session (05/25)', 'fig14_extension_dependence_streptavidin')


# ---------------------------------------------------------------- fig15
def fig15():
    """Conjugate classification applied to no-biotin + antibody, origami only and antibody only.

    Uses the identical gate and thresholds as figure 13's % bound panel
    (quantify_bound_origami_antibody.py): origami gate from shared_utils, and conjugate =
    B525-H above T(coating, replicate) = P99 of that replicate's origami-only (no antibody)
    events, buffer-subtracted. Thresholds are read from quantify() so they are identical.
    Antibody-only (one file) is classified with each replicate's threshold in turn.
    Every sample is buffer-subtracted with its own session's buffer, gated the same way.
    """
    s = SESSIONS['Antibody session (04/18)']
    dfs = session_dfs(s)
    gates = compute_monomer_gates(dfs, ctl=s['noprobe'])
    buf = dfs['Buffer']; tb = acq_time(buf)
    ab = L('probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs')
    ab_buf = L('probe_only_controls', 'Buffer__Shervin03_13_2026.fcs')
    pos_gated = lambda d, c: (lambda v: v[v > 0])(apply_gate(d, c, 'A', gates)[CH_S].values)
    groups = [('Origami only (no antibody)', '0.75'), ('No-biotin origami + antibody', ORI_COL),
              ('Antibody only (03/13)', '#6A3D9A')]
    from quantify_bound_origami_antibody import quantify
    thr_map = quantify(verbose=False).groupby(['coating', 'rep']).thr.first().to_dict()
    rows = []
    for c in COATINGS:
        if (c, 'A') not in gates:
            continue
        bg = pos_gated(buf, c)
        samples = []
        for r in REPLICATES:
            if (c, r) not in thr_map:
                continue
            thr = thr_map[(c, r)]
            if ('N', c, s['noprobe'], r) in dfs:
                samples.append((groups[0][0], r, dfs[('N', c, s['noprobe'], r)], bg, tb, thr))
            if ('N', c, s['probe'], r) in dfs:
                samples.append((groups[1][0], r, dfs[('N', c, s['probe'], r)], bg, tb, thr))
            samples.append((groups[2][0], f'threshold of rep {r}', ab, pos_gated(ab_buf, c), acq_time(ab_buf), thr))
        for grp, r, d, b, t_b, thr in samples:
            x = pos_gated(d, c); t = acq_time(d)
            net, above = bs_counts(x, t, b, t_b, thr)
            rows.append(dict(coating=COAT_LBL[c], sample=grp, replicate=r, B525_threshold=thr,
                             n_gated=len(x), net_gated=net, net_conjugate=above,
                             pct_gated_classified_conjugate=above / net * 100 if net > 0 else np.nan,
                             conjugate_events_per_s=above / t, acq_time_s=t))
    df = pd.DataFrame(rows)
    for c in df.coating.unique():
        q = df[df.coating == c]
        a = q[q['sample'] == groups[1][0]].pct_gated_classified_conjugate
        b = q[q['sample'] == groups[0][0]].pct_gated_classified_conjugate
        if len(a) >= 2 and len(b) >= 2:
            t = ttest_ind(a, b, equal_var=False)
            stat('fig15', 'Welch t-test, % classified conjugate: no-biotin + antibody vs origami only', c,
                 float(a.mean() - b.mean()), t.pvalue, len(a) + len(b),
                 'value = difference in mean % (percentage points)')
        abv = q[q['sample'] == groups[2][0]].pct_gated_classified_conjugate
        stat('fig15', '% classified conjugate, antibody only (single file, 03/13 session)', c,
             float(abv.mean()) if len(abv) else np.nan, n=len(abv),
             note='mean over the three replicate thresholds')
    order = [COAT_LBL[c] for c in COATINGS]
    fig, axes = plt.subplots(1, 2, figsize=(19, 7))
    for ax, col, ylabel in [(axes[0], 'pct_gated_classified_conjugate', '% of gated events classified as conjugates'),
                            (axes[1], 'conjugate_events_per_s', 'Conjugate-classified events per second')]:
        sns.barplot(data=df, x='coating', y=col, hue='sample', order=order, hue_order=[g for g, _ in groups],
                    palette=[c for _, c in groups], errorbar='se', capsize=0.08, edgecolor='black', ax=ax)
        sns.stripplot(data=df, x='coating', y=col, hue='sample', order=order, hue_order=[g for g, _ in groups],
                      dodge=True, color='black', size=7, legend=False, ax=ax)
        ax.set_xlabel(''); ax.set_ylabel(ylabel)
        ax.legend(fontsize=12, frameon=False, title='')
    axes[0].axhline(1, color='red', ls='--', lw=1.5)
    axes[1].get_legend().remove()
    axes[0].set_title('Share of gated events (red line = 1%, origami-only P99 by construction;\n'
                      'antibody only: one file scored with each replicate threshold)', fontsize=13)
    axes[1].set_title('Rate (buffer-subtracted)', fontsize=14)
    fig.suptitle('Events classified as origami-antibody conjugates (same gate and per-replicate thresholds as fig13 % bound)', y=1.05)
    save(fig, 'fig15_nobiotin_conjugate_classification', {'': df})


# ---------------------------------------------------------------- fig16 / fig16b
PROBE_ONLY_FILES = {'Streptavidin session (05/25)': ('Streptavidin_Only__Shervin5_03.fcs', 'Buffer__Shervin5_03.fcs',
                                                     'Streptavidin only (05/03)'),
                    'Antibody session (04/18)': ('AB_Only__Shervin03_13_2026.fcs', 'Buffer__Shervin03_13_2026.fcs',
                                                 'Antibody only (03/13)')}


_NS_CACHE = {}


def nonspecific_fig(figname, exclude_debris=False, probe_only=False, data_only=False):
    """Non-specific probe association by coating (no-biotin origami), both probes.

    Per coating and replicate, on the origami gate with session-buffer subtraction:
      log2_median_ratio      log2(median B525-H with probe / without probe)
      pct_above_noprobe_P99  % of gated with-probe events above the P99 of the same
                             replicate's no-probe sample (same thresholds as fig13 % bound / fig15)
      eps_above_noprobe_P99  events/s above that threshold
    exclude_debris: also drop events with SS > uncoated P95 SS and V525c < uncoated median V525c
      (V525c = V525 - k*B525), both from the session's pooled uncoated no-probe origami
      (scatter-triggered); applied identically to every sample and buffer.
    probe_only: also score the probe-only control (other session, own buffer) with each
      replicate's gate and threshold in turn.
    """
    from shared_utils import spillover_k, v525c
    rows, excl = [], []
    for sname, s in SESSIONS.items():
        dfs = session_dfs(s)
        gates = compute_monomer_gates(dfs, ctl=s['noprobe'])
        k = spillover_k(s['noprobe'])
        buf = dfs['Buffer']; tb = acq_time(buf)
        if exclude_debris:
            uo = pd.concat([scat(dfs[('N', 'NoCoating', s['noprobe'], r)]) for r in REPLICATES
                            if ('N', 'NoCoating', s['noprobe'], r) in dfs])
            ss_hi = float(np.percentile(uo[CH_SS], 95)); vc_lo = float(np.median(v525c(uo, k)))
        def sel(d, c):
            g = apply_gate(d, c, 'A', gates)
            if exclude_debris:
                g = g[~((g[CH_SS] > ss_hi) & (v525c(g, k) < vc_lo))]
            v = g[CH_S].values
            return v[v > 0]
        if probe_only:
            pf, pbf, plabel = PROBE_ONLY_FILES[sname]
            po = L('probe_only_controls', pf); pob = L('probe_only_controls', pbf)
        for c in COATINGS:
            if (c, 'A') not in gates:
                continue
            b = sel(buf, c)
            for r in REPLICATES:
                d0 = dfs.get(('N', c, s['noprobe'], r)); d1 = dfs.get(('N', c, s['probe'], r))
                if d0 is None or d1 is None:
                    continue
                v0 = sel(d0, c); t0 = acq_time(d0); v1 = sel(d1, c); t1 = acq_time(d1)
                m0 = bs_percentile(v0, t0, b, tb, 50); m1 = bs_percentile(v1, t1, b, tb, 50)
                thr = bs_percentile(v0, t0, b, tb, 99)
                net, above = bs_counts(v1, t1, b, tb, thr)
                rows.append(dict(session=sname, probe=s['probe_name'], coating=COAT_LBL[c], replicate=r,
                                 sample='No-biotin origami + probe', median_B525_no_probe=m0,
                                 median_B525_with_probe=m1, log2_median_ratio=np.log2(m1 / m0),
                                 noprobe_P99_threshold=thr, net_gated=net, net_above_threshold=above,
                                 pct_above_noprobe_P99=above / net * 100 if net > 0 else np.nan,
                                 eps_above_noprobe_P99=above / t1))
                if exclude_debris:
                    for lbl, d in [('no probe', d0), ('with probe', d1)]:
                        g = apply_gate(d, c, 'A', gates)
                        excl.append(dict(session=sname, coating=COAT_LBL[c], replicate=r, sample=lbl,
                                         gated=len(g), excluded=int(((g[CH_SS] > ss_hi) & (v525c(g, k) < vc_lo)).sum()),
                                         ss_high_cut=ss_hi, v525c_low_cut=vc_lo))
                if probe_only:
                    vp = sel(po, c); vb = sel(pob, c); tp = acq_time(po)
                    net, above = bs_counts(vp, tp, vb, acq_time(pob), thr)
                    rows.append(dict(session=sname, probe=s['probe_name'], coating=COAT_LBL[c],
                                     replicate=f'threshold of rep {r}', sample=plabel, noprobe_P99_threshold=thr,
                                     net_gated=net, net_above_threshold=above,
                                     pct_above_noprobe_P99=above / net * 100 if net > 0 else np.nan,
                                     eps_above_noprobe_P99=above / tp))
    df = pd.DataFrame(rows)
    _NS_CACHE[(exclude_debris, probe_only)] = df
    if data_only:
        return df
    ori = df[df['sample'] == 'No-biotin origami + probe']
    metrics = [('log2_median_ratio', 'log$_2$(median B525-H with / without probe)', 0),
               ('pct_above_noprobe_P99', '% of gated events above no-probe P99', 1)]
    if probe_only:
        metrics.append(('eps_above_noprobe_P99', 'Events/s above no-probe P99', None))
    ptab = []
    for sname in SESSIONS:
        q = ori[ori.session == sname]
        for col, _, _ in metrics:
            groups = [q[q.coating == COAT_LBL[c]][col].dropna() for c in COATINGS]
            if all(len(g) >= 2 for g in groups):
                a = f_oneway(*groups)
                stat(figname, f'one-way ANOVA across coatings, {col}', sname, a.statistic, a.pvalue,
                     sum(len(g) for g in groups))
                ptab.append(dict(session=sname, metric=col, comparison='ANOVA, all coatings',
                                 statistic=a.statistic, p=a.pvalue, n=sum(len(g) for g in groups)))
            si = q[q.coating == 'Silica'][col].dropna()
            for other in ['Uncoated', 'PLL-PEG']:
                o = q[q.coating == other][col].dropna()
                if len(si) >= 2 and len(o) >= 2:
                    t = ttest_ind(si, o, equal_var=False)
                    stat(figname, f'Welch t-test, Silica vs {other}, {col}', sname,
                         float(si.mean() - o.mean()), t.pvalue, len(si) + len(o), 'value = Silica minus other')
                    ptab.append(dict(session=sname, metric=col, comparison=f'Silica vs {other}',
                                     statistic=float(si.mean() - o.mean()), p=t.pvalue, n=len(si) + len(o),
                                     note='statistic = difference in means (Silica - other)'))
        if probe_only:
            pq = df[(df.session == sname) & (df['sample'] != 'No-biotin origami + probe')]
            for c in pq.coating.unique():
                po_rate = pq[pq.coating == c].eps_above_noprobe_P99.mean()
                or_rate = q[q.coating == c].eps_above_noprobe_P99.mean()
                ptab.append(dict(session=sname, metric='eps_above_noprobe_P99',
                                 comparison=f'{c}: probe-only / no-biotin origami + probe',
                                 statistic=po_rate / or_rate if or_rate > 0 else np.nan, p=np.nan,
                                 note='ratio of mean rates; probe-only acquired in another session'))
                stat(figname, 'rate ratio above threshold, probe-only / no-biotin origami + probe',
                     f'{sname}, {c}', po_rate / or_rate if or_rate > 0 else np.nan,
                     note='probe-only from another session, own buffer')
    P = pd.DataFrame(ptab)
    order = [COAT_LBL[c] for c in COATINGS]
    pal = {COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}
    fig, axes = plt.subplots(len(metrics), 2, figsize=(17, 6 * len(metrics)))
    for i, (col, ylabel, null) in enumerate(metrics):
        for j, sname in enumerate(SESSIONS):
            ax = axes[i, j]; q = df[df.session == sname]; qo = q[q['sample'] == 'No-biotin origami + probe']
            if probe_only and col != 'log2_median_ratio':
                hue_order = ['No-biotin origami + probe', PROBE_ONLY_FILES[sname][2]]
                sns.barplot(data=q, x='coating', y=col, hue='sample', order=order, hue_order=hue_order,
                            palette=[ORI_COL, '#6A3D9A'], errorbar='se', capsize=0.1, edgecolor='black', ax=ax)
                sns.stripplot(data=q, x='coating', y=col, hue='sample', order=order, hue_order=hue_order,
                              dodge=True, color='black', size=7, legend=False, ax=ax)
                ax.legend(fontsize=11, frameon=False, title='', loc='upper right')
                pos = {c: order.index(c) - 0.2 for c in order}
            else:
                sns.barplot(data=qo, x='coating', y=col, hue='coating', order=order, palette=pal, errorbar='se',
                            capsize=0.15, edgecolor='black', legend=False, ax=ax)
                sns.stripplot(data=qo, x='coating', y=col, order=order, color='black', size=8, ax=ax)
                pos = {c: order.index(c) for c in order}
            if null is not None:
                ax.axhline(null, color='red', ls='--', lw=1.5)
            ymax = np.nanmax(q[col].values) if len(q) else 1
            for kk, other in enumerate(['Uncoated', 'PLL-PEG']):
                pr = P[(P.session == sname) & (P.metric == col) & (P.comparison == f'Silica vs {other}')]
                if len(pr):
                    x0, x1 = pos[other], pos['Silica']; y = ymax * (1.08 + 0.12 * kk) + 0.02
                    ax.plot([x0, x0, x1, x1], [y * 0.98, y, y, y * 0.98], color='black', lw=1.2)
                    ax.text((x0 + x1) / 2, y, f'p = {pr.p.iloc[0]:.3f}', ha='center', va='bottom', fontsize=12)
            ax.set_ylim(top=ymax * 1.45 + 0.05)
            ax.set_xlabel(''); ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(sname, fontsize=16)
    extra = ' excluding high-scatter / low-V525 events,' if exclude_debris else ''
    fig.suptitle(f'Non-specific probe association on no-biotin origami by coating (origami gate,{extra} '
                 'buffer-subtracted; p = Welch t-test, Silica vs each coating, origami samples only)', y=1.02)
    tables = {'': df, '_stats': P}
    if exclude_debris:
        E = pd.DataFrame(excl); E['pct_excluded'] = E.excluded / E.gated * 100
        tables['_excluded_events'] = E
    save(fig, figname, tables)


def fig16():
    nonspecific_fig('fig16_nonspecific_by_coating')


def fig16b():
    nonspecific_fig('fig16b_nonspecific_by_coating_excl_debris_vs_probe_only', exclude_debris=True, probe_only=True)


# ---------------------------------------------------------------- fig17
def fig17():
    """Non-specific adsorption summary, both probes (same data as fig16b).

    A  log2(median B525-H with / without probe) on no-biotin origami, per coating and probe;
       one-sample t-test of the per-replicate log2 ratios against 0 (= paired t-test on log medians).
    B  Share of above-threshold events that free-probe clumps could account for, per replicate:
       probe-only events/s above the replicate's no-probe P99 / no-biotin origami + probe events/s
       above the same threshold (probe-only acquired in another session; an estimate that assumes
       comparable probe concentration).
    """
    df = _NS_CACHE.get((True, True))
    if df is None:
        df = nonspecific_fig('fig17', exclude_debris=True, probe_only=True, data_only=True)
    ori = df[df['sample'] == 'No-biotin origami + probe'].copy()
    po = df[df['sample'] != 'No-biotin origami + probe'].copy()
    po['rep'] = po.replicate.str[-1]
    m = ori.merge(po[['session', 'coating', 'rep', 'eps_above_noprobe_P99']].rename(
        columns={'rep': 'replicate', 'eps_above_noprobe_P99': 'probe_only_eps_above_P99'}),
        on=['session', 'coating', 'replicate'], how='left')
    m['clump_share_pct'] = m.probe_only_eps_above_P99 / m.eps_above_noprobe_P99 * 100
    m['probe_label'] = m.probe.map({'streptavidin': 'Streptavidin', 'anti-biotin IgG': 'Anti-biotin IgG'})
    ptab = []
    for (probe, coat), q in m.groupby(['probe_label', 'coating'], sort=False):
        v = q.log2_median_ratio.dropna()
        t = ttest_1samp(v, 0) if len(v) >= 2 else None
        ptab.append(dict(probe=probe, coating=coat, n=len(v), mean_log2_ratio=v.mean(), sem=v.sem(),
                         p_vs_0=t.pvalue if t else np.nan,
                         mean_clump_share_pct=q.clump_share_pct.mean(), sem_clump_share=q.clump_share_pct.sem()))
        stat('fig17', 'one-sample t-test, log2(median with/without probe) vs 0', f'{probe}, {coat}',
             v.mean(), t.pvalue if t else np.nan, len(v))
        stat('fig17', 'free-probe clump share of above-threshold events (%)', f'{probe}, {coat}',
             q.clump_share_pct.mean(), n=len(q), note='probe-only from another session')
    P = pd.DataFrame(ptab)
    order = [COAT_LBL[c] for c in COATINGS]; probes = ['Anti-biotin IgG', 'Streptavidin']
    pal = {'Anti-biotin IgG': '#E6862E', 'Streptavidin': '#6A3D9A'}
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, col, ylabel in [(axes[0], 'log2_median_ratio', 'log$_2$(median B525-H with / without probe)'),
                            (axes[1], 'clump_share_pct', '% of above-threshold events\nexplainable by free-probe clumps')]:
        sns.barplot(data=m, x='coating', y=col, hue='probe_label', order=order, hue_order=probes, palette=pal,
                    errorbar='se', capsize=0.08, edgecolor='black', ax=ax)
        sns.stripplot(data=m, x='coating', y=col, hue='probe_label', order=order, hue_order=probes, dodge=True,
                      color='black', size=7, legend=False, ax=ax)
        ax.set_xlabel(''); ax.set_ylabel(ylabel); ax.legend(fontsize=12, frameon=False, title='')
    for _, r in P.iterrows():
        x = order.index(r.coating) + (-0.2 if r.probe == probes[0] else 0.2)
        top = m[(m.probe_label == r.probe) & (m.coating == r.coating)].log2_median_ratio.max()
        axes[0].text(x, top + 0.04, f'p = {r.p_vs_0:.3f}' if r.p_vs_0 >= 0.001 else 'p < 0.001',
                     ha='center', va='bottom', fontsize=11)
    axes[0].axhline(0, color='red', ls='--', lw=1.5)
    axes[0].set_title('A  Non-specific association (no-biotin origami)\np = one-sample t-test vs 0', fontsize=14)
    axes[1].set_title('B  Upper estimate of free-probe clump contribution\n(probe-only control, other session)', fontsize=14)
    fig.suptitle('Non-specific probe adsorption on origami (origami gate excluding high-scatter / low-V525 '
                 'events, buffer-subtracted)', y=1.03)
    save(fig, 'fig17_nonspecific_adsorption_summary',
         {'': m.drop(columns=['probe_label']), '_stats': P})


# ---------------------------------------------------------------- fig18 / fig19
def coating_only_event_rate_fig(figname, coat, control_file, control_label, color, include_uncoated=False,
                                subdir=None):
    """Event rate of coated origami (no probe) vs the coating reagent alone (no origami).

    Recorded trigger settings are read from every file. The coating-only controls (02/13) were
    acquired with the scatter trigger alone (VSSC-1 >= 200, no B525 trigger); all origami files used
    VSSC-1 >= 200 OR B525 >= 150/250. Only scatter-triggered events (SS >= 200) are recorded identically
    in both, so the comparison is made on those; all-event rates are saved to the CSV but flagged as
    not comparable. No buffer subtraction; buffers are shown for reference (no 02/13 buffer exists).
    """
    lbl = COAT_LBL[coat]
    groups = [(f'{control_label} (02/13)', 'coatings_controls', [control_file], color),
              (f'{lbl} origami, no probe (05/25)', 'Shervin5_25', [f'N_{coat}_NoS_{r}.fcs' for r in REPLICATES], COAT_COLORS[coat]),
              (f'{lbl} origami, no probe (04/18)', 'Shervin4_18', [f'N_{coat}_NoAB_{r}.fcs' for r in REPLICATES], COAT_COLORS[coat]),
              (f'{lbl} origami, no probe (03/13)', 'probe_only_controls', [f'ATTO_only_N_{coat}_NoAB__Shervin03_13_2026.fcs'], COAT_COLORS[coat]),
              ('Buffer (05/25)', 'Shervin5_25', ['Buffer.fcs'], BUF_COL),
              ('Buffer (04/18)', 'Shervin4_18', ['Buffer.fcs'], BUF_COL),
              ('Buffer (03/13)', 'probe_only_controls', ['Buffer__Shervin03_13_2026.fcs'], BUF_COL)]
    if include_uncoated:
        groups[4:4] = [('Uncoated origami, no probe (05/25)', 'Shervin5_25',
                        [f'N_NoCoating_NoS_{r}.fcs' for r in REPLICATES], COAT_COLORS['NoCoating']),
                       ('Uncoated origami, no probe (04/18)', 'Shervin4_18',
                        [f'N_NoCoating_NoAB_{r}.fcs' for r in REPLICATES], COAT_COLORS['NoCoating'])]
    ref = acq_trigger_settings('coatings_controls', control_file)
    rows = []
    for g, folder, files, _ in groups:
        for f in files:
            d = L(folder, f)
            if d is None:
                continue
            st = acq_trigger_settings(folder, f); t = acq_time(d)
            same_scatter = (st['primary_channel'], st['primary_threshold']) == (ref['primary_channel'], ref['primary_threshold'])
            n_ss = int((d[CH_SS] >= SS_TRIGGER).sum())
            rows.append(dict(group=g, file=f'{folder}/{f}', replicate=f.split('_')[-1][:-4] if len(files) > 1 else '',
                             eps_scatter_triggered=n_ss / t, poisson_se_scatter=np.sqrt(n_ss) / t,
                             eps_all_events=len(d) / t, all_events_comparable=all(st[k] == ref[k] for k in st),
                             scatter_trigger_identical_to_control=same_scatter,
                             n_scatter_triggered=n_ss, n_all=len(d), acq_time_s=t,
                             **{f'recorded_{k}': v for k, v in st.items()}))
    df = pd.DataFrame(rows)
    assert df.scatter_trigger_identical_to_control.all(), 'scatter trigger differs'
    ctrl = df[df.group == groups[0][0]].eps_scatter_triggered.iloc[0]
    for g in df.group.unique()[1:]:
        v = df[df.group == g].eps_scatter_triggered
        t = ttest_1samp(v, ctrl) if len(v) >= 2 else None
        stat(figname, f'scatter-triggered events/s vs {control_label} (one-sample t-test when n >= 2)', g,
             v.mean() / ctrl, t.pvalue if t else np.nan, len(v),
             f'value = mean ratio to {control_label}; no buffer subtraction')
    stat(figname, f'recorded trigger settings, {control_label}', control_file, np.nan, note=json_key(ref))
    order = [g for g, *_ in groups]; pal = {g: c for g, _, _, c in groups}
    fig, ax = plt.subplots(figsize=(14 + 2.5 * include_uncoated, 7))
    sns.barplot(data=df, x='group', y='eps_scatter_triggered', hue='group', order=order, palette=pal,
                errorbar='se', capsize=0.15, edgecolor='black', legend=False, ax=ax)
    sns.stripplot(data=df, x='group', y='eps_scatter_triggered', order=order, color='black', size=8, ax=ax)
    for i, g in enumerate(order):
        v = df[df.group == g].eps_scatter_triggered
        ax.text(i, v.max() * 1.03 + 5, f'{v.mean():.0f}', ha='center', va='bottom', fontsize=12)
    ax.set_xticklabels([t.get_text().replace(', ', ',\n').replace(' (', '\n(') for t in ax.get_xticklabels()],
                       fontsize=11)
    ax.set_xlabel(''); ax.set_ylabel('Scatter-triggered events per second (SS ≥ 200)')
    ax.set_title(f'{control_label} vs {lbl}-coated origami (no probe): events recorded by the shared scatter trigger\n'
                 f'{control_label} used scatter-only triggering; origami files used scatter OR B525; no buffer subtraction',
                 fontsize=13)
    save(fig, figname, {'': df}, subdir=subdir)


def fig18():
    coating_only_event_rate_fig('fig18_pll_origami_vs_pll_only_event_rate', 'PLL', 'PLL_PEG_NoOrigami.fcs',
                                'PLL-PEG only, no origami', '#1B5E20', include_uncoated=True, subdir=PLL_DIR)


def fig19():
    coating_only_event_rate_fig('fig19_silica_origami_vs_silica_chemicals_only_event_rate', 'Silica',
                                'SilicaChemicalsOnly.fcs', 'Silica chemicals only, no origami', '#A0522D',
                                include_uncoated=True, subdir=SILICA_DIR)


# ---------------------------------------------------------------- fig20 / fig21
def coating_confident_fig(figname, coat, control_file, control_label, control_color, subdir):
    """Side scatter of coated origami: all events, and events with origami-level ATTO 425 fluorescence.

    A  VSSC-1-H distributions (scatter-triggered, pooled replicates): coated origami, uncoated origami,
       coating-only control (no origami).
    B  V525-H vs VSSC-1-H of coated origami with the background threshold per side-scatter bin
       (P95 and P99 of pooled session buffer + coating-only control, scatter-triggered).
    C  Median VSSC-1-H of origami-confident events (V525-H above the P95 background threshold at the
       same side scatter), coated vs uncoated, per replicate; Welch t-test.
    D  Origami-confident events/s in coated samples vs false positives expected if the excess
       scatter-triggered events over uncoated origami (same session) were background particles
       (excess x 5 % for the P95 threshold).
    E  Median V525-H per side-scatter bin: coated and uncoated origami (buffer-subtracted), coating-only
       control and buffer (raw).
    No buffer subtraction in A-D (per-event classification). Coating-only controls: 02/13, no same-day buffer.
    """
    lbl = COAT_LBL[coat]
    edges = np.logspace(np.log10(SS_TRIGGER), np.log10(20000), 13)
    hist_bins = np.logspace(np.log10(SS_TRIGGER), 5, 101)
    ss_bins = np.logspace(np.log10(SS_TRIGGER), 5, 15)
    ctrl = scat(L('coatings_controls', control_file))

    def thresholds(bgs, q):
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            v = np.concatenate([bg[(bg[CH_SS] >= a) & (bg[CH_SS] < b)][CH_ATTO].values for bg in bgs])
            out.append(np.percentile(v, q) if len(v) >= 30 else np.nan)
        return np.array(out)

    def confident(d, th):
        d = d[(d[CH_SS] >= edges[0]) & (d[CH_SS] < edges[-1])]
        i = np.clip(np.searchsorted(edges, d[CH_SS].values, side='right') - 1, 0, len(th) - 1)
        t = th[i]
        return d[np.isfinite(t) & (d[CH_ATTO].values > t)]

    hists, ths, conf, ptab, curves = [], [], [], [], []
    fig = plt.figure(figsize=(35, 14))
    gs = fig.add_gridspec(2, 4)
    for j, (sname, s) in enumerate(SESSIONS.items()):
        bufall = L(s['folder'], 'Buffer.fcs'); buf = scat(bufall); tb = acq_time(bufall)
        th = {q: thresholds([buf, ctrl], q) for q in (95, 99)}
        ths += [dict(session=sname, ss_bin_lo=a, ss_bin_hi=b, P95_threshold=t95, P99_threshold=t99)
                for a, b, t95, t99 in zip(edges[:-1], edges[1:], th[95], th[99])]
        files = {c: [(r, d) for r in REPLICATES if (d := L(s['folder'], f"N_{c}_{s['noprobe']}_{r}.fcs")) is not None]
                 for c in ['NoCoating', coat]}
        pooled = {c: (pd.concat([scat(d) for _, d in files[c]]), sum(acq_time(d) for _, d in files[c])) for c in files}
        # A
        ax = fig.add_subplot(gs[0, j])
        xcen = np.sqrt(hist_bins[:-1] * hist_bins[1:])
        for name, z, colr, ls in [('Uncoated origami (pooled)', pooled['NoCoating'][0], COAT_COLORS['NoCoating'], '-'),
                                  (f'{lbl} origami (pooled)', pooled[coat][0], COAT_COLORS[coat], '-'),
                                  (f'{control_label} (02/13)', ctrl, control_color, '--')]:
            hh = log_density(z[CH_SS].values, hist_bins)
            ax.step(xcen, hh, where='mid', color=colr, lw=2.5, ls=ls, label=name)
            hists += [dict(session=sname, sample=name, ss_bin_center=a, density=b) for a, b in zip(xcen, hh)]
        ax.set_xscale('log'); ax.set_xlabel('VSSC-1-H (a.u.)'); ax.set_ylabel('Density (per log$_{10}$ unit)')
        ax.set_title(f'A  {sname}: side-scatter distributions (scatter-triggered)', fontsize=14)
        ax.legend(fontsize=11, frameon=False)
        # B
        ax = fig.add_subplot(gs[1, j])
        co = pooled[coat][0]
        xb = np.logspace(np.log10(SS_TRIGGER), 5, 71); yb = np.logspace(0.5, 3.5, 61)
        h = np.histogram2d(co[CH_SS].clip(lower=xb[0]), co[CH_ATTO].clip(lower=yb[0]), [xb, yb])[0]
        pc = ax.pcolormesh(xb, yb, np.where(h > 0, h / h.sum(), np.nan).T, norm=LogNorm(vmin=1e-6), cmap='viridis',
                           rasterized=True)
        ecen = np.sqrt(edges[:-1] * edges[1:])
        ax.step(ecen, th[95], where='mid', color='red', lw=2.5, label=f'Background P95 (buffer + {control_label.split(",")[0]})')
        ax.step(ecen, th[99], where='mid', color='red', lw=1.8, ls='--', label='Background P99')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('VSSC-1-H (a.u.)'); ax.set_ylabel('V525-H (a.u.)')
        ax.set_title(f'B  {sname}: {lbl} origami, with background threshold', fontsize=14)
        ax.legend(fontsize=10, frameon=False, loc='lower right'); fig.colorbar(pc, ax=ax, label='Fraction of events')
        # E
        ax = fig.add_subplot(gs[j, 3])
        xc = np.sqrt(ss_bins[:-1] * ss_bins[1:])
        for c in ['NoCoating', coat]:
            d, t = pooled[c]; med = []
            for a, b in zip(ss_bins[:-1], ss_bins[1:]):
                sv = d[(d[CH_SS] >= a) & (d[CH_SS] < b)][CH_ATTO].values
                bv = buf[(buf[CH_SS] >= a) & (buf[CH_SS] < b)][CH_ATTO].values
                med.append(bs_percentile(sv[sv > 0], t, bv[bv > 0], tb, 50) if len(sv) - len(bv) * t / tb >= 50 else np.nan)
            ax.plot(xc, med, 'o-', color=COAT_COLORS[c], lw=2.5, label=f'{COAT_LBL[c]} origami (buffer-subtracted)')
            curves += [dict(session=sname, sample=f'{COAT_LBL[c]} origami (buffer-subtracted)', ss_bin_center=a,
                            median_V525=b) for a, b in zip(xc, med)]
        for name, z, style, colr, nmin in [(f'{control_label} (raw)', ctrl, 's--', control_color, 50),
                                           ('Buffer (raw)', buf, 'v:', BUF_COL, 20)]:
            med = [z[(z[CH_SS] >= a) & (z[CH_SS] < b)][CH_ATTO].median()
                   if ((z[CH_SS] >= a) & (z[CH_SS] < b)).sum() >= nmin else np.nan for a, b in zip(ss_bins[:-1], ss_bins[1:])]
            ax.plot(xc, med, style, color=colr, lw=2, label=name)
            curves += [dict(session=sname, sample=name, ss_bin_center=a, median_V525=b) for a, b in zip(xc, med)]
        ax.set_xscale('log'); ax.set_xlabel('VSSC-1-H (a.u.)'); ax.set_ylabel('Median V525-H (a.u.)')
        ax.set_title(f'E  {sname}: V525-H across side scatter', fontsize=14); ax.legend(fontsize=10, frameon=False)
        # C / D data
        unc_eps = np.mean([len(scat(d)) / acq_time(d) for _, d in files['NoCoating']])
        for c in ['NoCoating', coat]:
            for r, d in files[c]:
                x = scat(d); t = acq_time(d)
                for q in (95, 99):
                    cf = confident(x, th[q])
                    excess = max(len(x) / t - unc_eps, 0) if c == coat else np.nan
                    conf.append(dict(session=sname, coating=COAT_LBL[c], replicate=r, threshold=f'P{q}',
                                     scatter_triggered_eps=len(x) / t, confident_eps=len(cf) / t,
                                     pct_confident=len(cf) / len(x) * 100, median_SS_confident=float(np.median(cf[CH_SS])),
                                     median_SS_all_scatter_triggered=float(np.median(x[CH_SS])),
                                     uncoated_mean_scatter_triggered_eps=unc_eps, excess_eps_over_uncoated=excess,
                                     expected_false_positive_eps=excess * (100 - q) / 100))
    C = pd.DataFrame(conf)
    for sname in SESSIONS:
        for q in ('P95', 'P99'):
            a = C[(C.session == sname) & (C.threshold == q) & (C.coating == lbl)].median_SS_confident
            b = C[(C.session == sname) & (C.threshold == q) & (C.coating == 'Uncoated')].median_SS_confident
            t = ttest_ind(a, b, equal_var=False)
            ptab.append(dict(session=sname, threshold=q, comparison=f'median SS of confident events, {lbl} vs Uncoated',
                             coated_mean=a.mean(), uncoated_mean=b.mean(), p=t.pvalue, n=len(a) + len(b)))
            stat(figname, f'Welch t-test, median SS of origami-confident events ({q}), {lbl} vs Uncoated', sname,
                 float(a.mean() - b.mean()), t.pvalue, len(a) + len(b), f'value = {lbl} minus Uncoated')
    P = pd.DataFrame(ptab)
    ax = fig.add_subplot(gs[0, 2])
    q95 = C[C.threshold == 'P95'].assign(sess=lambda z: z.session.str.replace(' session', '\nsession'))
    sns.barplot(data=q95, x='sess', y='median_SS_confident', hue='coating', hue_order=['Uncoated', lbl],
                palette=[COAT_COLORS['NoCoating'], COAT_COLORS[coat]], errorbar='se', capsize=0.1,
                edgecolor='black', ax=ax)
    sns.stripplot(data=q95, x='sess', y='median_SS_confident', hue='coating', hue_order=['Uncoated', lbl],
                  dodge=True, color='black', size=8, legend=False, ax=ax)
    for i, sname in enumerate(SESSIONS):
        pr = P[(P.session == sname) & (P.threshold == 'P95')].p.iloc[0]
        top = q95[q95.session == sname].median_SS_confident.max()
        ax.plot([i - 0.2, i - 0.2, i + 0.2, i + 0.2], [top * 1.04, top * 1.08, top * 1.08, top * 1.04], color='black', lw=1.2)
        ax.text(i, top * 1.09, f'p = {pr:.3f}', ha='center', va='bottom', fontsize=12)
    ax.set_ylim(top=q95.median_SS_confident.max() * 1.25)
    ax.set_xlabel(''); ax.set_ylabel('Median VSSC-1-H of origami-confident events')
    ax.set_title('C  Side scatter of events above background P95', fontsize=14)
    ax.legend(fontsize=11, frameon=False, title='')
    ax = fig.add_subplot(gs[1, 2])
    D = C[(C.coating == lbl) & (C.threshold == 'P95')].melt(
        id_vars=['session', 'replicate'], value_vars=['confident_eps', 'expected_false_positive_eps'],
        var_name='quantity', value_name='eps')
    D['quantity'] = D.quantity.map({'confident_eps': 'Observed origami-confident',
                                    'expected_false_positive_eps': 'Expected false positives\n(excess over uncoated x 5%)'})
    D['sess'] = D.session.str.replace(' session', '\nsession')
    sns.barplot(data=D, x='sess', y='eps', hue='quantity', palette=[COAT_COLORS[coat], '0.7'], errorbar='se',
                capsize=0.1, edgecolor='black', ax=ax)
    sns.stripplot(data=D, x='sess', y='eps', hue='quantity', dodge=True, color='black', size=8, legend=False, ax=ax)
    ax.set_xlabel(''); ax.set_ylabel(f'Events per second ({lbl} samples)')
    ax.set_title(f'D  Confident {lbl} events vs expected background false positives', fontsize=14)
    ax.legend(fontsize=11, frameon=False, title='')
    fig.suptitle(f'{lbl} side scatter: all events, and events with origami-level ATTO 425 fluorescence at matched side scatter', y=1.02)
    save(fig, figname,
         {'_confident_events': C, '_stats': P, '_thresholds': pd.DataFrame(ths), '_ss_histograms': pd.DataFrame(hists),
          '_median_v525_by_ss': pd.DataFrame(curves)}, subdir=subdir)


def fig20():
    coating_confident_fig('fig20_silica_side_scatter_confident_origami', 'Silica', 'SilicaChemicalsOnly.fcs',
                          'Silica chemicals only', '#A0522D', SILICA_DIR)


def fig21():
    coating_confident_fig('fig21_pll_side_scatter_confident_origami', 'PLL', 'PLL_PEG_NoOrigami.fcs',
                          'PLL-PEG only', '#1B5E20', PLL_DIR)


def copy_into(src_subdir, dst_subdir, prefix):
    """Copy an already-written figure (svg + csv) into another figure folder."""
    import shutil, glob as _glob
    os.makedirs(os.path.join(OUT, dst_subdir), exist_ok=True)
    for f in _glob.glob(os.path.join(OUT, src_subdir, f'{prefix}*')):
        try:
            shutil.copy2(f, os.path.join(OUT, dst_subdir, os.path.basename(f)))
        except PermissionError:
            print(f'  WARNING: could not copy {os.path.basename(f)} (file open elsewhere?)')


# ---------------------------------------------------------------- fig22
def fig22():
    """2x2 quadrants of side scatter x ATTO 425 fluorescence, per day and coating.

    Scatter-triggered events (SS >= 200). Per day (session):
      high scatter      VSSC-1-H > P95 of that day's pooled uncoated origami (no probe)
      high fluorescence V525-H  > median of that day's pooled uncoated origami (no probe)
    Origami samples (replicates pooled) are buffer-subtracted with the same-day buffer, so each cell
    is a % of net events; coating-only controls (02/13, no same-day buffer) are raw and classified with
    each day's cut-offs. Day 1 = streptavidin session (05/25), Day 2 = antibody session (04/18).
    """
    days = [('Day 1 (05/25)', 'Streptavidin session (05/25)'), ('Day 2 (04/18)', 'Antibody session (04/18)')]
    cols = [('Uncoated origami', 'NoCoating', None), ('PLL-PEG origami', 'PLL', None),
            ('PLL-PEG only (no origami)', None, 'PLL_PEG_NoOrigami.fcs'), ('Silica origami', 'Silica', None),
            ('Silica precursor only (no origami)', None, 'SilicaChemicalsOnly.fcs')]
    ctrls = {f: scat(L('coatings_controls', f)) for _, c, f in cols if f}
    rows = []
    fig, axes = plt.subplots(len(days), len(cols), figsize=(5.2 * len(cols), 10.5))
    for i, (day, sname) in enumerate(days):
        s = SESSIONS[sname]
        bufall = L(s['folder'], 'Buffer.fcs'); buf = scat(bufall); tb = acq_time(bufall)
        unc = pd.concat([scat(d) for r in REPLICATES if (d := L(s['folder'], f"N_NoCoating_{s['noprobe']}_{r}.fcs")) is not None])
        ss_cut = float(np.percentile(unc[CH_SS], 95)); v_cut = float(np.median(unc[CH_ATTO]))
        quad = lambda z: {('High', 'Low'): ((z[CH_ATTO] > v_cut) & (z[CH_SS] <= ss_cut)).sum(),
                          ('High', 'High'): ((z[CH_ATTO] > v_cut) & (z[CH_SS] > ss_cut)).sum(),
                          ('Low', 'Low'): ((z[CH_ATTO] <= v_cut) & (z[CH_SS] <= ss_cut)).sum(),
                          ('Low', 'High'): ((z[CH_ATTO] <= v_cut) & (z[CH_SS] > ss_cut)).sum()}
        for j, (label, coat, cfile) in enumerate(cols):
            if coat:
                ds = [d for r in REPLICATES if (d := L(s['folder'], f"N_{coat}_{s['noprobe']}_{r}.fcs")) is not None]
                z = pd.concat([scat(d) for d in ds]); t = sum(acq_time(d) for d in ds)
                qs, qb = quad(z), quad(buf)
                net = {k: qs[k] / t - qb[k] / tb for k in qs}
                note = 'buffer-subtracted, replicates pooled'
            else:
                z = ctrls[cfile]; t = acq_time(L('coatings_controls', cfile))
                net = {k: v / t for k, v in quad(z).items()}
                note = 'raw (02/13 acquisition, no same-day buffer)'
            tot = sum(net.values())
            for (fl, sc), v in net.items():
                rows.append(dict(day=day, session=sname, sample=label, fluorescence=fl, scatter=sc,
                                 events_per_s=v, pct_of_events=v / tot * 100 if tot > 0 else np.nan,
                                 ss_cut=ss_cut, v525_cut=v_cut, note=note))
            M = pd.DataFrame([[net[('High', 'Low')], net[('High', 'High')]], [net[('Low', 'Low')], net[('Low', 'High')]]],
                             index=['High\nATTO 425', 'Low\nATTO 425'], columns=['Low\nscatter', 'High\nscatter'])
            P = M / tot * 100
            ax = axes[i, j]
            annot = np.array([[f'{P.iat[a, b]:.1f}%\n({M.iat[a, b]:.0f}/s)' for b in range(2)] for a in range(2)])
            sns.heatmap(P.clip(lower=0), annot=annot, fmt='', cmap='magma_r', vmin=0, vmax=100, cbar=False,
                        linewidths=2, linecolor='white', annot_kws={'fontsize': 15}, ax=ax)
            ax.set_title(f'{label}\n{day}' + ('' if coat else '\n(control classified with this day\'s cut-offs)'),
                         fontsize=13)
            ax.tick_params(axis='both', labelsize=12, rotation=0)
        stat('fig22', 'quadrant cut-offs', day, np.nan,
             note=f'high scatter: VSSC-1-H > {ss_cut:.0f} (uncoated P95); high fluorescence: V525-H > {v_cut:.1f} (uncoated median)')
    df = pd.DataFrame(rows)
    for r in df[df.fluorescence.eq('Low') & df.scatter.eq('High')].itertuples():
        stat('fig22', '% of events with high scatter and low ATTO 425', f'{r.day}, {r.sample}', r.pct_of_events, note=r.note)
    fig.suptitle('Side scatter x ATTO 425 quadrants (scatter-triggered events; cut-offs from same-day uncoated origami:\n'
                 'scatter > uncoated P95, fluorescence > uncoated median; origami buffer-subtracted, controls raw)', y=1.04)
    save(fig, 'fig22_scatter_fluorescence_quadrants', {'': df}, subdir=SILICA_DIR)
    copy_into(SILICA_DIR, PLL_DIR, 'fig22_scatter_fluorescence_quadrants')


# ---------------------------------------------------------------- fig23
def fig23():
    """Side scatter of fluorescent, non-aggregate particles (ATTO 425 window), per coating and day.

    Window (scatter-triggered events, per day):
      V525-H > P95 of the same-day buffer        ("above buffer")
      V525-H <= P99 of same-day uncoated origami ("below aggregation": brighter than 99% of single origami)
    No side-scatter limit. Per replicate: buffer-subtracted median VSSC-1-H of window events, and the % of
    window events above the uncoated-origami P95 side scatter (high-scatter fraction).
    Coating-only controls (02/13, raw) are passed through the same window. Expected coating-derived
    high-scatter window events = excess scatter-triggered events/s over uncoated origami (same day) x the
    fraction of the coating-only control that is in the window and above the high-scatter cut.
    """
    days = [('Day 1 (05/25)', 'Streptavidin session (05/25)'), ('Day 2 (04/18)', 'Antibody session (04/18)')]
    ctrl_files = {'PLL': ('PLL_PEG_NoOrigami.fcs', 'PLL-PEG only'), 'Silica': ('SilicaChemicalsOnly.fcs', 'Silica precursor only')}
    ctrls = {c: scat(L('coatings_controls', f)) for c, (f, _) in ctrl_files.items()}
    rows, crow, ptab = [], [], []
    for day, sname in days:
        s = SESSIONS[sname]
        bufall = L(s['folder'], 'Buffer.fcs'); buf = scat(bufall); tb = acq_time(bufall)
        files = {c: [(r, d) for r in REPLICATES if (d := L(s['folder'], f"N_{c}_{s['noprobe']}_{r}.fcs")) is not None]
                 for c in COATINGS}
        unc = pd.concat([scat(d) for _, d in files['NoCoating']])
        lo = float(np.percentile(buf[CH_ATTO], 95)); hi = float(np.percentile(unc[CH_ATTO], 99))
        ss_hi = float(np.percentile(unc[CH_SS], 95))
        win = lambda z: z[(z[CH_ATTO] > lo) & (z[CH_ATTO] <= hi)]
        bw = win(buf)
        unc_eps = np.mean([len(scat(d)) / acq_time(d) for _, d in files['NoCoating']])
        cfrac = {}
        for c, z in ctrls.items():
            w = win(z)
            cfrac[c] = ((z[CH_ATTO] > lo) & (z[CH_ATTO] <= hi) & (z[CH_SS] > ss_hi)).mean()
            crow.append(dict(day=day, control=ctrl_files[c][1], pct_in_window=len(w) / len(z) * 100,
                             median_SS_in_window=float(np.median(w[CH_SS])),
                             pct_of_window_events_high_SS=float((w[CH_SS] > ss_hi).mean() * 100),
                             fraction_in_window_and_high_SS=cfrac[c], note='raw, 02/13 acquisition'))
        for c in COATINGS:
            for r, d in files[c]:
                x = scat(d); t = acq_time(d); w = win(x)
                med = bs_percentile(w[CH_SS].values, t, bw[CH_SS].values, tb, 50)
                net, above = bs_counts(w[CH_SS].values, t, bw[CH_SS].values, tb, ss_hi)
                excess = max(len(x) / t - unc_eps, 0) if c != 'NoCoating' else np.nan
                rows.append(dict(day=day, coating=COAT_LBL[c], replicate=r, window_low_V525=lo, window_high_V525=hi,
                                 high_SS_cut=ss_hi, pct_events_in_window=len(w) / len(x) * 100,
                                 median_SS_window_buffer_subtracted=med,
                                 pct_window_events_high_SS=above / net * 100 if net > 0 else np.nan,
                                 window_high_SS_events_per_s=above / t,
                                 excess_scatter_triggered_eps_over_uncoated=excess,
                                 expected_coating_derived_window_high_SS_eps=excess * cfrac[c] if c in cfrac else np.nan))
        stat('fig23', 'ATTO window and high-scatter cut', day, np.nan,
             note=f'{lo:.1f} < V525-H <= {hi:.1f} (buffer P95, uncoated P99); high SS: VSSC-1-H > {ss_hi:.0f} (uncoated P95)')
    df = pd.DataFrame(rows); C = pd.DataFrame(crow)
    for day, _ in days:
        q = df[df.day == day]
        for col in ['median_SS_window_buffer_subtracted', 'pct_window_events_high_SS']:
            u = q[q.coating == 'Uncoated'][col]
            for c in ['PLL-PEG', 'Silica']:
                v = q[q.coating == c][col]
                t = ttest_ind(v, u, equal_var=False)
                ptab.append(dict(day=day, metric=col, comparison=f'{c} vs Uncoated', coated_mean=v.mean(),
                                 uncoated_mean=u.mean(), p=t.pvalue, n=len(v) + len(u)))
                stat('fig23', f'Welch t-test, {col}, {c} vs Uncoated', day, float(v.mean() - u.mean()), t.pvalue,
                     len(v) + len(u), f'value = {c} minus Uncoated')
    P = pd.DataFrame(ptab)
    order = [COAT_LBL[c] for c in COATINGS]; pal = {COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    for i, (day, _) in enumerate(days):
        q = df[df.day == day]
        for j, (col, ylabel) in enumerate([('median_SS_window_buffer_subtracted', 'Median VSSC-1-H (a.u.)'),
                                           ('pct_window_events_high_SS', f'% above high-scatter cut')]):
            ax = axes[i, j]
            sns.barplot(data=q, x='coating', y=col, hue='coating', order=order, palette=pal, errorbar='se',
                        capsize=0.15, edgecolor='black', legend=False, ax=ax)
            sns.stripplot(data=q, x='coating', y=col, order=order, color='black', size=8, ax=ax)
            top = q[col].max()
            for k, c in enumerate(['PLL-PEG', 'Silica']):
                pr = P[(P.day == day) & (P.metric == col) & (P.comparison == f'{c} vs Uncoated')].p.iloc[0]
                x1 = order.index(c); y = top * (1.08 + 0.13 * k)
                ax.plot([0, 0, x1, x1], [y * 0.98, y, y, y * 0.98], color='black', lw=1.2)
                ax.text(x1 / 2, y, f'p = {pr:.3f}', ha='center', va='bottom', fontsize=12)
            ax.set_ylim(top=top * 1.45)
            ax.set_xlabel(''); ax.set_ylabel(ylabel)
            cut = q.high_SS_cut.iloc[0]
            ax.set_title(f'{day}: ' + ('median side scatter in ATTO window' if j == 0 else
                                       f'high-scatter fraction (VSSC-1-H > {cut:.0f})'), fontsize=14)
        ax = axes[i, 2]
        si = pd.concat([
            q[q.coating == 'Uncoated'][['replicate', 'window_high_SS_events_per_s']].rename(
                columns={'window_high_SS_events_per_s': 'eps'}).assign(quantity='Uncoated origami\n(observed)'),
            q[q.coating == 'Silica'][['replicate', 'window_high_SS_events_per_s']].rename(
                columns={'window_high_SS_events_per_s': 'eps'}).assign(quantity='Silica origami\n(observed)'),
            q[q.coating == 'Silica'][['replicate', 'expected_coating_derived_window_high_SS_eps']].rename(
                columns={'expected_coating_derived_window_high_SS_eps': 'eps'}).assign(
                quantity='Expected from silica\nexcess particles')])
        sns.barplot(data=si, x='quantity', y='eps', hue='quantity',
                    palette=[COAT_COLORS['NoCoating'], COAT_COLORS['Silica'], '0.7'],
                    errorbar='se', capsize=0.15, edgecolor='black', legend=False, ax=ax)
        sns.stripplot(data=si, x='quantity', y='eps', color='black', size=8, ax=ax)
        ax.set_xlabel(''); ax.set_ylabel('High-scatter window events per second')
        ax.set_title(f'{day}: silica, observed vs precursor-expected', fontsize=14)
    lo1, hi1 = df.window_low_V525.unique(), df.window_high_V525.unique()
    fig.suptitle('Side scatter of particles with ATTO 425 above buffer but below aggregation '
                 f'(V525-H > same-day buffer P95 and <= uncoated P99); buffer-subtracted, scatter-triggered events', y=1.02)
    save(fig, 'fig23_side_scatter_fluorescent_nonaggregate', {'': df, '_stats': P, '_controls_through_window': C},
         subdir=SILICA_DIR)
    copy_into(SILICA_DIR, PLL_DIR, 'fig23_side_scatter_fluorescent_nonaggregate')


# ---------------------------------------------------------------- fig24
def fig24():
    """Free-antibody solution effect on B525-H (background fluorescence / coincidence).

    A  B525-H of scatter-triggered particles with VSSC-1-H 200-260 (no origami) in buffer vs antibody-only
       solution (both 03/13, identical trigger settings); medians marked.
    B  B525-H quantile shifts caused by antibody: non-origami particles (antibody-only minus buffer, SS 200-260)
       and no-biotin origami (with minus without antibody, origami gate, same coating and replicate, 04/18).
    C  Median B525-H shift of no-biotin origami (log2 with/without antibody) vs the free-antibody event rate
       of the with-antibody sample (fluorescence-triggered events: VSSC-1-H < 200 and B525-H >= 250);
       Pearson correlation across samples.
    Raw quantiles (no buffer subtraction).
    """
    from scipy.stats import pearsonr, spearmanr
    Q = [5, 25, 50, 75, 95]
    win = lambda d: d[(d[CH_SS] >= SS_TRIGGER) & (d[CH_SS] < 260)]
    bu = L('probe_only_controls', 'Buffer__Shervin03_13_2026.fcs')
    ab = L('probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs')
    vb, va = win(bu)[CH_S].values, win(ab)[CH_S].values
    qb, qa = np.percentile(vb, Q), np.percentile(va, Q)
    allb, alla = scat(bu)[CH_S].values, scat(ab)[CH_S].values
    ctrl = pd.DataFrame([dict(sample='Buffer (03/13)', events='SS 200-260', n=len(vb), **{f'P{q}': v for q, v in zip(Q, qb)}),
                         dict(sample='Antibody only (03/13)', events='SS 200-260', n=len(va), **{f'P{q}': v for q, v in zip(Q, qa)}),
                         dict(sample='Buffer (03/13)', events='SS >= 200', n=len(allb), **{f'P{q}': v for q, v in zip(Q, np.percentile(allb, Q))}),
                         dict(sample='Antibody only (03/13)', events='SS >= 200', n=len(alla), **{f'P{q}': v for q, v in zip(Q, np.percentile(alla, Q))})])
    s4 = SESSIONS['Antibody session (04/18)']
    dfs = session_dfs(s4)
    gates = compute_monomer_gates(dfs, ctl=s4['noprobe'])
    rows = []
    for c in COATINGS:
        for r in REPLICATES:
            d0 = dfs.get(('N', c, s4['noprobe'], r)); d1 = dfs.get(('N', c, s4['probe'], r))
            if d0 is None or d1 is None:
                continue
            x0 = apply_gate(d0, c, r, gates)[CH_S].values; x1 = apply_gate(d1, c, r, gates)[CH_S].values
            q0, q1 = np.percentile(x0, Q), np.percentile(x1, Q)
            t1 = acq_time(d1)
            rows.append(dict(coating=COAT_LBL[c], replicate=r,
                             free_antibody_events_per_s=((d1[CH_SS] < SS_TRIGGER) & (d1[CH_S] >= 250)).sum() / t1,
                             median_no_antibody=q0[2], median_with_antibody=q1[2], log2_median_shift=np.log2(q1[2] / q0[2]),
                             pct_above_no_antibody_P99=(x1 > np.percentile(x0, 99)).mean() * 100,
                             **{f'shift_P{q}': b - a for q, a, b in zip(Q, q0, q1)}))
    df = pd.DataFrame(rows)
    corr = []
    for y in ['log2_median_shift', 'pct_above_no_antibody_P99', 'shift_P50']:
        pr = pearsonr(df.free_antibody_events_per_s, df[y]); sr = spearmanr(df.free_antibody_events_per_s, df[y])
        corr.append(dict(y=y, x='free_antibody_events_per_s', pearson_r=pr.statistic, pearson_p=pr.pvalue,
                         spearman_rho=sr.statistic, spearman_p=sr.pvalue, n=len(df)))
        stat('fig24', f'correlation of {y} with free-antibody event rate', 'no-biotin origami + antibody (04/18)',
             pr.statistic, pr.pvalue, len(df), f'Spearman rho {sr.statistic:.2f}, p {sr.pvalue:.3f}')
    stat('fig24', 'median B525-H, SS 200-260 particles, antibody-only vs buffer (03/13)', 'non-origami particles',
         qa[2] - qb[2], note=f'{qa[2]:.0f} vs {qb[2]:.0f}')
    Cr = pd.DataFrame(corr)
    fig, axes = plt.subplots(1, 3, figsize=(25, 7))
    # A
    ax = axes[0]
    bins = np.logspace(0, 3.3, 100); xc = np.sqrt(bins[:-1] * bins[1:])
    hist = []
    for name, v, colr in [('Buffer (03/13)', vb, BUF_COL), ('Antibody only (03/13)', va, '#E6862E')]:
        h = log_density(v[v > 0], bins)
        ax.step(xc, h, where='mid', color=colr, lw=2.5, label=f'{name}: median {np.median(v):.0f}')
        ax.axvline(np.median(v), color=colr, ls='--', lw=2)
        hist += [dict(sample=name, b525_bin_center=a, density=b) for a, b in zip(xc, h)]
    ax.set_xscale('log'); ax.set_xlabel('B525-H (a.u.)'); ax.set_ylabel('Density (per log$_{10}$ unit)')
    ax.set_title('A  Particles with no origami (VSSC-1-H 200-260)', fontsize=14); ax.legend(fontsize=12, frameon=False)
    # B
    ax = axes[1]
    xs = np.arange(len(Q))
    for _, r in df.iterrows():
        ax.plot(xs, [r[f'shift_P{q}'] for q in Q], 'o-', color=COAT_COLORS[[c for c in COATINGS if COAT_LBL[c] == r.coating][0]],
                alpha=0.6, lw=1.5)
    for c in COATINGS:
        ax.plot([], [], 'o-', color=COAT_COLORS[c], label=f'No-biotin {COAT_LBL[c]} origami (each replicate)')
    ax.plot(xs, qa - qb, 's-', color='black', lw=3, ms=10, label='Non-origami particles (antibody only - buffer)')
    ax.set_xticks(xs); ax.set_xticklabels([f'P{q}' for q in Q]); ax.axhline(0, color='0.6', lw=1)
    ax.set_xlabel('B525-H quantile'); ax.set_ylabel('Shift in B525-H with antibody (a.u.)')
    ax.set_title('B  Quantile shifts caused by antibody', fontsize=14); ax.legend(fontsize=10, frameon=False)
    # C
    ax = axes[2]
    for c in COATINGS:
        q = df[df.coating == COAT_LBL[c]]
        ax.scatter(q.free_antibody_events_per_s, q.log2_median_shift, s=110, color=COAT_COLORS[c], edgecolor='black',
                   label=COAT_LBL[c], zorder=3)
    lr = linregress(df.free_antibody_events_per_s, df.log2_median_shift)
    xx = np.linspace(df.free_antibody_events_per_s.min(), df.free_antibody_events_per_s.max(), 50)
    ax.plot(xx, lr.intercept + lr.slope * xx, color='black', ls='--', lw=1.5)
    c0 = Cr[Cr.y == 'log2_median_shift'].iloc[0]
    ax.text(0.04, 0.95, f'Pearson r = {c0.pearson_r:.2f} (p = {c0.pearson_p:.1e})\nn = {len(df)} samples',
            transform=ax.transAxes, va='top', fontsize=13)
    ax.set_xlabel('Free-antibody events/s in sample\n(VSSC-1-H < 200, B525-H ≥ 250)')
    ax.set_ylabel('log$_2$(median B525-H with / without antibody)')
    ax.set_title('C  Shift scales with free antibody (no-biotin origami, 04/18)', fontsize=14)
    ax.legend(fontsize=12, frameon=False, loc='lower right')
    fig.suptitle('Antibody-associated B525-H increase: non-origami particles in antibody solution and no-biotin origami '
                 '(raw quantiles, no buffer subtraction)', y=1.03)
    save(fig, 'fig24_antibody_solution_effect', {'_samples': df, '_controls_quantiles': ctrl,
                                                  '_correlations': Cr, '_histograms': pd.DataFrame(hist)})


# ---------------------------------------------------------------- fig25
def acq_record(folder, fname):
    """Acquisition record (NANOFLOWINFO JSON in the FCS ANALYSIS segment)."""
    import re, json
    from fcsparser import parse
    path = os.path.join(PROJECT_ROOT, folder, fname)
    meta, _ = parse(path, reformat_meta=True)
    raw = open(path, 'rb').read()[int(meta['$BEGINANALYSIS']):int(meta['$ENDANALYSIS']) + 1].decode('latin1')
    return json.loads(bytes.fromhex(re.search(r'NANOFLOWINFO\n([0-9A-Fa-f]+)', raw).group(1)).decode('utf8'))['Acquisition']


def fig25():
    """Antibody session (04/18): what the biotin-extension signal co-varies with.

    Per with-antibody sample (all extensions, all coatings, replicates A-C):
      fluorescence-triggered events  VSSC-1-H < 200 and B525-H >= 250 (rate, median B525-H, median V525-H)
      gated origami                  origami gate (from no-antibody controls): median and P5 of B525-H
      acquisition                    start time and order within each coating/replicate set (from FCS record)
    A  M1 (log2 median ext / N, same set) vs log2 ratio of fluorescence-triggered event rate (ext / N)
    B  Median B525-H of fluorescence-triggered events vs their rate (all samples)
    C  P5 of gated-origami B525-H vs fluorescence-triggered rate (dimmest origami)
    D  Median V525-H of fluorescence-triggered events vs their rate (ATTO among them)
    E  Gated-origami median B525-H by acquisition order within each set (extension order N, 3, 6, 12)
    F  Gated-origami median B525-H vs time in session
    Raw values (no buffer subtraction). Equal antibody added to every sample.
    """
    from scipy.stats import pearsonr, spearmanr
    s4 = SESSIONS['Antibody session (04/18)']
    dfs = session_dfs(s4)
    gates = compute_monomer_gates(dfs, ctl=s4['noprobe'])
    rows = []
    for c in COATINGS:
        for r in REPLICATES:
            for e in EXTENSIONS:
                f = f'{e}_{c}_AB_{r}.fcs'
                d = L(s4['folder'], f)
                if d is None:
                    continue
                t = acq_time(d); a = acq_record(s4['folder'], f)
                sub = d[(d[CH_SS] < SS_TRIGGER) & (d[CH_S] >= 250)]
                gx = apply_gate(d, c, r, gates)[CH_S].values; gx = gx[gx > 0]
                rows.append(dict(coating=COAT_LBL[c], replicate=r, extension=e, file=f,
                                 start_time=pd.Timestamp(a['StartTime']),
                                 fluor_triggered_events_per_s=len(sub) / t,
                                 fluor_triggered_median_B525=float(sub[CH_S].median()),
                                 fluor_triggered_median_V525=float(sub[CH_ATTO].median()),
                                 gated_median_B525=float(np.median(gx)), gated_P5_B525=float(np.percentile(gx, 5))))
    df = pd.DataFrame(rows)
    df['minutes_in_session'] = (df.start_time - df.start_time.min()).dt.total_seconds() / 60
    df['order_in_set'] = df.groupby(['coating', 'replicate']).start_time.rank().astype(int)
    base = df[df.extension == 'N'].set_index(['coating', 'replicate'])
    df['M1_log2_median_vs_N'] = [np.log2(m / base.loc[(c, r), 'gated_median_B525']) if (c, r) in base.index else np.nan
                                 for m, c, r in zip(df.gated_median_B525, df.coating, df.replicate)]
    df['log2_fluor_rate_vs_N'] = [np.log2(v / base.loc[(c, r), 'fluor_triggered_events_per_s']) if (c, r) in base.index else np.nan
                                  for v, c, r in zip(df.fluor_triggered_events_per_s, df.coating, df.replicate)]
    ext = df[df.extension != 'N']
    pairs = [('A', ext, 'log2_fluor_rate_vs_N', 'M1_log2_median_vs_N'),
             ('B', df, 'fluor_triggered_events_per_s', 'fluor_triggered_median_B525'),
             ('C', df, 'fluor_triggered_events_per_s', 'gated_P5_B525'),
             ('D', df, 'fluor_triggered_events_per_s', 'fluor_triggered_median_V525'),
             ('F', df, 'minutes_in_session', 'gated_median_B525')]
    corr = []
    for panel, q, x, y in pairs:
        pr = pearsonr(q[x], q[y]); sr = spearmanr(q[x], q[y])
        corr.append(dict(panel=panel, x=x, y=y, n=len(q), pearson_r=pr.statistic, pearson_p=pr.pvalue,
                         spearman_rho=sr.statistic, spearman_p=sr.pvalue))
        stat('fig25', f'correlation {y} vs {x}', f'panel {panel}', pr.statistic, pr.pvalue, len(q),
             f'Spearman rho {sr.statistic:.2f}')
    Cr = pd.DataFrame(corr)
    order_tab = df.groupby('order_in_set').agg(extension=('extension', lambda z: ','.join(sorted(set(z)))),
                                               mean_gated_median_B525=('gated_median_B525', 'mean'),
                                               mean_fluor_rate=('fluor_triggered_events_per_s', 'mean'),
                                               n=('file', 'count')).reset_index()
    ext_col = {'N': '#4D4D4D', '3': '#377EB8', '6': '#4DAF4A', '12': '#E41A1C'}
    fig, axes = plt.subplots(2, 3, figsize=(25, 14))
    labels = {'A': ('log$_2$(fluorescence-triggered rate, extension / N)', 'M1: log$_2$(median B525-H, extension / N)'),
              'B': ('Fluorescence-triggered events/s', 'Median B525-H of fluorescence-triggered events'),
              'C': ('Fluorescence-triggered events/s', 'P5 B525-H of gated origami (dimmest origami)'),
              'D': ('Fluorescence-triggered events/s', 'Median V525-H of fluorescence-triggered events'),
              'F': ('Minutes into the session', 'Median B525-H of gated origami')}
    titles = {'A': 'A  Biotin signal vs change in fluorescence-triggered events',
              'B': 'B  Brightness of fluorescence-triggered events',
              'C': 'C  Dimmest origami also brighten',
              'D': 'D  ATTO signal among fluorescence-triggered events',
              'F': 'F  Drift across the session'}
    pos = {'A': (0, 0), 'B': (0, 1), 'C': (0, 2), 'D': (1, 0), 'F': (1, 2)}
    for panel, q, x, y in pairs:
        ax = axes[pos[panel]]
        for e in EXTENSIONS:
            z = q[q.extension == e]
            if len(z):
                ax.scatter(z[x], z[y], s=90, color=ext_col[e], edgecolor='black', label=f'{e} nt' if e != 'N' else 'N (no biotin)', zorder=3)
        lr = linregress(q[x], q[y]); xx = np.linspace(q[x].min(), q[x].max(), 50)
        ax.plot(xx, lr.intercept + lr.slope * xx, 'k--', lw=1.3)
        cr = Cr[Cr.panel == panel].iloc[0]
        ax.text(0.03, 0.96, f'r = {cr.pearson_r:.2f} (p = {cr.pearson_p:.1e}), n = {cr.n}', transform=ax.transAxes,
                va='top', fontsize=13)
        if panel == 'A':
            ax.axhline(0, color='0.6', lw=1); ax.axvline(0, color='0.6', lw=1)
            ax.text(0.03, 0.88, f'M1 at equal rate (intercept): {lr.intercept:+.3f} ± {lr.intercept_stderr:.3f}',
                    transform=ax.transAxes, va='top', fontsize=12)
        ax.set_xlabel(labels[panel][0]); ax.set_ylabel(labels[panel][1]); ax.set_title(titles[panel], fontsize=15)
        ax.legend(fontsize=10, frameon=False, loc='lower right')
    ax = axes[1, 1]
    for (c, r), z in df.groupby(['coating', 'replicate']):
        z = z.sort_values('order_in_set')
        ax.plot(z.order_in_set, z.gated_median_B525, '-', color=COAT_COLORS[[k for k in COATINGS if COAT_LBL[k] == c][0]],
                alpha=0.6, lw=1.5)
        for _, rr in z.iterrows():
            ax.scatter(rr.order_in_set, rr.gated_median_B525, s=70, color=ext_col[rr.extension], edgecolor='black', zorder=3)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(['1st', '2nd', '3rd', '4th'])
    ax.set_xlabel('Acquisition order within each coating/replicate set\n(always N, 3, 6, 12 nt; 4-5 min apart)')
    ax.set_ylabel('Median B525-H of gated origami')
    ax.set_title('E  Extension is confounded with acquisition order', fontsize=15)
    for c in COATINGS:
        ax.plot([], [], '-', color=COAT_COLORS[c], label=f'{COAT_LBL[c]} set')
    ax.legend(fontsize=10, frameon=False)
    fig.suptitle('Antibody session (04/18), equal antibody added to every sample: the biotin-extension signal co-varies with '
                 'the number and brightness of fluorescence-triggered events and with acquisition order (raw values)', y=1.02)
    save(fig, 'fig25_antibody_signal_covariation_and_order', {'_samples': df, '_correlations': Cr,
                                                              '_by_acquisition_order': order_tab})


# ---------------------------------------------------------------- fig26
def fig26():
    """Brightness of free probe in the low-scatter regime (VSSC-1-H < 50) vs the origami signal, per sample.

    Low-scatter fluorescence-triggered events (VSSC-1-H < 50, B525-H >= the session B525 trigger: 250 antibody,
    150 streptavidin) are free probe or noise; origami, even below the scatter trigger, sit at higher side
    scatter (V525-H of these events is buffer-like). If their median B525-H moves together with the
    gated-origami signal, the change is sample-wide (brightness/sensitivity/background), not binding.
    Per with-probe sample (all coatings, extensions, replicates):
      low_SS_median_B525, low_SS_events_per_s, low_SS_median_V525; gated-origami median B525-H (origami gate);
      M1 = log2(gated median ext / N, same set); change in low-SS median (ext - N).
    Rows: antibody session (04/18), streptavidin session (05/25). Raw values (no buffer subtraction).
    """
    from scipy.stats import pearsonr, spearmanr
    LOW_SS = 50
    sess = [('Antibody session (04/18)', 250, 'probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs', 'Antibody only (03/13)'),
            ('Streptavidin session (05/25)', 150, 'probe_only_controls', 'Streptavidin_Only__Shervin5_03.fcs', 'Streptavidin only (05/03)')]
    ext_col = {'N': '#4D4D4D', '3': '#377EB8', '6': '#4DAF4A', '12': '#E41A1C'}
    rows, refs = [], []
    for sname, trig, rf, rfile, rlabel in sess:
        s_ = SESSIONS[sname]
        dfs = session_dfs(s_)
        gates = compute_monomer_gates(dfs, ctl=s_['noprobe'])
        rd = L(rf, rfile); rl = rd[(rd[CH_SS] < LOW_SS) & (rd[CH_S] >= trig)]
        refs.append(dict(session=sname, reference=rlabel, low_SS_median_B525=float(rl[CH_S].median()),
                         low_SS_median_V525=float(rl[CH_ATTO].median()), low_SS_events_per_s=len(rl) / acq_time(rd),
                         note='probe-only control from another session; events with B525-H >= this session trigger'))
        bu = dfs['Buffer']; bl = bu[(bu[CH_SS] < LOW_SS) & (bu[CH_S] >= trig)]
        refs.append(dict(session=sname, reference='Buffer (same session)', low_SS_median_B525=float(bl[CH_S].median()) if len(bl) else np.nan,
                         low_SS_median_V525=float(bl[CH_ATTO].median()) if len(bl) else np.nan,
                         low_SS_events_per_s=len(bl) / acq_time(bu), note=''))
        for c in COATINGS:
            for r in REPLICATES:
                for e in EXTENSIONS:
                    d = L(s_['folder'], f"{e}_{c}_{s_['probe']}_{r}.fcs")
                    if d is None:
                        continue
                    t = acq_time(d); lo = d[(d[CH_SS] < LOW_SS) & (d[CH_S] >= trig)]
                    gx = apply_gate(d, c, r, gates)[CH_S].values; gx = gx[gx > 0]
                    rows.append(dict(session=sname, coating=COAT_LBL[c], replicate=r, extension=e,
                                     low_SS_events_per_s=len(lo) / t, low_SS_median_B525=float(lo[CH_S].median()),
                                     low_SS_median_V525=float(lo[CH_ATTO].median()), gated_median_B525=float(np.median(gx))))
    df = pd.DataFrame(rows); R = pd.DataFrame(refs)
    key = ['session', 'coating', 'replicate']
    base = df[df.extension == 'N'].set_index(key)
    df['M1_log2_gated_vs_N'] = [np.log2(m / base.loc[k, 'gated_median_B525']) if k in base.index else np.nan
                                for m, k in zip(df.gated_median_B525, zip(df.session, df.coating, df.replicate))]
    df['change_low_SS_median_vs_N'] = [m - base.loc[k, 'low_SS_median_B525'] if k in base.index else np.nan
                                       for m, k in zip(df.low_SS_median_B525, zip(df.session, df.coating, df.replicate))]
    corr = []
    for sname, *_ in sess:
        q = df[df.session == sname]; x = q[q.extension != 'N'].dropna(subset=['M1_log2_gated_vs_N', 'change_low_SS_median_vs_N'])
        for nm, a, b, z in [('low-SS median vs gated median (all samples)', 'low_SS_median_B525', 'gated_median_B525', q),
                            ('change in low-SS median vs M1 (extension samples)', 'change_low_SS_median_vs_N', 'M1_log2_gated_vs_N', x)]:
            pr = pearsonr(z[a], z[b]); sr = spearmanr(z[a], z[b])
            corr.append(dict(session=sname, comparison=nm, n=len(z), pearson_r=pr.statistic, pearson_p=pr.pvalue,
                             spearman_rho=sr.statistic, spearman_p=sr.pvalue))
            stat('fig26', nm, sname, pr.statistic, pr.pvalue, len(z), f'Spearman rho {sr.statistic:.2f}')
    Cr = pd.DataFrame(corr)
    fig, axes = plt.subplots(2, 3, figsize=(25, 14))
    for row, (sname, trig, *_r) in enumerate(sess):
        q = df[df.session == sname]; x = q[q.extension != 'N']
        for col, (data, a, b, xl, yl, cname) in enumerate([
                (q, 'low_SS_median_B525', 'gated_median_B525', f'Median B525-H, low-scatter events (VSSC-1-H < {LOW_SS})',
                 'Median B525-H, gated origami', 'low-SS median vs gated median (all samples)'),
                (x, 'change_low_SS_median_vs_N', 'M1_log2_gated_vs_N', 'Change in low-scatter median B525-H (extension - N)',
                 'M1: log$_2$(gated median, extension / N)', 'change in low-SS median vs M1 (extension samples)')]):
            ax = axes[row, col]
            for e in EXTENSIONS:
                z = data[data.extension == e]
                if len(z):
                    ax.scatter(z[a], z[b], s=90, color=ext_col[e], edgecolor='black', zorder=3,
                               label='N (no biotin)' if e == 'N' else f'{e} nt')
            for _, rr in data[(data.coating == 'Silica') & (data.extension == '6')].iterrows():
                ax.annotate(f'Silica 6 nt {rr.replicate}', (rr[a], rr[b]), xytext=(6, -12), textcoords='offset points', fontsize=9)
            cr = Cr[(Cr.session == sname) & (Cr.comparison == cname)].iloc[0]
            lr = linregress(data[a], data[b]); xx = np.linspace(data[a].min(), data[a].max(), 50)
            ax.plot(xx, lr.intercept + lr.slope * xx, 'k--', lw=1.2)
            ax.text(0.03, 0.96, f'r = {cr.pearson_r:.2f} (p = {cr.pearson_p:.1e}), n = {cr.n}', transform=ax.transAxes,
                    va='top', fontsize=13)
            if col == 0:
                ref = R[(R.session == sname) & (R.reference != 'Buffer (same session)')].iloc[0]
                ax.axvline(ref.low_SS_median_B525, color='#E6862E', ls=':', lw=2, label=f'{ref.reference}: {ref.low_SS_median_B525:.0f}')
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=10, frameon=False, loc='lower right')
            ax.set_title(f"{'AB'[col]}{row + 1}  {sname}", fontsize=14)
        ax = axes[row, 2]
        order = [f'{e}' for e in EXTENSIONS]
        sns.stripplot(data=q, x='extension', y='low_SS_events_per_s', hue='coating', order=order,
                      palette={COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}, dodge=True, size=9, ax=ax)
        ax2 = ax.twinx()
        sns.pointplot(data=q, x='extension', y='low_SS_median_V525', order=order, color='0.3', errorbar='se',
                      markers='s', linestyles=':', ax=ax2)
        ax2.set_ylabel('Median V525-H of low-scatter events (squares)', color='0.3')
        ax2.set_ylim(40, 70)
        ax.set_xlabel('Biotin extension'); ax.set_ylabel(f'Low-scatter fluorescence-triggered events/s (circles)')
        ax.set_title(f'C{row + 1}  {sname}: low-scatter events and their ATTO signal', fontsize=14)
        ax.legend(fontsize=10, frameon=False, title='', loc='upper left')
    fig.suptitle(f'Free-probe brightness at very low side scatter (VSSC-1-H < {LOW_SS}, no origami) vs the origami signal '
                 '(raw values; equal probe added to every sample)', y=1.02)
    save(fig, 'fig26_low_scatter_probe_brightness', {'_samples': df, '_correlations': Cr, '_references': R})


# ---------------------------------------------------------------- fig27
def fig27():
    """Percent conjugation vs sample-wide antibody brightness, and a brightness-calibrated threshold.

    Free-antibody brightness of a sample = median B525-H of its fluorescence-triggered events at very low
    side scatter (VSSC-1-H < 50, B525-H >= 250; no origami there). Change = extension sample - no-biotin
    sample of the same coating/replicate set.
    A  Original Delta % conjugated (fig13 method: gated + below-trigger unmixing, threshold = no-antibody P99)
       vs change in free-antibody brightness; regression intercept = Delta at no brightness change.
    B  Same, gated part only.
    C  Calibrated % conjugated: gated origami (origami gate, buffer-subtracted) with B525-H above the median
       B525-H of the SAME sample's low-scatter free antibody, i.e. at least as bright as a typical recorded
       free antibody in that acquisition. No-antibody origami scored with each set's no-biotin threshold.
    D  Calibrated Delta % conjugated (extension - no biotin) by extension and coating.
    E  Calibrated Delta vs change in free-antibody brightness (does the dependence disappear?).
    F  Calibrated threshold per sample (free-antibody median is truncated at the 250 trigger, so it moves little).
    Antibody session (04/18).
    """
    from scipy.stats import pearsonr, spearmanr
    from quantify_bound_origami_antibody import quantify, with_delta
    LOW_SS, TRIG = 50, 250
    s4 = SESSIONS['Antibody session (04/18)']
    dfs = session_dfs(s4, exts=tuple(EXTENSIONS))
    gates = compute_monomer_gates(dfs, ctl=s4['noprobe'])
    buf = dfs['Buffer']; tb = acq_time(buf)
    rows = []
    for c in COATINGS:
        bb = apply_gate(buf, c, 'A', gates)[CH_S].values
        for r in REPLICATES:
            nlow = None
            for e in EXTENSIONS:
                d = dfs.get((e, c, s4['probe'], r))
                if d is None:
                    continue
                t = acq_time(d)
                lo = d[(d[CH_SS] < LOW_SS) & (d[CH_S] >= TRIG)]
                thr = float(lo[CH_S].median())
                gx = apply_gate(d, c, r, gates)[CH_S].values
                net, above = bs_counts(gx, t, bb, tb, thr)
                rows.append(dict(coating=COAT_LBL[c], replicate=r, extension=e, free_antibody_median_B525=thr,
                                 calibrated_threshold=thr, net_gated=net, net_above=above,
                                 calibrated_pct_conjugated=above / net * 100 if net > 0 else np.nan))
            d0 = dfs.get(('N', c, s4['noprobe'], r))
            nrow = [q for q in rows if q['coating'] == COAT_LBL[c] and q['replicate'] == r and q['extension'] == 'N']
            if d0 is not None and nrow:
                x0 = apply_gate(d0, c, r, gates)[CH_S].values
                net, above = bs_counts(x0, acq_time(d0), bb, tb, nrow[0]['calibrated_threshold'])
                rows.append(dict(coating=COAT_LBL[c], replicate=r, extension='No antibody',
                                 free_antibody_median_B525=np.nan, calibrated_threshold=nrow[0]['calibrated_threshold'],
                                 net_gated=net, net_above=above, calibrated_pct_conjugated=above / net * 100 if net > 0 else np.nan))
    C = pd.DataFrame(rows)
    key = ['coating', 'replicate']
    base = C[C.extension == 'N'].set_index(key)
    C['change_free_antibody_median_vs_N'] = [m - base.loc[k, 'free_antibody_median_B525'] if k in base.index else np.nan
                                             for m, k in zip(C.free_antibody_median_B525, zip(C.coating, C.replicate))]
    C['calibrated_delta_vs_N'] = [v - base.loc[k, 'calibrated_pct_conjugated'] if k in base.index else np.nan
                                  for v, k in zip(C.calibrated_pct_conjugated, zip(C.coating, C.replicate))]
    # original fig13 % conjugated
    Q = with_delta(quantify(verbose=False))
    Q['coating'] = Q.coating.map(COAT_LBL)
    Nq = Q[Q.ext == 'N'].set_index(['coating', 'rep'])
    Q['delta_gated_only'] = [v - Nq.loc[(c, r), 'pct_bound_hiSS_only'] for v, c, r in zip(Q.pct_bound_hiSS_only, Q.coating, Q.rep)]
    O = Q.merge(C[['coating', 'replicate', 'extension', 'change_free_antibody_median_vs_N', 'free_antibody_median_B525']],
                left_on=['coating', 'rep', 'ext'], right_on=['coating', 'replicate', 'extension'], how='left')
    O = O[O.ext != 'N']
    X = C[C.extension.isin(['3', '6', '12'])].dropna(subset=['calibrated_delta_vs_N', 'change_free_antibody_median_vs_N'])
    fits = []
    for nm, z, xcol, ycol in [('A: original Delta % conjugated', O, 'change_free_antibody_median_vs_N', 'delta_vs_N'),
                              ('B: original Delta, gated part only', O, 'change_free_antibody_median_vs_N', 'delta_gated_only'),
                              ('E: calibrated Delta % conjugated', X, 'change_free_antibody_median_vs_N', 'calibrated_delta_vs_N')]:
        z = z.dropna(subset=[xcol, ycol])
        pr = pearsonr(z[xcol], z[ycol]); sr = spearmanr(z[xcol], z[ycol]); lr = linregress(z[xcol], z[ycol])
        fits.append(dict(panel=nm, n=len(z), pearson_r=pr.statistic, pearson_p=pr.pvalue, spearman_rho=sr.statistic,
                         slope=lr.slope, intercept=lr.intercept, intercept_se=lr.intercept_stderr))
        stat('fig27', f'{nm} vs change in free-antibody brightness', 'antibody session (04/18)', pr.statistic, pr.pvalue,
             len(z), f'intercept (Delta at no brightness change) {lr.intercept:+.2f} +/- {lr.intercept_stderr:.2f}')
    for e in ['3', '6', '12']:
        v = X[X.extension == e].calibrated_delta_vs_N
        t = ttest_1samp(v, 0) if len(v) >= 2 else None
        stat('fig27', 'calibrated Delta % conjugated vs 0 (one-sample t-test)', f'{e} nt, all coatings', v.mean(),
             t.pvalue if t else np.nan, len(v))
    Fz = pd.DataFrame(fits)
    ext_col = {'3': '#377EB8', '6': '#4DAF4A', '12': '#E41A1C'}
    fig, axes = plt.subplots(2, 3, figsize=(25, 14))

    def scatter_fit(ax, z, xcol, ycol, panel, ylabel):
        z = z.dropna(subset=[xcol, ycol])
        for e in ['3', '6', '12']:
            q = z[z[('ext' if 'ext' in z else 'extension')] == e]
            ax.scatter(q[xcol], q[ycol], s=100, color=ext_col[e], edgecolor='black', label=f'{e} nt', zorder=3)
        f = Fz[Fz.panel == panel].iloc[0]
        xx = np.linspace(z[xcol].min(), z[xcol].max(), 50)
        ax.plot(xx, f.intercept + f.slope * xx, 'k--', lw=1.3)
        ax.axhline(0, color='0.6', lw=1); ax.axvline(0, color='0.6', lw=1)
        ax.text(0.03, 0.96, f'r = {f.pearson_r:.2f} (p = {f.pearson_p:.1e}), n = {f.n}\n'
                            f'at no brightness change: {f.intercept:+.1f} ± {f.intercept_se:.1f}', transform=ax.transAxes,
                va='top', fontsize=13)
        ax.set_xlabel('Change in free-antibody brightness, extension - N\n(median B525-H, VSSC-1-H < 50)')
        ax.set_ylabel(ylabel); ax.legend(fontsize=11, frameon=False, loc='lower right')

    scatter_fit(axes[0, 0], O, 'change_free_antibody_median_vs_N', 'delta_vs_N', 'A: original Delta % conjugated',
                'Δ % conjugated (original, extension - N)')
    axes[0, 0].set_title('A  Original % conjugated tracks antibody brightness', fontsize=14)
    scatter_fit(axes[0, 1], O, 'change_free_antibody_median_vs_N', 'delta_gated_only', 'B: original Delta, gated part only',
                'Δ % bound, gated part only (extension - N)')
    axes[0, 1].set_title('B  Gated part only', fontsize=14)
    scatter_fit(axes[1, 1], X, 'change_free_antibody_median_vs_N', 'calibrated_delta_vs_N', 'E: calibrated Delta % conjugated',
                'Δ % conjugated (calibrated threshold, extension - N)')
    axes[1, 1].set_title('E  Calibrated threshold vs antibody brightness', fontsize=14)
    ax = axes[0, 2]
    order = ['No antibody', 'N', '3', '6', '12']
    sns.barplot(data=C, x='extension', y='calibrated_pct_conjugated', hue='coating', order=order,
                hue_order=[COAT_LBL[c] for c in COATINGS], palette={COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS},
                errorbar='se', capsize=0.05, edgecolor='black', ax=ax)
    sns.stripplot(data=C, x='extension', y='calibrated_pct_conjugated', hue='coating', order=order,
                  hue_order=[COAT_LBL[c] for c in COATINGS], dodge=True, color='black', size=5, legend=False, ax=ax)
    ax.set_xlabel('Sample (N = no biotin)'); ax.set_ylabel('% of gated origami above own free-antibody median')
    ax.set_title('C  Calibrated % conjugated', fontsize=14); ax.legend(fontsize=11, frameon=False, title='')
    ax = axes[1, 0]
    sns.barplot(data=X, x='extension', y='calibrated_delta_vs_N', hue='coating', order=['3', '6', '12'],
                hue_order=[COAT_LBL[c] for c in COATINGS], palette={COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS},
                errorbar='se', capsize=0.05, edgecolor='black', ax=ax)
    sns.stripplot(data=X, x='extension', y='calibrated_delta_vs_N', hue='coating', order=['3', '6', '12'],
                  hue_order=[COAT_LBL[c] for c in COATINGS], dodge=True, color='black', size=6, legend=False, ax=ax)
    ax.axhline(0, color='red', ls='--', lw=1.3)
    ax.set_xlabel('Biotin extension (nt)'); ax.set_ylabel('Δ % conjugated, calibrated (extension - N)')
    ax.set_title('D  Calibrated Δ % conjugated', fontsize=14); ax.legend(fontsize=11, frameon=False, title='')
    ax = axes[1, 2]
    th = C[C.extension.isin(EXTENSIONS)]
    sns.stripplot(data=th, x='extension', y='calibrated_threshold', hue='coating', order=EXTENSIONS,
                  hue_order=[COAT_LBL[c] for c in COATINGS], palette={COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS},
                  dodge=True, size=9, ax=ax)
    ax.set_xlabel('Sample'); ax.set_ylabel('Calibrated threshold = free-antibody median B525-H')
    ax.text(0.03, 0.96, f'range {th.calibrated_threshold.min():.0f}-{th.calibrated_threshold.max():.0f} a.u.\n'
                        f'original threshold (no-antibody P99): {O.thr.min():.0f}-{O.thr.max():.0f} a.u.',
            transform=ax.transAxes, va='top', fontsize=13)
    ax.set_title('F  Calibrated threshold per sample', fontsize=14); ax.legend(fontsize=11, frameon=False, title='', loc='lower right')
    fig.suptitle('Antibody session (04/18): percent conjugation vs sample-wide antibody brightness, and a brightness-calibrated '
                 'threshold (free-antibody median of the same sample)', y=1.02)
    save(fig, 'fig27_percent_conjugation_brightness', {'_calibrated': C, '_original_vs_brightness': O, '_fits': Fz})


# ---------------------------------------------------------------- fig27b
def fig27b():
    """fig27 repeated with the fig13 corrections (unbound tail, free-antibody clump correction, unmixing
    below the scatter trigger), applied at the brightness-calibrated threshold.

    Calibrated threshold (per sample) = median B525-H of that sample's own low-scatter free antibody
    (VSSC-1-H < 50, B525-H >= 250); quantify(threshold='calibrated') in quantify_bound_origami_antibody.py:
      unbound tail = buffer-subtracted fraction of gated no-antibody origami above the threshold;
      clump correction = free antibody below the scatter trigger x antibody-only ratio (gated, B525-H > threshold);
      bound origami below the trigger unmixed from events above the threshold only.
    % conjugated = (gated bound + bound below trigger) / (net gated origami + bound below trigger).
    A  % conjugated per sample            B  Delta (extension - N) by extension
    C  Delta vs change in free-antibody brightness (extension - N)
    D  Same, gated part only (unbound tail and clump corrected)
    E  Delta at no brightness change (regression intercept) after each correction step,
       fig13 threshold (no-antibody P99) vs calibrated threshold
    F  Clump correction per counted origami vs net gated origami, both thresholds
    Antibody session (04/18).
    """
    from scipy.stats import pearsonr, spearmanr
    from quantify_bound_origami_antibody import quantify
    modes = {'noab_p99': 'Figure 13 threshold (no-antibody P99)',
             'calibrated': 'Calibrated threshold (own free-antibody median)'}
    mode_col = {'noab_p99': '#7F7F7F', 'calibrated': '#6A3D9A'}
    steps = [('1 raw gated count', 'raw_gated_pct'), ('2 minus unbound tail', 'tail_corrected_pct'),
             ('3 minus clump correction (gated only)', 'pct_bound_hiSS_only'),
             ('4 plus unmixing below trigger (full)', 'pct_bound')]
    tabs = {}
    for mode, mlabel in modes.items():
        Q = quantify(verbose=False, threshold=mode)
        Q = Q[Q.qc_pass].copy()
        Q['tail_corrected_pct'] = Q.raw_gated_pct - Q.unbound_tail_pct
        Q['coating'] = Q.coating.map(COAT_LBL); Q['ext'] = Q.ext.astype(str)
        N = Q[Q.ext == 'N'].set_index(['coating', 'rep'])
        for col in ['free_ab_median_B525'] + [s for _, s in steps]:
            Q[f'delta_{col}'] = [v - N.loc[(c, r), col] if (c, r) in N.index else np.nan
                                 for v, c, r in zip(Q[col], Q.coating, Q.rep)]
        Q = Q.rename(columns={'delta_free_ab_median_B525': 'change_free_antibody_median_vs_N'})
        Q.insert(0, 'threshold_mode', mlabel)
        tabs[mode] = Q
    fits = []
    for mode, Q in tabs.items():
        X = Q[Q.ext != 'N']
        for sname, col in steps:
            z = X.dropna(subset=['change_free_antibody_median_vs_N', f'delta_{col}'])
            x, y = z.change_free_antibody_median_vs_N, z[f'delta_{col}']
            pr = pearsonr(x, y); lr = linregress(x, y)
            fits.append(dict(mode=mode, threshold_mode=modes[mode], step=sname, metric=col, n=len(z),
                             pearson_r=pr.statistic, pearson_p=pr.pvalue, spearman_rho=spearmanr(x, y).statistic,
                             slope=lr.slope, intercept=lr.intercept, intercept_se=lr.intercept_stderr,
                             mean_delta=y.mean()))
    Fz = pd.DataFrame(fits)
    for _, f in Fz.iterrows():
        stat('fig27b', f"{f['step']}: Delta % conjugated vs change in free-antibody brightness", f['threshold_mode'],
             f['pearson_r'], f['pearson_p'], f['n'],
             f"intercept (Delta at no brightness change) {f['intercept']:+.2f} +/- {f['intercept_se']:.2f}")
    C = tabs['calibrated']; X = C[C.ext != 'N']
    for e in ['3', '6', '12']:
        v = X[X.ext == e].delta_pct_bound.dropna()
        t = ttest_1samp(v, 0) if len(v) >= 2 else None
        stat('fig27b', 'calibrated + corrected Delta % conjugated vs 0 (one-sample t-test)', f'{e} nt, all coatings',
             v.mean(), t.pvalue if t else np.nan, len(v))

    ext_col = {'3': '#377EB8', '6': '#4DAF4A', '12': '#E41A1C'}
    pal = {COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}
    hue_order = [COAT_LBL[c] for c in COATINGS]
    fig, axes = plt.subplots(2, 3, figsize=(25, 14))

    def scatter_fit(ax, metric, ylabel):
        z = X.dropna(subset=['change_free_antibody_median_vs_N', f'delta_{metric}'])
        xc, yc = 'change_free_antibody_median_vs_N', f'delta_{metric}'
        for e in ['3', '6', '12']:
            q = z[z.ext == e]
            ax.scatter(q[xc], q[yc], s=100, color=ext_col[e], edgecolor='black', label=f'{e} nt', zorder=3)
        f = Fz[(Fz['mode'] == 'calibrated') & (Fz.metric == metric)].iloc[0]
        xx = np.linspace(z[xc].min(), z[xc].max(), 50)
        ax.plot(xx, f['intercept'] + f['slope'] * xx, 'k--', lw=1.3)
        ax.axhline(0, color='0.6', lw=1); ax.axvline(0, color='0.6', lw=1)
        ax.text(0.03, 0.96, f"r = {f['pearson_r']:.2f} (p = {f['pearson_p']:.1e}), n = {f['n']}\n"
                            f"at no brightness change: {f['intercept']:+.1f} ± {f['intercept_se']:.1f}",
                transform=ax.transAxes, va='top', fontsize=13)
        ax.set_xlabel('Change in free-antibody brightness, extension - N\n(median B525-H, VSSC-1-H < 50)')
        ax.set_ylabel(ylabel); ax.legend(fontsize=11, frameon=False, loc='lower right')

    ax = axes[0, 0]
    order = ['N', '3', '6', '12']
    sns.barplot(data=C, x='ext', y='pct_bound', hue='coating', order=order, hue_order=hue_order, palette=pal,
                errorbar='se', capsize=0.05, edgecolor='black', ax=ax)
    sns.stripplot(data=C, x='ext', y='pct_bound', hue='coating', order=order, hue_order=hue_order,
                  dodge=True, color='black', size=5, legend=False, ax=ax)
    ax.set_xlabel('Sample (N = no biotin)'); ax.set_ylabel('% conjugated (calibrated threshold, corrected)')
    ax.set_title('A  % conjugated, calibrated threshold + corrections', fontsize=14)
    ax.legend(fontsize=11, frameon=False, title='')
    ax = axes[0, 1]
    sns.barplot(data=X, x='ext', y='delta_pct_bound', hue='coating', order=['3', '6', '12'], hue_order=hue_order,
                palette=pal, errorbar='se', capsize=0.05, edgecolor='black', ax=ax)
    sns.stripplot(data=X, x='ext', y='delta_pct_bound', hue='coating', order=['3', '6', '12'], hue_order=hue_order,
                  dodge=True, color='black', size=6, legend=False, ax=ax)
    ax.axhline(0, color='red', ls='--', lw=1.3)
    ax.set_xlabel('Biotin extension (nt)'); ax.set_ylabel('Δ % conjugated (extension - N)')
    ax.set_title('B  Δ % conjugated, calibrated + corrected', fontsize=14)
    ax.legend(fontsize=11, frameon=False, title='')
    scatter_fit(axes[0, 2], 'pct_bound', 'Δ % conjugated (calibrated + corrected, extension - N)')
    axes[0, 2].set_title('C  Δ vs antibody brightness (full: gated + below trigger)', fontsize=14)
    scatter_fit(axes[1, 0], 'pct_bound_hiSS_only', 'Δ % bound, gated part only (extension - N)')
    axes[1, 0].set_title('D  Δ vs antibody brightness (gated part only)', fontsize=14)
    ax = axes[1, 1]
    w = 0.38
    for j, mode in enumerate(modes):
        f = Fz[Fz['mode'] == mode].reset_index(drop=True)
        xs = np.arange(len(steps)) + (j - 0.5) * w
        ax.bar(xs, f['intercept'], w, yerr=f['intercept_se'], capsize=4, color=mode_col[mode], edgecolor='black',
               label=modes[mode])
        for xi, (_, row) in zip(xs, f.iterrows()):
            top = row['intercept'] + row['intercept_se'] if row['intercept'] >= 0 else row['intercept'] - row['intercept_se']
            ax.text(xi, top + (0.3 if row['intercept'] >= 0 else -0.3), f"r={row['pearson_r']:.2f}", ha='center',
                    va='bottom' if row['intercept'] >= 0 else 'top', fontsize=10)
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels(['raw gated\ncount', '- unbound\ntail', '- clump\ncorrection', '+ unmixing\nbelow trigger'])
    ax.set_ylabel('Δ % conjugated at no brightness change\n(regression intercept ± SE)')
    ax.set_title('E  Brightness-independent Δ after each step (r = with brightness)', fontsize=14)
    ax.legend(fontsize=11, frameon=False, loc='upper left')
    ax = axes[1, 2]
    for mode, Q in tabs.items():
        ax.scatter(Q.n_origami_hiSS, Q.clump_pct, s=70, color=mode_col[mode], edgecolor='black',
                   marker='o' if mode == 'noab_p99' else '^', label=modes[mode], zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('Net gated origami (buffer-subtracted count)')
    ax.set_ylabel('Clump correction (% of gated origami)')
    ax.set_title('F  Clump correction size vs origami counted', fontsize=14)
    ax.legend(fontsize=11, frameon=False, loc='upper right')
    fig.suptitle('Antibody session (04/18): Figure 27 repeated with the Figure 13 corrections (unbound tail, '
                 'free-antibody clumps, unmixing below the scatter trigger) at the brightness-calibrated threshold', y=1.02)
    save(fig, 'fig27b_percent_conjugation_brightness_corrected',
         {'_samples': pd.concat(tabs.values(), ignore_index=True), '_fits': Fz})


# ---------------------------------------------------------------- fig27c
def fig27c():
    """Conjugation threshold from N + antibody, adjusted per extension by the gated-origami P25.

    Each extension sample is paired with the N + antibody sample of the SAME coating and SAME replicate
    (e.g. 3_PLL_AB_B with N_PLL_AB_B); nothing is pooled across replicates. Antibody session (04/18);
    origami gate; buffer-subtracted percentiles of gated B525-H (session buffer gated the same way,
    scaled by acquisition time):
      threshold(N)            = P99 of N+AB (same coating, same replicate)
      shifted threshold(ext)  = threshold(N) + [P25(ext) - P25(N)]
      scaled threshold(ext)   = threshold(N) x [P25(ext) / P25(N)]
      unshifted               = threshold(N) for every extension
    % conjugated = % of net gated origami above that sample's threshold (N = 1% by definition).
    N+AB received the same antibody, so its P99 already includes free-antibody clumps and non-specific
    signal. Brightness change = change in median B525-H of low-scatter free antibody (VSSC-1-H < 50,
    B525-H >= 250), extension - N, same replicate.
    A  P25 ratio ext / N per replicate   B  thresholds   C  % conjugated (scaled)
    D  Delta (extension - N) for the three thresholds   E  scaled Delta vs brightness   F  shifted Delta vs brightness
    """
    from scipy.stats import pearsonr
    s4 = SESSIONS['Antibody session (04/18)']
    dfs = session_dfs(s4, exts=tuple(EXTENSIONS))
    gates = compute_monomer_gates(dfs, ctl=s4['noprobe'])
    buf = dfs['Buffer']; tb = acq_time(buf)
    rows = []
    for c in COATINGS:
        bb = apply_gate(buf, c, 'A', gates)[CH_S].values
        for r in REPLICATES:
            dN = dfs.get(('N', c, s4['probe'], r))
            if dN is None:
                continue
            gN, tN = apply_gate(dN, c, r, gates)[CH_S].values, acq_time(dN)
            thrN, p25N = bs_percentile(gN, tN, bb, tb, 99), bs_percentile(gN, tN, bb, tb, 25)
            for e in EXTENSIONS:
                d = dfs.get((e, c, s4['probe'], r))
                if d is None:
                    continue
                g, t = apply_gate(d, c, r, gates)[CH_S].values, acq_time(d)
                p25 = bs_percentile(g, t, bb, tb, 25)
                thr_sh, thr_sc = thrN + (p25 - p25N), thrN * p25 / p25N
                net, a_sh = bs_counts(g, t, bb, tb, thr_sh)
                _, a_sc = bs_counts(g, t, bb, tb, thr_sc)
                _, a_0 = bs_counts(g, t, bb, tb, thrN)
                lo = d[(d[CH_SS] < 50) & (d[CH_S] >= 250)][CH_S]
                pc = (lambda a: a / net * 100 if net > 0 else np.nan)
                rows.append(dict(coating=COAT_LBL[c], replicate=r, extension=e,
                                 sample=f"{e}_{c}_{s4['probe']}_{r}", paired_N_sample=f"N_{c}_{s4['probe']}_{r}",
                                 net_gated_origami=net, P25_B525=p25, P25_N=p25N,
                                 P25_shift_vs_N=p25 - p25N, P25_ratio_vs_N=p25 / p25N, threshold_N_P99=thrN,
                                 threshold_shifted=thr_sh, threshold_scaled=thr_sc,
                                 pct_conjugated_unshifted=pc(a_0), pct_conjugated_shifted=pc(a_sh),
                                 pct_conjugated_scaled=pc(a_sc), free_antibody_median_B525=float(lo.median())))
    Call = pd.DataFrame(rows)
    Call['qc_pass'] = Call.net_gated_origami > 0          # no net origami above buffer -> excluded
    C = Call[Call.qc_pass].copy()
    base = C[C.extension == 'N'].set_index(['coating', 'replicate'])
    methods = {'unshifted': 'Unshifted (N+AB P99)', 'shifted': 'Shifted: + [P25(ext) - P25(N)]',
               'scaled': 'Scaled: x P25(ext) / P25(N)'}
    for m in methods:
        C[f'delta_vs_N_{m}'] = [v - base.loc[k, f'pct_conjugated_{m}']
                                for v, k in zip(C[f'pct_conjugated_{m}'], zip(C.coating, C.replicate))]
    C['change_free_antibody_median_vs_N'] = [v - base.loc[k, 'free_antibody_median_B525']
                                             for v, k in zip(C.free_antibody_median_B525, zip(C.coating, C.replicate))]
    X = C[C.extension != 'N']
    fits, summ = [], []
    for m, mlabel in methods.items():
        x, y = X.change_free_antibody_median_vs_N, X[f'delta_vs_N_{m}']
        pr = pearsonr(x, y); lr = linregress(x, y)
        fits.append(dict(method=m, threshold=mlabel, n=len(X), pearson_r=pr.statistic, pearson_p=pr.pvalue,
                         slope=lr.slope, intercept=lr.intercept, intercept_se=lr.intercept_stderr))
        stat('fig27c', f'Delta % conjugated vs change in free-antibody brightness, {mlabel}', 'antibody session (04/18)',
             pr.statistic, pr.pvalue, len(X), f'intercept {lr.intercept:+.2f} +/- {lr.intercept_stderr:.2f}')
        for e in ['3', '6', '12']:
            v = X[X.extension == e][f'delta_vs_N_{m}']
            t = ttest_1samp(v, 0)
            summ.append(dict(extension=e, method=m, threshold=mlabel, mean_delta=v.mean(), sem=v.sem(), n=len(v),
                             p_vs_0=t.pvalue, mean_P25_ratio=X[X.extension == e].P25_ratio_vs_N.mean(),
                             mean_P25_shift=X[X.extension == e].P25_shift_vs_N.mean()))
            stat('fig27c', f'Delta % conjugated vs 0 (one-sample t-test), {mlabel}', f'{e} nt, all coatings',
                 v.mean(), t.pvalue, len(v))
    Fz, Sm = pd.DataFrame(fits), pd.DataFrame(summ)

    pal = {COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}
    hue_order = [COAT_LBL[c] for c in COATINGS]
    ext_col = {'3': '#377EB8', '6': '#4DAF4A', '12': '#E41A1C'}
    mcol = {methods['unshifted']: '#BDBDBD', methods['shifted']: '#9E9AC8', methods['scaled']: '#54278F'}
    fig, axes = plt.subplots(2, 3, figsize=(27, 15))
    # A: P25 ratio per replicate, each extension paired with its own replicate's N
    ax = axes[0, 0]
    xpos = {'3': 0, '6': 1, '12': 2}
    off = {('Uncoated', 'A'): -0.3, ('Uncoated', 'B'): -0.24, ('Uncoated', 'C'): -0.18,
           ('PLL-PEG', 'A'): -0.06, ('PLL-PEG', 'B'): 0.0, ('PLL-PEG', 'C'): 0.06,
           ('Silica', 'A'): 0.18, ('Silica', 'B'): 0.24, ('Silica', 'C'): 0.3}
    for _, q in X.iterrows():
        xx = xpos[q.extension] + off[(q.coating, q.replicate)]
        ax.scatter(xx, q.P25_ratio_vs_N, s=90, color=pal[q.coating], edgecolor='black', zorder=3)
        ax.annotate(q.replicate, (xx, q.P25_ratio_vs_N), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(1, color='0.5', ls='--', lw=1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['3', '6', '12'])
    ax.set_xlabel('Biotin extension (nt); letter = replicate')
    ax.set_ylabel('P25(extension) / P25(N + AB), same coating & replicate')
    ax.set_title('A  P25 ratio, each replicate vs its own N', fontsize=14)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=pal[h], markeredgecolor='black', markersize=10, label=h)
                       for h in hue_order], fontsize=11, frameon=False, loc='upper right')
    # B: thresholds
    ax = axes[0, 1]
    T = C.melt(id_vars=['coating', 'replicate', 'extension'], value_vars=['threshold_shifted', 'threshold_scaled'],
               var_name='method', value_name='threshold')
    T['method'] = T.method.map({'threshold_shifted': methods['shifted'], 'threshold_scaled': methods['scaled']})
    T.loc[T.extension == 'N', 'method'] = methods['unshifted']
    T = T.drop_duplicates(['coating', 'replicate', 'extension', 'method'])
    sns.stripplot(data=T, x='extension', y='threshold', hue='method', order=EXTENSIONS, palette=mcol,
                  hue_order=list(mcol), dodge=True, size=9, edgecolor='black', linewidth=0.8, ax=ax)
    ax.set_xlabel('Sample (N = no biotin + antibody)'); ax.set_ylabel('Conjugation threshold, B525-H')
    ax.set_title('B  Thresholds (N = P99 of N + AB, per replicate)', fontsize=14)
    ax.legend(fontsize=11, frameon=False, title='', loc='upper left')
    # C: % conjugated, scaled
    ax = axes[0, 2]
    sns.barplot(data=C, x='extension', y='pct_conjugated_scaled', hue='coating', order=EXTENSIONS, hue_order=hue_order,
                palette=pal, errorbar='se', capsize=0.05, edgecolor='black', ax=ax)
    sns.stripplot(data=C, x='extension', y='pct_conjugated_scaled', hue='coating', order=EXTENSIONS, hue_order=hue_order,
                  dodge=True, color='black', size=5, legend=False, ax=ax)
    ax.set_xlabel('Sample (N = no biotin + antibody)'); ax.set_ylabel('% of gated origami above scaled threshold')
    ax.set_title('C  % conjugated, scaled threshold (N = 1% by definition)', fontsize=14)
    ax.legend(fontsize=11, frameon=False, title='')
    # D: Delta, three methods
    ax = axes[1, 0]
    L_ = X.melt(id_vars=['coating', 'replicate', 'extension'], value_vars=[f'delta_vs_N_{m}' for m in methods],
                var_name='method', value_name='delta')
    L_['method'] = L_.method.str.replace('delta_vs_N_', '').map(methods)
    sns.barplot(data=L_, x='extension', y='delta', hue='method', order=['3', '6', '12'], palette=mcol,
                hue_order=list(mcol), errorbar='se', capsize=0.05, edgecolor='black', ax=ax)
    sns.stripplot(data=L_, x='extension', y='delta', hue='method', order=['3', '6', '12'], hue_order=list(mcol),
                  dodge=True, color='black', size=5, legend=False, ax=ax)
    ax.axhline(0, color='red', ls='--', lw=1.3)
    for i, e in enumerate(['3', '6', '12']):
        s = Sm[(Sm.extension == e) & (Sm['method'] == 'scaled')].iloc[0]
        ax.text(i, ax.get_ylim()[1], f"scaled {s['mean_delta']:+.2f} ± {s['sem']:.2f}\np = {s['p_vs_0']:.2f}",
                ha='center', va='top', fontsize=11)
    ax.set_xlabel('Biotin extension (nt)'); ax.set_ylabel('Δ % conjugated (extension - N, same replicate)')
    ax.set_title('D  Δ % conjugated by threshold', fontsize=14)
    ax.legend(fontsize=11, frameon=False, title='', loc='lower left')

    def scatter_fit(ax, m, title):
        f = Fz[Fz['method'] == m].iloc[0]
        for e in ['3', '6', '12']:
            q = X[X.extension == e]
            ax.scatter(q.change_free_antibody_median_vs_N, q[f'delta_vs_N_{m}'], s=100, color=ext_col[e],
                       edgecolor='black', label=f'{e} nt', zorder=3)
        xx = np.linspace(X.change_free_antibody_median_vs_N.min(), X.change_free_antibody_median_vs_N.max(), 50)
        ax.plot(xx, f['intercept'] + f['slope'] * xx, 'k--', lw=1.3)
        ax.axhline(0, color='0.6', lw=1); ax.axvline(0, color='0.6', lw=1)
        ax.text(0.03, 0.96, f"r = {f['pearson_r']:.2f} (p = {f['pearson_p']:.1e}), n = {int(f['n'])}\n"
                            f"at no brightness change: {f['intercept']:+.2f} ± {f['intercept_se']:.2f}",
                transform=ax.transAxes, va='top', fontsize=12, zorder=5,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
        ax.set_xlabel('Change in free-antibody brightness, extension - N\n(median B525-H, VSSC-1-H < 50)')
        ax.set_ylabel('Δ % conjugated (extension - N)')
        ax.set_title(title, fontsize=14); ax.legend(fontsize=11, frameon=False, loc='lower right')

    scatter_fit(axes[1, 1], 'scaled', 'E  Scaled threshold: Δ vs antibody brightness')
    scatter_fit(axes[1, 2], 'shifted', 'F  Shifted threshold: Δ vs antibody brightness')
    fig.suptitle('Antibody session (04/18): threshold = P99 of no-biotin + antibody (per replicate), adjusted for each '
                 'extension by its own replicate\'s gated-origami P25 (shifted: + difference; scaled: x ratio)', y=1.02)
    save(fig, 'fig27c_threshold_N_antibody_P25_shift',
         {'_samples': Call.merge(C, how='left'), '_by_extension': Sm, '_fits': Fz})


# ---------------------------------------------------------------- fig27d
def fig27d():
    """Mixture model for antibody-bound origami with a brightness gain (antibody session 04/18).

    Each extension sample is modelled from the N + antibody sample of the SAME coating and replicate:
      extension = (1 - f) x [N+AB, stretched by gain s] + f x [N+AB + one antibody, stretched by s]
    f = excess fraction of origami carrying one more antibody than in N (may be negative).
    Single-antibody brightness A: lognormal; free antibody is only recorded above the B525 trigger (250), so A is
    fitted as a truncated lognormal to the low-scatter free-antibody events (VSSC-1-H < 50, B525-H >= 250):
    sigma from all N+AB samples pooled, median per replicate set from its own N+AB sample.
    Gain s (main variant) is estimated from origami-free data only: the shift of the same sample's free-antibody
    distribution relative to its N+AB (truncated lognormal, same sigma). f is then fitted to the gated-origami
    B525-H histogram (Poisson likelihood, the session buffer gated the same way and scaled by acquisition time
    added as a known background; 70 log bins 10-6000 + under/overflow).
    Sensitivity variants: A median x 0.6 and x 1.5; s and f both free (from origami alone); s shared jointly
    between origami and free antibody.
    A  example fit (sample with the largest f)   B  gain from free antibody vs origami stretch
    C  why s must be constrained: f with s free vs brightness change   D  f (s from free antibody) by extension
    E  f (s from free antibody) vs brightness change   F  mean f per extension, all variants
    """
    from scipy.optimize import minimize, minimize_scalar
    from scipy.signal import fftconvolve
    from scipy.stats import norm, pearsonr
    W, XMAX, T, LOW_SS = 2.0, 6000.0, 250, 50
    EDG = np.arange(0, XMAX + W, W); NB = len(EDG) - 1
    ABIN = np.r_[0, np.logspace(1, np.log10(XMAX), 70)]; AIDX = np.searchsorted(EDG, ABIN)

    def fine_hist(v):
        return np.histogram(np.clip(v, 0, None), np.r_[EDG, np.inf])[0].astype(float)

    def to_analysis(p):
        cs = np.r_[0, np.cumsum(p[:-1])]
        return np.r_[np.diff(cs[AIDX]), p[:-1][AIDX[-1]:].sum() + p[-1]]

    def stretch(p, s):
        F = np.r_[0, np.cumsum(p[:-1])]
        G = np.interp(EDG / s, EDG, F, right=F[-1])
        return np.r_[np.diff(G), 1 - G[-1]]

    def lognorm_pmf(med, sg):
        cdf = norm.cdf((np.log(np.maximum(EDG, 1e-9)) - np.log(med)) / sg); cdf[0] = 0
        return np.r_[np.diff(cdf), 1 - cdf[-1]]

    def conv(p, a):
        q = np.clip(fftconvolve(p[:-1], a[:-1])[:NB], 0, None)
        return np.r_[q, 1 - q.sum()]

    def fa_nll(mu, v, sg):
        return -(norm.logpdf(np.log(v), mu, sg) - norm.logsf((np.log(T) - mu) / sg)).sum()

    def fit_mu(v, sg):
        return minimize_scalar(lambda m: fa_nll(m, v, sg), bounds=(np.log(20), np.log(400)), method='bounded').x

    s4 = SESSIONS['Antibody session (04/18)']
    dfs = session_dfs(s4, exts=tuple(EXTENSIONS))
    gates = compute_monomer_gates(dfs, ctl=s4['noprobe'])
    buf = dfs['Buffer']; tb = acq_time(buf)
    free = lambda d: d[(d[CH_SS] < LOW_SS) & (d[CH_S] >= T)][CH_S].values
    pooled = np.concatenate([free(dfs[('N', c, s4['probe'], r)]) for c in COATINGS for r in REPLICATES
                             if ('N', c, s4['probe'], r) in dfs])
    o = min([minimize(lambda th: fa_nll(th[0], pooled, np.exp(th[1])), [np.log(m0), np.log(s0)], method='Nelder-Mead',
                      options={'xatol': 1e-7, 'fatol': 1e-5, 'maxiter': 4000})
             for m0 in (50, 150, 300) for s0 in (0.2, 0.5)], key=lambda z: z.fun)
    SG = float(np.exp(o.x[1])); A_POOLED = float(np.exp(o.x[0]))
    stat('fig27d', 'single-antibody B525-H, truncated lognormal fit to free antibody (N+AB pooled)', 'antibody session (04/18)',
         A_POOLED, np.nan, len(pooled), f'median {A_POOLED:.1f}, sigma {SG:.3f}; '
         f'fraction above the 250 trigger {norm.sf((np.log(T) - o.x[0]) / SG):.3f}')
    SCALES = (0.6, 1.0, 1.5)
    rows, comps = [], {}
    for c in COATINGS:
        bfine = fine_hist(apply_gate(buf, c, 'A', gates)[CH_S].values); ba = to_analysis(bfine)
        for r in REPLICATES:
            dN = dfs.get(('N', c, s4['probe'], r))
            if dN is None:
                continue
            vN = free(dN); muN = fit_mu(vN, SG)
            pN = np.clip(fine_hist(apply_gate(dN, c, r, gates)[CH_S].values) - acq_time(dN) / tb * bfine, 0, None)
            pN /= pN.sum()
            pNA = {sc: conv(pN, lognorm_pmf(np.exp(muN) * sc, SG)) for sc in SCALES}
            for e in ['3', '6', '12']:
                d = dfs.get((e, c, s4['probe'], r))
                if d is None:
                    continue
                k = acq_time(d) / tb
                h = to_analysis(fine_hist(apply_gate(d, c, r, gates)[CH_S].values)); n = h.sum() - k * ba.sum()
                if n <= 0:
                    continue
                v = free(d); s_fa = float(np.exp(fit_mu(v, SG) - muN))

                def model(s, f, sc):
                    qu = to_analysis(np.clip(stretch(pN, s), 0, None)); qb = to_analysis(np.clip(stretch(pNA[sc], s), 0, None))
                    return qu / qu.sum(), qb / qb.sum()

                def nll(s, f, sc=1.0):
                    qu, qb = model(s, f, sc)
                    qa = np.clip((1 - f) * qu + f * qb, 1e-12, None); qa /= qa.sum()
                    lam = np.clip(k * ba + n * qa, 1e-9, None)
                    return float((lam - h * np.log(lam)).sum())

                row = dict(coating=COAT_LBL[c], replicate=r, extension=e, sample=f"{e}_{c}_{s4['probe']}_{r}",
                           paired_N_sample=f"N_{c}_{s4['probe']}_{r}", net_gated_origami=n, n_free_antibody=len(v),
                           single_antibody_median_N=float(np.exp(muN)), single_antibody_sigma=SG,
                           change_free_antibody_median_vs_N=float(np.median(v) - np.median(vN)), s_free_antibody=s_fa)
                row['s_origami_stretch_only'] = minimize_scalar(lambda s: nll(s, 0.0), bounds=(0.5, 2), method='bounded').x
                for sc in SCALES:
                    fg = np.linspace(-0.3, 0.6, 181)               # grid first, then refine (avoid local minima)
                    f0 = fg[int(np.argmin([nll(s_fa, f, sc) for f in fg]))]
                    of = minimize_scalar(lambda f: nll(s_fa, f, sc), bounds=(f0 - 0.005, f0 + 0.005), method='bounded')
                    row[f'f_s_from_free_antibody_A_x{sc}'] = of.x
                    if sc == 1.0:
                        row['nll_s_fixed_mixture'] = of.fun; row['nll_s_fixed_f0'] = nll(s_fa, 0.0)
                        comps[(c, r, e)] = (h, k * ba, n, s_fa, of.x, *model(s_fa, of.x, 1.0))
                row['nll_stretch_only_s_free'] = nll(row['s_origami_stretch_only'], 0.0)
                om = min([minimize(lambda th: nll(th[0], th[1]), x0, method='Nelder-Mead',
                                   options={'xatol': 1e-6, 'fatol': 1e-4, 'maxiter': 3000})
                          for x0 in ([row['s_origami_stretch_only'], 0.0], [s_fa, 0.0], [1.0, 0.05], [1.0, 0.2])],
                         key=lambda z: z.fun)
                row['s_free_mixture'], row['f_s_free'] = om.x
                oj = min([minimize(lambda th: nll(th[0], th[1]) + fa_nll(muN + np.log(th[0]), v, SG), x0,
                                   method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-4, 'maxiter': 3000})
                          for x0 in ([s_fa, 0.0], [row['s_origami_stretch_only'], 0.0], [1.0, 0.05])], key=lambda z: z.fun)
                row['s_joint'], row['f_joint'] = oj.x
                rows.append(row)
    D = pd.DataFrame(rows)
    D['f_main_pct'] = D['f_s_from_free_antibody_A_x1.0'] * 100
    variants = {'f_s_from_free_antibody_A_x1.0': 's from free antibody (main)',
                'f_s_from_free_antibody_A_x0.6': 's from free antibody, A x 0.6',
                'f_s_from_free_antibody_A_x1.5': 's from free antibody, A x 1.5',
                'f_s_free': 's free (origami only)', 'f_joint': 's joint (origami + free antibody)'}
    summ, fits = [], []
    for col, vl in variants.items():
        x, y = D.change_free_antibody_median_vs_N, D[col] * 100
        pr = pearsonr(x, y); lr = linregress(x, y)
        fits.append(dict(variant=vl, column=col, n=len(D), pearson_r=pr.statistic, pearson_p=pr.pvalue, slope=lr.slope,
                         intercept=lr.intercept, intercept_se=lr.intercept_stderr))
        stat('fig27d', f'excess bound fraction (%) vs change in free-antibody brightness, {vl}', 'antibody session (04/18)',
             pr.statistic, pr.pvalue, len(D), f'intercept {lr.intercept:+.2f} +/- {lr.intercept_stderr:.2f}')
        for e in ['3', '6', '12']:
            vv = D[D.extension == e][col] * 100; t = ttest_1samp(vv, 0)
            summ.append(dict(variant=vl, extension=e, mean_excess_bound_pct=vv.mean(), sem=vv.sem(), n=len(vv), p_vs_0=t.pvalue))
            stat('fig27d', f'excess bound fraction (%) vs 0 (one-sample t-test), {vl}', f'{e} nt, all coatings',
                 vv.mean(), t.pvalue, len(vv))
    pr = pearsonr(np.log(D.s_free_antibody), np.log(D.s_origami_stretch_only))
    stat('fig27d', 'gain from free antibody vs gain from origami stretch (log)', 'antibody session (04/18)', pr.statistic,
         pr.pvalue, len(D))
    Sm, Fz = pd.DataFrame(summ), pd.DataFrame(fits)

    pal = {COAT_LBL[c]: COAT_COLORS[c] for c in COATINGS}
    hue_order = [COAT_LBL[c] for c in COATINGS]
    ext_col = {'3': '#377EB8', '6': '#4DAF4A', '12': '#E41A1C'}
    fig, axes = plt.subplots(2, 3, figsize=(27, 15))
    # A: example fit
    ax = axes[0, 0]
    ex = D.loc[D['f_main_pct'].idxmax()]
    h, bk, n, s_fa, f, qu, qb = comps[(next(cc for cc in COATINGS if COAT_LBL[cc] == ex.coating), ex.replicate, ex.extension)]
    ctr = np.sqrt(ABIN[1:-1] * ABIN[2:]); dl = np.diff(np.log10(ABIN[1:]))
    sl = slice(1, len(ABIN) - 1)
    ax.step(ctr, (h - bk)[sl] / n / dl, where='mid', color='black', lw=1.5, label='observed (buffer-subtracted)')
    ax.plot(ctr, (1 - f) * qu[sl] / dl, color='#7F7F7F', lw=2, label=f'unbound: N+AB stretched x{s_fa:.2f}')
    ax.plot(ctr, f * qb[sl] / dl, color='#6A3D9A', lw=2, label=f'+1 antibody: f = {f * 100:.1f}%')
    ax.plot(ctr, ((1 - f) * qu + f * qb)[sl] / dl, color='#E6862E', lw=2, ls='--', label='model total')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-5, None)
    ax.set_xlabel('B525-H (gated origami)'); ax.set_ylabel('Fraction per log10 unit')
    ax.set_title(f'A  Example: {ex["sample"]} (largest f)', fontsize=14); ax.legend(fontsize=10, frameon=False, loc='upper right')
    # B: gain agreement
    ax = axes[0, 1]
    for e in ['3', '6', '12']:
        q = D[D.extension == e]
        ax.scatter(q.s_free_antibody, q.s_origami_stretch_only, s=100, color=ext_col[e], edgecolor='black', label=f'{e} nt', zorder=3)
    lim = [min(D.s_free_antibody.min(), D.s_origami_stretch_only.min()) - 0.03,
           max(D.s_free_antibody.max(), D.s_origami_stretch_only.max()) + 0.03]
    ax.plot(lim, lim, color='0.5', ls='--', lw=1)
    ax.text(0.03, 0.96, f'r = {pr.statistic:.2f} (log), n = {len(D)}', transform=ax.transAxes, va='top', fontsize=12)
    ax.set_xlabel('Gain from free antibody (vs own N+AB)'); ax.set_ylabel('Gain from origami (stretch only)')
    ax.set_title('B  One gain explains free antibody and origami', fontsize=14); ax.legend(fontsize=11, frameon=False, loc='lower right')

    def scatter_fit(ax, col, title, ylabel):
        f_ = Fz[Fz.column == col].iloc[0]
        for e in ['3', '6', '12']:
            q = D[D.extension == e]
            ax.scatter(q.change_free_antibody_median_vs_N, q[col] * 100, s=100, color=ext_col[e], edgecolor='black',
                       label=f'{e} nt', zorder=3)
        xx = np.linspace(D.change_free_antibody_median_vs_N.min(), D.change_free_antibody_median_vs_N.max(), 50)
        ax.plot(xx, f_['intercept'] + f_['slope'] * xx, 'k--', lw=1.3)
        ax.axhline(0, color='0.6', lw=1); ax.axvline(0, color='0.6', lw=1)
        ax.text(0.03, 0.96, f"r = {f_['pearson_r']:.2f} (p = {f_['pearson_p']:.1e}), n = {int(f_['n'])}\n"
                            f"at no brightness change: {f_['intercept']:+.2f} ± {f_['intercept_se']:.2f}",
                transform=ax.transAxes, va='top', fontsize=12, zorder=5, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
        ax.set_xlabel('Change in free-antibody brightness, extension - N\n(median B525-H, VSSC-1-H < 50)')
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=14); ax.legend(fontsize=11, frameon=False, loc='lower right')

    scatter_fit(axes[0, 2], 'f_s_free', 'C  s free: model trades gain for binding', 'Excess bound fraction, s free (%)')
    ax = axes[1, 0]
    sns.barplot(data=D, x='extension', y='f_main_pct', hue='coating', order=['3', '6', '12'], hue_order=hue_order,
                palette=pal, errorbar='se', capsize=0.05, edgecolor='black', ax=ax)
    sns.stripplot(data=D, x='extension', y='f_main_pct', hue='coating', order=['3', '6', '12'], hue_order=hue_order,
                  dodge=True, color='black', size=6, legend=False, ax=ax)
    ax.axhline(0, color='red', ls='--', lw=1.3)
    for i, e in enumerate(['3', '6', '12']):
        s_ = Sm[(Sm.variant == variants['f_s_from_free_antibody_A_x1.0']) & (Sm.extension == e)].iloc[0]
        ax.text(i, ax.get_ylim()[1], f"{s_['mean_excess_bound_pct']:+.2f} ± {s_['sem']:.2f}\np = {s_['p_vs_0']:.2f}",
                ha='center', va='top', fontsize=11)
    ax.set_xlabel('Biotin extension (nt)'); ax.set_ylabel('Excess bound fraction vs own N+AB (%)')
    ax.set_title('D  Excess bound fraction, s from free antibody', fontsize=14); ax.legend(fontsize=11, frameon=False, title='', loc='lower left')
    scatter_fit(axes[1, 1], 'f_s_from_free_antibody_A_x1.0', 'E  s from free antibody: f vs brightness',
                'Excess bound fraction, s from free antibody (%)')
    ax = axes[1, 2]
    vpal = dict(zip(variants.values(), ['#54278F', '#9E9AC8', '#3F007D', '#BDBDBD', '#E6862E']))
    sns.barplot(data=Sm, x='extension', y='mean_excess_bound_pct', hue='variant', order=['3', '6', '12'],
                hue_order=list(variants.values()), palette=vpal, edgecolor='black', ax=ax)
    for bar_group, vl in zip(ax.containers, variants.values()):
        sub = Sm[Sm.variant == vl].set_index('extension').reindex(['3', '6', '12'])
        ax.errorbar([b.get_x() + b.get_width() / 2 for b in bar_group], sub.mean_excess_bound_pct, yerr=sub['sem'],
                    fmt='none', ecolor='black', capsize=3, lw=1.2)
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('Biotin extension (nt)'); ax.set_ylabel('Mean excess bound fraction (%) ± SEM')
    ax.set_title('F  Sensitivity: antibody brightness and gain constraint', fontsize=14)
    ax.legend(fontsize=10, frameon=False, title='', loc='upper right')
    fig.suptitle('Antibody session (04/18): mixture model, extension = (1 - f) x stretched N+AB + f x (N+AB + one antibody), '
                 'each replicate vs its own N+AB; gain from the sample\'s free antibody', y=1.02)
    save(fig, 'fig27d_mixture_model', {'_samples': D, '_by_extension': Sm, '_fits': Fz})


FIGS = {f.__name__: f for f in [fig00, fig01, fig02, fig03, fig06, fig07, fig08, fig09,
                                 fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig16b, fig17, fig18, fig19, fig20, fig21, fig22, fig23, fig24, fig25, fig26, fig27, fig27b, fig27c, fig27d]}

if __name__ == '__main__':
    todo = sys.argv[1:] or list(FIGS)
    for name in todo:
        print(name)
        FIGS[name]()
    pd.DataFrame(STATS).to_csv(os.path.join(OUT, 'stats_summary.csv' if len(todo) == len(FIGS)
                                            else f'stats_summary_{"_".join(todo)}.csv'), index=False)
    print(f'wrote stats to {OUT}')
