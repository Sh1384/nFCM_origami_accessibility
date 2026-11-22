"""Figure S4B: local slope of the cumulative side-scatter distribution of bare origami without probe.

N(>=s) is the number of recorded events with VSSC-1-H >= s, from the raw events of the three bare-origami tubes of each
batch, pooled. The local slope -d ln N(>=s) / d ln s is a central difference on a grid of 0.05 decade that starts at the
trigger (200), so each point spans 0.1 decade and no value below the trigger is used. The first point is at 224 (200 to
252). Minus buffer: the buffer counts of the batch, scaled by acquisition time, are subtracted; a point is left empty
where the buffer is half or more of the events above its window. Same definition as fig41_caption_stat_checks.py q4
(checks_11_08/fig41d, middle plot), shown from 200 to about 500.

Out (package folder Figure_S4/Panel_B):
  Figure_S4_B.svg               the panel
  Figure_S4_B.csv               plotted values and the counts behind them
  Figure_S4_B_tubes.csv         the same quantities for each tube (not plotted)
  Figure_S4_B_acquisitions.csv  source file, acquisition time and event counts of each acquisition
  Figure_S4_B_raw_events.csv    VSSC-1-H of every recorded event of those acquisitions, unfiltered
  README.txt                    method and suggested caption
Run from nFCM Manuscript/scripts:  python fig_s4b_local_slope.py
build_figure_package.py also writes this panel when it rebuilds the package.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import fig37_proposed_flow_panels as P          # sets the plot style
import make_manuscript_figures as M
from shared_utils import acq_time

SS, TRIG = 'VSSC-1-H', 200
STEP = 0.05                 # decade; the slope at each point spans two steps
N_POINTS = 8                # 224 to 502
MIN_EVENTS = 200            # events required above a window
MAX_BUFFER_SHARE = 0.5      # buffer share allowed above a window for the subtracted slope
TITLE = 'Local slope of the cumulative side-scatter distribution of bare origami'
CAPTION = ('(B) Local slope of the cumulative side-scatter distribution of bare origami, -d ln N(>=s) / d ln s, where N(>=s) '
           'is the number of recorded events with VSSC-1-H >= s. Events from three technical replicates were pooled in each '
           'batch. Slopes were calculated over windows of 0.1 decade from the trigger (200) upward, without (filled) and '
           'with (open) subtraction of buffer events scaled by acquisition time. Subtracted values are omitted where buffer '
           'accounts for at least half of the events above the window.')


def load():
    """Raw VSSC-1-H values of every acquisition used."""
    acq = []
    for sess in [P.B1, P.B2]:
        folder, tag = M.SESSIONS[sess]['folder'], M.SESSIONS[sess]['noprobe']
        tubes = P.files(sess, 'NoCoating'); buf = P.buffer(sess)
        if len(tubes) != 3 or buf is None:
            raise SystemExit(f'{sess}: expected three bare-origami tubes and a buffer in {folder}')
        for sample, rep, d, f in [('bare', r, d, f'N_NoCoating_{tag}_{r}.fcs') for r, d in tubes] + [('buffer', '', buf, 'Buffer.fcs')]:
            acq.append(dict(batch=P.BATCH[sess], sample=sample, replicate=rep, file=f'{folder}/{f}',
                            acq_time_s=acq_time(d), values=d[SS].values.astype(float)))
    return acq


def local_slope(v, t, bv, bt):
    """Central-difference slope of ln N(>=s) vs ln s for events v (time t), raw and minus buffer bv (time bt)."""
    edges = TRIG * 10 ** (STEP * np.arange(N_POINTS + 2))
    count = lambda x: len(x) - np.searchsorted(np.sort(x), edges, side='left')
    n, b = count(v), count(bv) * t / bt
    rows = []
    for i in range(1, N_POINTS + 1):
        lo, hi = i - 1, i + 1
        dlns, net_lo, net_hi = np.log(edges[hi] / edges[lo]), n[lo] - b[lo], n[hi] - b[hi]
        raw = np.log(n[lo] / n[hi]) / dlns if n[hi] >= MIN_EVENTS else np.nan
        net = np.log(net_lo / net_hi) / dlns if b[hi] < MAX_BUFFER_SHARE * n[hi] and net_lo > net_hi > 0 else np.nan
        rows.append(dict(s=edges[i], window_lo=edges[lo], window_hi=edges[hi], local_slope_raw=raw,
                         local_slope_minus_buffer=net, events_ge_window_lo=n[lo], events_ge_window_hi=n[hi],
                         buffer_events_ge_window_lo_scaled=b[lo], buffer_events_ge_window_hi_scaled=b[hi]))
    return rows


def tables():
    acq = load(); pooled, tubes = [], []
    for batch in dict.fromkeys(a['batch'] for a in acq):
        bare = [a for a in acq if a['batch'] == batch and a['sample'] == 'bare']
        buf = next(a for a in acq if a['batch'] == batch and a['sample'] == 'buffer')
        v, t = np.concatenate([a['values'] for a in bare]), sum(a['acq_time_s'] for a in bare)
        pooled += [dict(batch=batch, **r) for r in local_slope(v, t, buf['values'], buf['acq_time_s'])]
        for a in bare:
            tubes += [dict(batch=batch, replicate=a['replicate'], **r)
                      for r in local_slope(a['values'], a['acq_time_s'], buf['values'], buf['acq_time_s'])]
    info = [dict({k: a[k] for k in ['batch', 'sample', 'replicate', 'file', 'acq_time_s']}, recorded_events=len(a['values']),
                 events_ge_200=int((a['values'] >= TRIG).sum())) for a in acq]
    raw = pd.concat([pd.DataFrame({'batch': a['batch'], 'sample': a['sample'], 'replicate': a['replicate'],
                                   'VSSC1H': a['values']}) for a in acq], ignore_index=True)
    return {'plotted': pd.DataFrame(pooled), 'tubes': pd.DataFrame(tubes), 'acquisitions': pd.DataFrame(info), 'raw_events': raw}


def plot(d):
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    series = [('local_slope_raw', 'raw', dict(color=P.COL['NoCoating'], ls='-', ms=4.2, zorder=3)),
              ('local_slope_minus_buffer', 'minus buffer', dict(color=P.COL['Buffer'], ls='--', ms=5.6, mfc='white', zorder=2))]
    for batch, mk in [('batch 1', '^'), ('batch 2', 'o')]:          # markers as in Figure 3 of the manuscript
        z = d[d.batch == batch]
        for col, lab, kw in series:
            ax.plot(z.s, z[col], marker=mk, lw=1.1, label=f'{batch}, {lab}', **kw)
    ax.set_xscale('log'); ax.set_xlim(TRIG, 540)
    ax.set_xticks([200, 250, 300, 400, 500]); ax.set_xticklabels(['200', '250', '300', '400', '500'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0, 6.5); ax.set_yticks(range(7))
    ax.set_xlabel('VSSC-1-H, s'); ax.set_ylabel('Local log–log slope')
    ax.legend(fontsize=6.5, loc='lower left', handlelength=2.6)
    fig.tight_layout()
    return fig


def write(write_panel):
    """Write the panel folder with build_figure_package.write_panel."""
    T = tables()
    write_panel('S4', 'B', TITLE,
                'local slope of the cumulative count N(>=s) of raw events against VSSC-1-H, s, for bare origami without probe '
                '(three tubes pooled per batch), raw and minus the session buffer (scaled by acquisition time); central '
                'difference over 0.1 decade on a 0.05-decade grid from the trigger (200), points at 224 to 502. '
                '_raw_events.csv holds every recorded event of the eight acquisitions, unfiltered; _tubes.csv holds the slope '
                'of each tube (not plotted)', 'fig_s4b_local_slope.py', plot(T['plotted']), T,
                note='Suggested caption: ' + CAPTION)


def main():
    import build_figure_package as BP
    folder = os.path.join(BP.PKG, 'Figure_S4'); index = os.path.join(BP.PKG, 'panel_index.csv')
    if not os.path.isfile(index) or not os.path.isdir(os.path.join(folder, 'Panel_A')):
        raise SystemExit(f'No figure package at {BP.PKG}; run from nFCM Manuscript/scripts or set NFCM_PACKAGE_DIR.')
    write(BP.write_panel)

    ix = pd.read_csv(index, dtype=str, keep_default_na=False)
    ix = ix[~((ix.figure == 'S4') & (ix.panel == 'B'))].reset_index(drop=True)
    at = ix.index[(ix.figure == 'S4') & (ix.panel == 'A')][0] + 1
    pd.concat([ix.iloc[:at], pd.DataFrame(BP.INDEX[-1:]), ix.iloc[at:]]).to_csv(index, index=False)

    readme = os.path.join(folder, 'README.txt')
    with open(readme, encoding='utf8') as fh:
        lines = [x for x in fh.read().splitlines() if not x.startswith('  B  ')]
    lines.insert(next(i for i, x in enumerate(lines) if x.startswith('  A  ')) + 1, f'  B  {TITLE}  [data]')
    with open(readme, 'w', encoding='utf8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
