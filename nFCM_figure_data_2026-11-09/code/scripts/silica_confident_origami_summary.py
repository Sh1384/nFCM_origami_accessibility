"""Silica vs bare origami side scatter after removing debris by ATTO 425 fluorescence (no-probe samples, both days).

Origami-confident events: scatter-triggered events (VSSC-1-H >= 200) whose V525-H (ATTO 425) exceeds the P95 (or P99)
of pooled background in the same side-scatter bin (12 log bins, 200-20,000); background = session buffer + silica
precursor only control (coatings_controls/SilicaChemicalsOnly.fcs, acquired 02/13, not same day). Same gating as
make_manuscript_figures.coating_confident_fig (fig20).
False positives: background particles that pass the threshold are expected at (1 - q) x the excess scatter-triggered
event rate of silica over bare origami (same session), i.e. if the excess events were background particles.
Day labels as in the manuscript: day 1 = 05/25 (streptavidin session), day 2 = 04/18 (antibody session).
Runs above 7,000 events/s are excluded (shared_utils.load_fcs).

Run:  python silica_confident_origami_summary.py
Out:  figures_svg/silica_confident_origami_replicates.csv, figures_svg/silica_confident_origami_summary.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import ttest_ind
import make_manuscript_figures as M
from shared_utils import CH_SS, CH_ATTO, REPLICATES, SS_TRIGGER, acq_time

EDGES = np.logspace(np.log10(SS_TRIGGER), np.log10(20000), 13)
DAY = {'Shervin5_25': 'day 1 (05/25)', 'Shervin4_18': 'day 2 (04/18)'}


def thresholds(bgs, q):
    out = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        v = np.concatenate([bg[(bg[CH_SS] >= a) & (bg[CH_SS] < b)][CH_ATTO].values for bg in bgs])
        out.append(np.percentile(v, q) if len(v) >= 30 else np.nan)
    return np.array(out)


def confident(d, th):
    d = d[(d[CH_SS] >= EDGES[0]) & (d[CH_SS] < EDGES[-1])]
    i = np.clip(np.searchsorted(EDGES, d[CH_SS].values, side='right') - 1, 0, len(th) - 1)
    t = th[i]
    return d[np.isfinite(t) & (d[CH_ATTO].values > t)]


def main():
    ctrl = M.scat(M.L('coatings_controls', 'SilicaChemicalsOnly.fcs'))
    rows = []
    for sname, s in M.SESSIONS.items():
        tag = 'NoAB' if '4_18' in s['folder'] else 'NoS'
        buf = M.scat(M.L(s['folder'], 'Buffer.fcs'))
        for q in (95, 99):
            th = thresholds([buf, ctrl], q)
            for c in ['NoCoating', 'Silica']:
                for r in REPLICATES:
                    d0 = M.L(s['folder'], f'N_{c}_{tag}_{r}.fcs')
                    if d0 is None:
                        continue
                    t = acq_time(d0); d = M.scat(d0); cf = confident(d, th)
                    rows.append(dict(day=DAY[s['folder']], session=sname, threshold=f'P{q}', coating=M.COAT_LBL[c],
                                     replicate=r, file=f'N_{c}_{tag}_{r}.fcs',
                                     median_SS_all_scatter_triggered=float(d[CH_SS].median()),
                                     median_SS_origami_confident=float(cf[CH_SS].median()),
                                     events_per_s_all=len(d) / t, events_per_s_origami_confident=len(cf) / t))
    R = pd.DataFrame(rows)
    summ = []
    for (day, thr), z in R.groupby(['day', 'threshold'], sort=False):
        si, ba = z[z.coating == 'Silica'], z[z.coating == M.COAT_LBL['NoCoating']]
        q = int(thr[1:])
        excess = max(si.events_per_s_all.mean() - ba.events_per_s_all.mean(), 0)
        fp = excess * (1 - q / 100)
        summ.append(dict(day=day, threshold=thr, n_silica=len(si), n_bare=len(ba),
                         median_SS_all_silica=si.median_SS_all_scatter_triggered.mean(),
                         median_SS_all_bare=ba.median_SS_all_scatter_triggered.mean(),
                         median_SS_confident_silica=si.median_SS_origami_confident.mean(),
                         median_SS_confident_bare=ba.median_SS_origami_confident.mean(),
                         ratio_confident_silica_over_bare=si.median_SS_origami_confident.mean() / ba.median_SS_origami_confident.mean(),
                         welch_p_confident=ttest_ind(si.median_SS_origami_confident, ba.median_SS_origami_confident,
                                                     equal_var=False).pvalue,
                         confident_events_per_s_silica=si.events_per_s_origami_confident.mean(),
                         excess_scatter_triggered_events_per_s_silica_over_bare=excess,
                         expected_false_positive_events_per_s=fp,
                         expected_false_positive_fraction=min(fp / si.events_per_s_origami_confident.mean(), 1.0)))
    S = pd.DataFrame(summ)
    for df, name in [(R, 'silica_confident_origami_replicates.csv'), (S, 'silica_confident_origami_summary.csv')]:
        df.to_csv(os.path.join(M.OUT, name), index=False)
        print('wrote', os.path.join(M.OUT, name))
    pd.set_option('display.width', 250)
    print(S.round(3).T.to_string())


if __name__ == '__main__':
    main()
