"""Serial dilution, step 1: read every FCS file once and write the tables the figure scripts use.

Per file (dilution_files.csv): FileInfo concentration and replicate; acquisition record (start time, tube sequence, duration,
volume, recorded and aborted events, trigger settings, voltages); acquisition time from the Time channel; counts and rates of
all events, scatter-triggered events (VSSC-1-H >= 200), fluorescence-only events (VSSC-1-H < 200, B525-H >= 250);
quantiles of VSSC-1-H, V525-H and VSSC-1-W (pulse width) of scatter-triggered events; coincidence window 2 tau from the
deficit of short inter-event intervals relative to a Poisson process (as in fig29_tccd_coincidence.py); stability of the
scatter-triggered rate over the acquisition (1 s bins).
Histograms (long format, per file): dilution_hist_ss.csv, dilution_hist_v525.csv, dilution_hist_width.csv (scatter-triggered
events), dilution_intervals.csv (all events; observed and Poisson-expected fraction per 16-tick bin), dilution_timecourse.csv.
Run:  python dilution_01_summarize.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import dilution_common as C

SS_EDGES = np.logspace(np.log10(C.SS_TRIG), 6.05, 71)
V_EDGES = np.logspace(0, 6.05, 91)
W_EDGES = np.logspace(0, 4, 81)
BW = 16 * C.TICK                                   # 7.63 us; timestamps cluster on 4/8/16-tick multiples
I_EDGES = np.arange(0, 60) * BW


def intervals(time):
    t = np.sort(time); dt = np.diff(t); R = len(dt) / (t[-1] - t[0])
    obs = np.histogram(dt, I_EDGES)[0] / len(dt)
    exp = np.exp(-R * I_EDGES[:-1]) - np.exp(-R * I_EDGES[1:])
    k = 6                                            # 2 tau = summed deficit over the first 6 bins (46 us)
    two_tau = float(np.sum((1 - obs[:k] / exp[:k]) * BW)) if len(dt) >= 2000 else np.nan
    return obs, exp, R, two_tau


def main():
    fi = C.file_info(); rows, hs, hv, hw, iv, tc = [], [], [], [], [], []
    for r in fi.itertuples():
        path = os.path.join(C.DIL, f'{r.sample}.fcs')
        rec = C.record(path); d = C.load(r.sample)
        t0, t1 = d.Time.min(), d.Time.max(); T = t1 - t0
        ss = d[d[C.SS] >= C.SS_TRIG]; fl = d[(d[C.SS] < C.SS_TRIG) & (d[C.B] >= C.B_TRIG)]
        q = lambda v, p: float(np.percentile(v, p)) if len(v) else np.nan
        obs, exp, R, two_tau = intervals(d.Time.values)
        sec = np.histogram(ss.Time, np.arange(t0, t1 + 1, 1.0))[0]
        sec = sec[:-1] if len(sec) > 2 else sec                      # drop the partial last second
        thirds = np.array_split(sec, 3) if len(sec) >= 6 else [sec] * 3
        rows.append(dict(sample=r.sample, series=r.series, conc_nM=r.conc_nM, replicate=r.replicate, **rec,
                         acq_time_s=T, flow_ul_per_min=60 * rec['volume_ul'] / rec['duration_s'],
                         events=len(d), events_per_s=len(d) / T,
                         ss_events=len(ss), ss_events_per_s=len(ss) / T,
                         fl_only_events=len(fl), fl_only_events_per_s=len(fl) / T,
                         b525_ge_250_per_s=int((d[C.B] >= C.B_TRIG).sum()) / T,
                         aborted_fraction=rec['aborted'] / (rec['aborted'] + rec['events_recorded']),
                         triggers_per_s_incl_aborted=(rec['aborted'] + rec['events_recorded']) / rec['duration_s'],
                         ss_p10=q(ss[C.SS], 10), ss_p25=q(ss[C.SS], 25), ss_median=q(ss[C.SS], 50), ss_p75=q(ss[C.SS], 75),
                         ss_p90=q(ss[C.SS], 90), ss_p99=q(ss[C.SS], 99),
                         frac_ss_lt_250=float((ss[C.SS] < 250).mean()) if len(ss) else np.nan,
                         frac_ss_ge_1e6=float((ss[C.SS] >= 1e6).mean()) if len(ss) else np.nan,
                         v525_p25=q(ss[C.F], 25), v525_median=q(ss[C.F], 50), v525_p75=q(ss[C.F], 75),
                         width_median=q(ss[C.W], 50), width_p75=q(ss[C.W], 75),
                         ss_area_median=q(ss['VSSC-1-A'], 50),
                         interval_rate_per_s=R, two_tau_us=two_tau * 1e6,
                         frac_intervals_lt_16us_obs=float(obs[:2].sum()), frac_intervals_lt_16us_poisson=float(exp[:2].sum()),
                         rate_cv_1s=float(sec.std() / sec.mean()) if sec.mean() > 0 else np.nan,
                         rate_last_over_first_third=float(thirds[2].mean() / thirds[0].mean()) if thirds[0].mean() > 0 else np.nan))
        for lst, col, e in [(hs, C.SS, SS_EDGES), (hv, C.F, V_EDGES), (hw, C.W, W_EDGES)]:
            h = np.histogram(ss[col].clip(lower=e[0]), e)[0]
            lst += [dict(sample=r.sample, bin_lo=a, bin_hi=b, count=n) for a, b, n in zip(e[:-1], e[1:], h)]
        iv += [dict(sample=r.sample, dt_lo_us=a * 1e6, dt_hi_us=b * 1e6, frac_obs=o, frac_poisson=x)
               for a, b, o, x in zip(I_EDGES[:-1], I_EDGES[1:], obs, exp)]
        tc += [dict(sample=r.sample, second=i, ss_events=int(n)) for i, n in enumerate(sec)]
        print(f'  {r.sample:8s} {r.conc_nM:>12g} nM  {len(d):>8d} events  {len(ss) / T:9.1f} ss/s  aborted {rows[-1]["aborted_fraction"]:.3f}')
    S = pd.DataFrame(rows).sort_values('start')
    S['acq_order'] = np.arange(1, len(S) + 1)
    S.to_csv(os.path.join(C.OUT, 'dilution_files.csv'), index=False)
    for df, name in [(hs, 'dilution_hist_ss.csv'), (hv, 'dilution_hist_v525.csv'), (hw, 'dilution_hist_width.csv'),
                     (iv, 'dilution_intervals.csv'), (tc, 'dilution_timecourse.csv')]:
        pd.DataFrame(df).to_csv(os.path.join(C.OUT, name), index=False)
    print('wrote', C.OUT)


if __name__ == '__main__':
    main()
