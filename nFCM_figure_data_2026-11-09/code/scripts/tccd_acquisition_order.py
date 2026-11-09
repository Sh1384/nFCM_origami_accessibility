"""Acquisition order of the antibody-session samples (04/18), from the start time in each FCS acquisition record.

Per antibody sample (bare, PLL-PEG, silica; no biotin and 3-, 6-, 12-bp stems; replicates A-C):
  start                              StartTime of the acquisition record
  minutes_in_session                 minutes after the first antibody sample of the session
  order_in_set                       position within its coating and replicate set (1 = first acquired)
  minutes_since_first_sample_of_set  minutes after the first sample of its set
  R_A, R_B, observed_over_chance     origami and antibody event rates and observed/chance (fig29_tccd_coincidence.py)
Run after fig29_tccd_coincidence.py:  python tccd_acquisition_order.py
Out:  figures_svg/tccd_acquisition_order.csv (read by fig35, fig37 panel_5f, fig39, tccd_log2fc_delta_U.py)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import make_manuscript_figures as M


def main():
    q = pd.read_csv(os.path.join(M.OUT, 'fig29_tccd_coincidence_samples.csv'), dtype={'extension': str, 'replicate': str})
    q = q[q.probe.str.contains('IgG')].copy()
    folder = M.SESSIONS['Antibody session (04/18)']['folder']
    q['start'] = pd.to_datetime([M.acq_record(folder, f'{s}.fcs')['StartTime'] for s in q['sample']]).tz_localize(None)
    q['minutes_in_session'] = (q.start - q.start.min()).dt.total_seconds() / 60
    q['order_in_set'] = q.groupby(['coating', 'replicate']).start.rank(method='first').astype(int)
    q['minutes_since_first_sample_of_set'] = (q.start - q.groupby(['coating', 'replicate']).start.transform('min')).dt.total_seconds() / 60
    q['observed_over_chance'] = q.obs_exp_deficit
    T = q[['sample', 'coating', 'replicate', 'extension', 'start', 'minutes_in_session', 'order_in_set',
           'minutes_since_first_sample_of_set', 'R_A', 'R_B', 'observed_over_chance', 'valid']].sort_values('start')
    T.to_csv(os.path.join(M.OUT, 'tccd_acquisition_order.csv'), index=False)
    print('wrote tccd_acquisition_order.csv,', len(T), 'samples')


if __name__ == '__main__':
    main()
