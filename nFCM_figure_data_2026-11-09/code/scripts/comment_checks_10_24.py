"""Checks behind the draft replies to the 10/24 manuscript comments (antibody session 04/18 unless stated).

1  gate      Sensitivity of the coincidence excess, fold change and delta U to the origami-gate upper bound: the
             per-coating bound used in the paper (510 bare, 810 PLL-PEG) vs a common bound of 510 or 810 for both.
             Same definitions as fig29_tccd_coincidence.py (buffer-subtracted counts) and Equations 3-6.
2  mixed     Linear mixed models of log2 F and of the excess: coating x spacer fixed, replicate set random.
3  quadrant  Fraction of high-scatter (SS+) events that are fluorescence-positive when F+ is set per scatter bin from
             the matched no-origami control (P95 of buffer + coating-only, thresholds of fig20/fig21), both batches.
4  rates     Highest total event rates per coating, the antibody-channel rate range, and recorded concentration
             (events per uL of sample) against the nominal 0.16 nM.
5  settings  Trigger channels, thresholds, logic, detector voltages and flow rate recorded in every file.

Out: figures_svg/comment_checks/*.csv        Run: python comment_checks_10_24.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import json, copy
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import make_manuscript_figures as M
from shared_utils import acq_time, compute_monomer_gates, apply_gate
from fig29_tccd_coincidence import tccd

OUTD = os.path.join(M.OUT, 'comment_checks'); os.makedirs(OUTD, exist_ok=True)
SESS = 'Antibody session (04/18)'; T = 250
WIN = {'w': 8.1e-6}                      # cancels in F and the excess


def tccd_table(gates):
    s = M.SESSIONS[SESS]; dfs = M.session_dfs(s, exts=('N', '3', '6', '12'))
    buf = dfs['Buffer']; tb = acq_time(buf); rows = []
    for c in ['NoCoating', 'PLL']:
        gb = apply_gate(buf, c, 'A', gates)
        for r in 'ABC':
            for e in ['N', '3', '6', '12']:
                d = dfs.get((e, c, s['probe'], r))
                if d is None:
                    continue
                t = acq_time(d); x = tccd(d, t, apply_gate(d, c, r, gates), T, buf, tb, gb, WIN)
                rows.append(dict(coating=M.COAT_LBL[c], replicate=r, extension=e, R_O=x['R_A'], R_Ab=x['R_B'],
                                 R_c=x['N_coinc'] / t, valid=x['valid'], rho_rel=x['obs_exp_w']))
    D = pd.DataFrame(rows)
    N = D[D.extension == 'N'].set_index(['coating', 'replicate'])
    out = []
    for z in D[D.extension != 'N'].itertuples():
        n = N.loc[(z.coating, z.replicate)]
        if not (z.valid and n.valid):
            continue
        F = z.rho_rel / n.rho_rel
        eps = 100 * (z.R_c - n.R_c * (z.R_O * z.R_Ab) / (n.R_O * n.R_Ab)) / z.R_O
        out.append(dict(coating=z.coating, replicate=z.replicate, spacer_nt=int(z.extension), F=F, log2F=np.log2(F), excess_pct=eps))
    return pd.DataFrame(out)


def delta_u(S):
    n = S.groupby(['replicate', 'spacer_nt']).coating.nunique()
    S = S[[n[(r, s)] == 2 for r, s in zip(S.replicate, S.spacer_nt)]]
    g = S.groupby(['replicate', 'coating'])[['excess_pct', 'log2F']].mean().unstack('coating')
    res = {}
    for name, v in [('from excess', np.log(g[('excess_pct', 'Uncoated')] / g[('excess_pct', 'PLL-PEG')])),
                    ('from log2F', np.log((2 ** g[('log2F', 'Uncoated')] - 1) / (2 ** g[('log2F', 'PLL-PEG')] - 1)))]:
        v = v.values; res[name] = dict(mean=v.mean(), sd=v.std(ddof=1), p=stats.ttest_1samp(v, 0).pvalue, sets=np.round(v, 3).tolist())
    return res


def check_gate():
    s = M.SESSIONS[SESS]; dfs = M.session_dfs(s, exts=('N',))
    base = compute_monomer_gates(dfs, ctl=s['noprobe'])
    variants = {'per-coating (paper)': base}
    for hi in ('bare', 'PLL-PEG'):
        g = copy.deepcopy(base); v = base[('NoCoating' if hi == 'bare' else 'PLL', 'A')]['ss_high']
        for k in g:
            g[k]['ss_high'] = v
        variants[f'common upper bound {v:.0f} (= {hi} 95th pct)'] = g
    rows, summ = [], []
    ref = pd.read_csv(os.path.join(M.OUT, 'fig35_antibody_length_dependence_samples.csv'))
    for name, g in variants.items():
        S = tccd_table(g); S['gate'] = name; rows.append(S)
        du = delta_u(S)
        m = S.groupby('coating')[['excess_pct', 'log2F']].mean()
        summ.append(dict(gate=name, ss_high_bare=g[('NoCoating', 'A')]['ss_high'], ss_high_pll=g[('PLL', 'A')]['ss_high'],
                         mean_excess_bare=m.loc['Uncoated', 'excess_pct'], mean_excess_pll=m.loc['PLL-PEG', 'excess_pct'],
                         mean_log2F_bare=m.loc['Uncoated', 'log2F'], mean_log2F_pll=m.loc['PLL-PEG', 'log2F'],
                         dU_excess_mean=du['from excess']['mean'], dU_excess_sd=du['from excess']['sd'], dU_excess_p=du['from excess']['p'],
                         dU_log2F_mean=du['from log2F']['mean'], dU_log2F_sd=du['from log2F']['sd'], dU_log2F_p=du['from log2F']['p']))
        if name.startswith('per-coating'):                      # reproduce the published per-sample values
            r = ref[ref.spacer_nt.notna()].assign(spacer_nt=lambda q: q.spacer_nt.astype(int))
            mm = S.merge(r, on=['coating', 'replicate', 'spacer_nt'])
            print('  reproduction: max |excess diff| %.2e, max |F diff| %.2e (n=%d)' % (
                (mm.excess_pct - mm.excess_pct_of_detected_origami).abs().max(), (mm.F - mm.fold_change_vs_no_biotin).abs().max(), len(mm)))
    pd.concat(rows).to_csv(os.path.join(OUTD, 'gate_sensitivity_samples.csv'), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUTD, 'gate_sensitivity_summary.csv'), index=False)
    return pd.DataFrame(summ)


def check_mixed():
    S = pd.read_csv(os.path.join(OUTD, 'gate_sensitivity_samples.csv'))
    S = S[S.gate == 'per-coating (paper)'].copy(); S['spacer'] = S.spacer_nt.astype(str)
    out = []
    for y in ['log2F', 'excess_pct']:
        for form in [f'{y} ~ C(coating) * C(spacer, Treatment("3"))', f'{y} ~ C(coating) + C(spacer, Treatment("3"))']:
            m = smf.mixedlm(form, S, groups=S['replicate']).fit(reml=True)
            ci = m.conf_int()
            for k in m.params.index:
                if k == 'Group Var':
                    continue
                out.append(dict(response=y, model=form, term=k, estimate=m.params[k], ci_low=ci.loc[k, 0], ci_high=ci.loc[k, 1],
                                p=m.pvalues[k]))
    R = pd.DataFrame(out); R.to_csv(os.path.join(OUTD, 'mixed_models.csv'), index=False)
    return R


def check_quadrant():
    rows = []
    for sess, sub, coat, fig in [('Streptavidin session (05/25)', 'silica_side_scatter', 'Silica', 'fig20_silica_side_scatter_confident_origami'),
                                 ('Antibody session (04/18)', 'silica_side_scatter', 'Silica', 'fig20_silica_side_scatter_confident_origami'),
                                 ('Streptavidin session (05/25)', 'pll_peg_side_scatter', 'PLL', 'fig21_pll_side_scatter_confident_origami'),
                                 ('Antibody session (04/18)', 'pll_peg_side_scatter', 'PLL', 'fig21_pll_side_scatter_confident_origami')]:
        th = pd.read_csv(os.path.join(M.OUT, sub, f'{fig}_thresholds.csv')); th = th[th.session == sess]
        s = M.SESSIONS[sess]
        bare = pd.concat([M.scat(M.L(s['folder'], f"N_NoCoating_{s['noprobe']}_{r}.fcs")) for r in 'ABC'])
        ss_hi = np.percentile(bare['VSSC-1-H'], 95)
        for c in ['NoCoating', coat]:
            for r in 'ABC':
                d = M.L(s['folder'], f"N_{c}_{s['noprobe']}_{r}.fcs")
                if d is None:
                    continue
                x = M.scat(d); hi = x[x['VSSC-1-H'] > ss_hi]
                lo_e, hi_e = th.ss_bin_lo.values, th.ss_bin_hi.values
                i = np.clip(np.searchsorted(hi_e, hi['VSSC-1-H'].values, side='right'), 0, len(th) - 1)
                inr = hi['VSSC-1-H'].values < hi_e[-1]
                fpos = (hi['V525-H'].values > th.P95_threshold.values[i]) & inr
                rows.append(dict(batch='batch 1' if '05/25' in sess else 'batch 2', comparison=f'bare vs {M.COAT_LBL[coat]}',
                                 coating=M.COAT_LBL[c], replicate=r, ss_plus_threshold=ss_hi, pct_events_ss_plus=100 * len(hi) / len(x),
                                 pct_ss_plus_events_F_plus_per_bin=100 * fpos.sum() / max(inr.sum(), 1),
                                 pct_events_ss_plus_F_plus=100 * fpos.sum() / len(x),
                                 ss_plus_events_beyond_threshold_table=int((~inr).sum())))
    Q = pd.DataFrame(rows); Q.to_csv(os.path.join(OUTD, 'quadrants_per_bin_threshold.csv'), index=False)
    return Q


def check_rates_settings():
    rows, sett = [], []
    files = []
    for sname, s in M.SESSIONS.items():
        files.append((sname, s['folder'], 'Buffer.fcs'))
        for c in ['NoCoating', 'PLL', 'Silica']:
            for e, tag in [('N', s['noprobe'])] + [(e, s['probe']) for e in ['N', '3', '6', '12']]:
                for r in 'ABC':
                    files.append((sname, s['folder'], f'{e}_{c}_{tag}_{r}.fcs'))
    files += [('controls', 'probe_only_controls', f) for f in ['AB_Only__Shervin03_13_2026.fcs', 'Buffer__Shervin03_13_2026.fcs',
                                                               'Streptavidin_Only__Shervin5_03.fcs', 'Buffer__Shervin5_03.fcs']]
    files += [('controls', 'coatings_controls', f) for f in ['PLL_PEG_NoOrigami.fcs', 'SilicaChemicalsOnly.fcs']]
    for sname, folder, f in files:
        d = M.L(folder, f)
        if d is None:
            continue
        a = M.acq_record(folder, f); t = acq_time(d)
        ch = a['ExtraProperties']['Channels']; ch = json.loads(ch) if isinstance(ch, str) else ch
        vol = {c['ChannelName']: c['Voltage'] for c in ch}; trig = {c['ChannelName']: c['Trig'] for c in ch}
        rows.append(dict(session=sname, file=f'{folder}/{f}', acq_time_s=t, events_per_s=len(d) / t,
                         scatter_events_per_s=(d['VSSC-1-H'] >= 200).sum() / t, B525_ge_250_per_s=(d['B525-H'] >= 250).sum() / t,
                         B525_ge_150_per_s=(d['B525-H'] >= 150).sum() / t, sample_ul=a['Volume'], flow_ul_per_min=60 * a['Volume'] / a['Duration'],
                         recorded_scatter_events_per_ul=(d['VSSC-1-H'] >= 200).sum() / a['Volume']))
        sett.append(dict(session=sname, file=f'{folder}/{f}', start=a['StartTime'], primary_channel=a['PrimaryChannel'],
                         primary_threshold=a['PrimaryThreshold'], secondary_channel=a['SecondaryChannel'],
                         secondary_threshold=a['SecondaryThreshold'], trigger_logic=a['ThresholdTrigLogic'],
                         trigger_type=a['ThresholdTrigType'], trigger_mode=a['ThresholdTrigMode'], flow_rate_value=a['FlowRateValue'],
                         voltages=';'.join(f'{k}={v}' for k, v in vol.items()), channel_trig=';'.join(f'{k}={v}' for k, v in trig.items()),
                         other_keys=';'.join(k for k in a if k not in ('ExtraProperties', 'Channels'))))
    R, S = pd.DataFrame(rows), pd.DataFrame(sett)
    R.to_csv(os.path.join(OUTD, 'event_rates_all_files.csv'), index=False); S.to_csv(os.path.join(OUTD, 'instrument_settings.csv'), index=False)
    return R, S


def main():
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 120)
    print('== gate sensitivity'); print(check_gate().round(3).T.to_string())
    print('== mixed models'); R = check_mixed(); print(R.round(4).to_string(index=False))
    print('== quadrants'); Q = check_quadrant()
    print(Q.groupby(['batch', 'comparison', 'coating'])[['pct_events_ss_plus', 'pct_ss_plus_events_F_plus_per_bin', 'pct_events_ss_plus_F_plus']].mean().round(2))
    print('== rates/settings'); R, S = check_rates_settings()
    print(R.sort_values('events_per_s', ascending=False).head(8)[['file', 'events_per_s', 'scatter_events_per_s', 'B525_ge_250_per_s']].round(0).to_string(index=False))
    print(S.drop(columns=['file', 'start', 'voltages', 'channel_trig', 'other_keys']).drop_duplicates().to_string(index=False))
    print(S[['session', 'voltages']].drop_duplicates().to_string(index=False)); print(S[['session', 'channel_trig']].drop_duplicates().to_string(index=False))
    print(S.other_keys.iloc[0])


if __name__ == '__main__':
    main()
