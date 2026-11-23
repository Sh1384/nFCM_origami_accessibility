"""Quantify antibody-bound origami across the full side-scatter range (Shervin4_18).

The instrument recorded an event if SS >= 200 OR B525 >= 250. Events below SS 200
are therefore fluorescence-triggered: free antibody, plus bound origami that fell
below the scatter trigger. Bound origami are counted in two regions and summed.

  SS >= 200  Shared origami gate (shared_utils: SS >= 200, per-coating SS upper
             limit, compensated-V525 upper limit). Bound = B525 above the
             same-coating, same-replicate no-probe P99 threshold (buffer-
             subtracted). Corrected for (i) the session buffer, gated the same
             way and scaled by acquisition time, (ii) the 1% of unbound origami
             that exceed a P99 threshold by definition, and (iii) free antibody
             clumps inside the gate, scaled from the free-antibody count found
             below SS 200 using the antibody-only control.
  SS <  200  Per SS bin, the V525-H distribution is unmixed (non-negative least
             squares) into a free-antibody shape (AB_Only control, V525 rescaled
             by the ratio of session-buffer medians because it was acquired on
             03/13) and a bound-origami shape (bound origami just above the
             trigger, SS 200-260 and B525 >= 250, pooled per coating).

% bound = (bound_hi + bound_lo) / (net gated origami + bound_lo).
Unbound origami below the scatter trigger are never recorded, so this fraction is
an upper bound; the biotin-specific difference (extension - N) is more robust.
Samples with no net origami above buffer in the gate are excluded (QC).
quantify(threshold='calibrated') replaces the no-antibody P99 with a per-sample threshold, the
median B525-H of that sample's own low-scatter free antibody (see quantify).

Output: figures_rerun/bound_origami_antibody.csv (per sample) and a printed summary.
Import and call quantify() to get the per-sample table.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.optimize import nnls
from shared_utils import (load_fcs, acq_time, bs_percentile, bs_pct_above, FIGURE_DIR, COATINGS, REPLICATES,
                          SS_TRIGGER, compute_monomer_gates, apply_gate)

SS, V, B = 'VSSC-1-H', 'V525-H', 'B525-H'
F = 'Shervin4_18'
EXTS = ['N', '3', '6', '12']
SS_BINS_LO = [0, 50, 100, 150, SS_TRIGGER]
V_BINS = np.r_[np.arange(0, 260, 10), 1e6]


def _hist(v):
    return np.histogram(v, V_BINS)[0].astype(float)


def _load(fname):
    try:
        return load_fcs(F, fname)
    except FileNotFoundError:
        return None


def quantify(verbose=True, threshold='noab_p99'):
    """Per-sample bound-origami table (all samples, with a qc_pass column).

    threshold='noab_p99' (default): same-coating, same-replicate no-antibody P99, one value for all
        extensions; unbound tail = 1% of gated origami; every recorded event below the scatter
        trigger (B525-H >= 250) is above it.
    threshold='calibrated': per sample, the median B525-H of that sample's own low-scatter free
        antibody (SS < 50, B525-H >= 250). The unbound tail is the buffer-subtracted fraction of
        gated no-antibody origami above it, the clump correction uses it, and bound origami below
        the scatter trigger are unmixed from events above it only (free-antibody count for the clump
        correction still from all events below the trigger, as in the default).
    """
    buf = load_fcs(F, 'Buffer.fcs'); tb = acq_time(buf)
    ab = load_fcs('probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs'); tab = acq_time(ab)
    buf0 = load_fcs('probe_only_controls', 'Buffer__Shervin03_13_2026.fcs'); tb0 = acq_time(buf0)
    # V525 baseline correction for the cross-session antibody-only control
    vscale = buf[V].median() / buf0[V].median()
    if verbose:
        print(f'V525 scale applied to AB-only (session buffer / 03-13 buffer medians): {vscale:.3f}')
    abv = ab.assign(**{V: ab[V] * vscale})

    noab = {(c, r): d for c in COATINGS for r in REPLICATES
            if (d := _load(f'N_{c}_NoAB_{r}.fcs')) is not None}
    gates = compute_monomer_gates({('N', c, 'NoAB', r): d for (c, r), d in noab.items()}, ctl='NoAB')

    rows = []
    for c in COATINGS:
        samples = {(e, r): d for e in EXTS for r in REPLICATES
                   if (d := _load(f'{e}_{c}_AB_{r}.fcs')) is not None}
        # bound-origami V525 template: bound origami just above the scatter trigger
        tmpl = pd.concat([d[(d[SS] >= SS_TRIGGER) & (d[SS] < 260) & (d[B] >= 250)][[V, B]]
                          for d in samples.values()])
        T_or = _hist(tmpl[V]); T_or /= T_or.sum()
        bh = apply_gate(buf, c, 'A', gates)
        ab_g = apply_gate(ab, c, 'A', gates); b0_g = apply_gate(buf0, c, 'A', gates)
        for r in REPLICATES:
            if (c, r) not in noab:
                continue
            nh = apply_gate(noab[(c, r)], c, r, gates); tn = acq_time(noab[(c, r)])
            thr_p99 = bs_percentile(nh[B].values, tn, bh[B].values, tb, 99)
            for e in EXTS:
                d = samples.get((e, r))
                if d is None:
                    continue
                t = acq_time(d); lo = d[d[SS] < SS_TRIGGER]; hi = apply_gate(d, c, r, gates)
                free_med = float(d[(d[SS] < 50) & (d[B] >= 250)][B].median())
                if threshold == 'calibrated':
                    thr = free_med
                    tail_frac = bs_pct_above(nh[B].values, tn, bh[B].values, tb, thr) / 100
                else:
                    thr, tail_frac = thr_p99, 0.01                 # 1% of origami exceed own P99
                # --- below the scatter trigger: unmix free antibody vs bound origami
                b_lo = a_lo = b_lo_thr = 0.0; b0 = None
                for s0, s1 in zip(SS_BINS_LO[:-1], SS_BINS_LO[1:]):
                    sel = lo[(lo[SS] >= s0) & (lo[SS] < s1)]
                    y = _hist(sel[V])
                    fa = abv[(abv[SS] >= s0) & (abv[SS] < s1)]; T_ab = _hist(fa[V]); T_ab /= T_ab.sum()
                    coef, _ = nnls(np.vstack([T_ab, T_or]).T, y)
                    a_lo += coef[0]; b_lo += coef[1]
                    if s0 == 0:
                        b0 = coef[1] / max(y.sum(), 1) * 100
                    if threshold == 'calibrated':
                        T_abt = _hist(fa[fa[B] > thr][V]); T_abt /= max(T_abt.sum(), 1)
                        T_ort = _hist(tmpl[tmpl[B] > thr][V]); T_ort /= max(T_ort.sum(), 1)
                        ct, _ = nnls(np.vstack([T_abt, T_ort]).T, _hist(sel[sel[B] > thr][V]))
                        b_lo_thr += ct[1]
                if threshold != 'calibrated':
                    b_lo_thr = b_lo
                # --- inside the gate, buffer-subtracted
                sc = t / tb
                n_hi = len(hi) - len(bh) * sc
                above = (hi[B] > thr).sum() - (bh[B] > thr).sum() * sc
                unbound_tail = tail_frac * n_hi
                # free-antibody clumps in the gate: AB-only ratio (gated & B525 > thr, minus its buffer) / (SS < 200)
                ab_hi = (ab_g[B] > thr).sum() - (b0_g[B] > thr).sum() * tab / tb0
                clump = a_lo * max(ab_hi, 0) / (ab[SS] < SS_TRIGGER).sum()
                b_hi = max(above - unbound_tail - clump, 0)
                tot = n_hi + b_lo
                pct = (lambda v: v / n_hi * 100 if n_hi > 0 else np.nan)
                rows.append(dict(coating=c, ext=e, rep=r, thr=thr, n_origami_hiSS=n_hi,
                                 bound_hiSS=b_hi, bound_loSS=b_lo_thr, bound_loSS_all=b_lo, free_ab=a_lo,
                                 clump_corr=clump, free_ab_median_B525=free_med,
                                 raw_gated_pct=pct(above), unbound_tail_pct=pct(unbound_tail), clump_pct=pct(clump),
                                 pct_bound=(b_hi + b_lo_thr) / tot * 100 if tot > 0 else np.nan,
                                 pct_bound_hiSS_only=pct(b_hi),
                                 frac_bound_from_loSS=b_lo_thr / max(b_hi + b_lo_thr, 1) * 100,
                                 origami_in_SS0_50_pct=b0, bound_eps=(b_hi + b_lo_thr) / t))
    R = pd.DataFrame(rows)
    R['qc_pass'] = R.n_origami_hiSS > 0
    if verbose:
        for _, q in R[~R.qc_pass].iterrows():
            print(f'QC: excluded {q.ext}_{q.coating}_AB_{q.rep} (no net origami above buffer in gate)')
    return R


def with_delta(R):
    """QC-passing rows with delta_vs_N = % bound (ext) - % bound (N), same replicate."""
    R = R[R.qc_pass].copy()
    n_ref = R[R.ext == 'N'].set_index(['coating', 'rep']).pct_bound
    R['delta_vs_N'] = [p - n_ref.get((c, r), np.nan) for p, c, r in zip(R.pct_bound, R.coating, R.rep)]
    return R


if __name__ == '__main__':
    R = quantify()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    R.to_csv(os.path.join(FIGURE_DIR, 'bound_origami_antibody.csv'), index=False)
    R = with_delta(R)
    pd.set_option('display.width', 250)
    agg = R.groupby(['coating', 'ext'], sort=False).agg(
        pct_bound=('pct_bound', 'mean'), sem=('pct_bound', 'sem'),
        pct_bound_hiSS_only=('pct_bound_hiSS_only', 'mean'),
        share_from_loSS=('frac_bound_from_loSS', 'mean'),
        bound_eps=('bound_eps', 'mean'),
        delta_vs_N=('delta_vs_N', 'mean'), delta_sem=('delta_vs_N', 'sem'), n=('rep', 'count'),
        loSS_0_50_origami=('origami_in_SS0_50_pct', 'mean'))
    print(agg.to_string(float_format='%.2f'))
    print('\nfree-antibody clump correction as % of raw gated positives:',
          (R.clump_corr / (R.bound_hiSS + R.clump_corr + 0.01 * R.n_origami_hiSS) * 100)
          .describe()[['mean', 'min', 'max']].round(1).to_dict())
