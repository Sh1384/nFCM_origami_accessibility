#!/usr/bin/env python3
"""Compute every per-replicate accessibility and non-specific binding metric.

This is the quantitative core of the paper. It walks both probe sessions, all
three coatings, all four biotin extensions, and all three biological
replicates, applies the monomer gate from ``config/gates.json``, and writes one
row per condition.

Outputs
-------
``results/main/method1_method2_metrics.csv``
    One row per (session, coating, extension, replicate) with the gated event
    count, the median probe signal, the two accessibility metrics, and the
    control values each was referenced to.
``results/main/nonspecific_binding_reduction.csv``
    One row per (session, coating, replicate) giving the change in non-specific
    probe binding relative to uncoated origami of the same replicate.
``results/main/summary_by_condition.csv``
    The same metrics collapsed to mean +/- SEM across replicates, which is the
    form Table 3 of the manuscript reports.

Run::

    python scripts/01_compute_metrics.py

Reproduces the published values exactly; the headline silica 6 nt streptavidin
result is Method 1 = +0.456 and Method 2 = 14.27 percent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from nfcm import constants as K
from nfcm.gating import apply_gate, gate_for, load_gates
from nfcm.io import read_fcs, sample_path
from nfcm.metrics import (
    method1_median_shift,
    method2_positive_fraction,
    method2_threshold,
    nonspecific_reduction,
    summarise,
    total_probe_association,
)
from nfcm.paths import RESULTS_MAIN, require_data_root

#: A condition needs at least this many gated monomers to contribute a row. The
#: bound only screens out degenerate cases: the sparsest condition the published
#: analysis retains has 84 gated monomers, and the one condition that drops out
#: (PLL-PEG, 12 nt, antibody, replicate C) does so because its gate yields
#: exactly zero events, which is why the manuscript reports n = 2 for that cell.
MIN_MONOMERS = 50

#: Short session tags used in the output CSV. These match the published data
#: files, so figure scripts downstream read this file without translation.
SESSION_TAG = {"streptavidin": "strep", "antibody": "ab"}


def gated_probe_signal(session, extension, coating, replicate, *, with_probe, gates):
    """Return the probe channel of gated monomers, or None if the file is absent."""
    path = sample_path(session, extension, coating, replicate, with_probe=with_probe)
    if not path.exists():
        return None
    window, v525_low = gate_for(gates, session, coating)
    monomers = apply_gate(read_fcs(path), window, v525_low)
    return monomers[K.CH_PROBE].to_numpy()


def main() -> int:
    require_data_root()
    gates = load_gates()

    rows: list[dict] = []
    # Cache the no-biotin references; each is reused by three extensions.
    no_biotin_probe: dict[tuple[str, str, str], np.ndarray] = {}

    for session in K.SESSIONS:
        tag = SESSION_TAG[session]
        for coating in K.COATINGS:
            for replicate in K.REPLICATES:
                # --- the two within-condition controls -----------------------
                ref_probe = gated_probe_signal(
                    session, "N", coating, replicate, with_probe=True, gates=gates
                )
                ref_no_probe = gated_probe_signal(
                    session, "N", coating, replicate, with_probe=False, gates=gates
                )
                if ref_probe is None or len(ref_probe) < MIN_MONOMERS:
                    print(f"  skip {session}/{coating}/{replicate}: "
                          f"no usable no-biotin control")
                    continue
                no_biotin_probe[(session, coating, replicate)] = ref_probe

                threshold = method2_threshold(ref_probe)
                association = (
                    total_probe_association(ref_probe, ref_no_probe)
                    if ref_no_probe is not None and len(ref_no_probe) > 0
                    else np.nan
                )

                # --- each extension against those controls -------------------
                for extension in K.EXTENSIONS:
                    signal = gated_probe_signal(
                        session, extension, coating, replicate,
                        with_probe=True, gates=gates,
                    )
                    if signal is None or len(signal) < MIN_MONOMERS:
                        continue
                    rows.append({
                        "session": tag,
                        "coating": coating,
                        "extension": extension,
                        "replicate": replicate,
                        "n_events": len(signal),
                        "B525_median": float(np.median(signal)),
                        "N_noprobe_median": (
                            float(np.median(ref_no_probe))
                            if ref_no_probe is not None and len(ref_no_probe)
                            else np.nan
                        ),
                        "N_probe_median": float(np.median(ref_probe)),
                        "N_probe_P99": threshold,
                        "log2fc_total": association,
                        "M1_biotin_MFI_shift": method1_median_shift(signal, ref_probe),
                        "M2_biotin_specific_fraction": method2_positive_fraction(
                            signal, threshold
                        ),
                    })

    metrics = pd.DataFrame(rows)
    RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_MAIN / "method1_method2_metrics.csv", index=False)
    print(f"\nwrote {RESULTS_MAIN / 'method1_method2_metrics.csv'} "
          f"({len(metrics)} rows)")

    # ---------------------------------------------------------- non-specific
    ns_rows = []
    for session in K.SESSIONS:
        for coating in K.COATINGS:
            if coating == "NoCoating":
                continue
            for replicate in K.REPLICATES:
                coated = no_biotin_probe.get((session, coating, replicate))
                uncoated = no_biotin_probe.get((session, "NoCoating", replicate))
                if coated is None or uncoated is None:
                    continue
                ns_rows.append({
                    "session": SESSION_TAG[session],
                    "coating": coating,
                    "replicate": replicate,
                    "reduction_log2": nonspecific_reduction(coated, uncoated),
                    "fold_change": float(
                        np.median(coated) / np.median(uncoated)
                    ),
                })
    ns = pd.DataFrame(ns_rows)
    ns.to_csv(RESULTS_MAIN / "nonspecific_binding_reduction.csv", index=False)
    print(f"wrote {RESULTS_MAIN / 'nonspecific_binding_reduction.csv'} "
          f"({len(ns)} rows)")

    # ---------------------------------------------------------------- summary
    summary = []
    for (session, coating, extension), grp in metrics.groupby(
        ["session", "coating", "extension"], sort=False
    ):
        m1 = summarise(grp.M1_biotin_MFI_shift)
        m2 = summarise(grp.M2_biotin_specific_fraction)
        summary.append({
            "session": session,
            "coating": coating,
            "extension": extension,
            "n_replicates": m1["n"],
            "method1_mean": m1["mean"], "method1_sem": m1["sem"],
            "method2_mean": m2["mean"], "method2_sem": m2["sem"],
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_MAIN / "summary_by_condition.csv", index=False)
    print(f"wrote {RESULTS_MAIN / 'summary_by_condition.csv'} "
          f"({len(summary_df)} rows)")

    # ------------------------------------------------------------- key result
    key = summary_df[
        (summary_df.session == "strep")
        & (summary_df.coating == "Silica")
        & (summary_df.extension == "6")
    ]
    if not key.empty:
        r = key.iloc[0]
        print(
            f"\nkey result, silica 6 nt streptavidin: "
            f"Method 1 = {r.method1_mean:+.3f} +/- {r.method1_sem:.3f}, "
            f"Method 2 = {r.method2_mean:.2f} +/- {r.method2_sem:.2f} percent "
            f"(published: +0.456, 14.27 percent)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
