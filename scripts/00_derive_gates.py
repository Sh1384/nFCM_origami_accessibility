#!/usr/bin/env python3
"""Re-derive the monomer gate from raw data and compare against the frozen file.

The manuscript was computed with the gate values in ``config/gates.json``. Those
are treated as frozen inputs, so this script never overwrites them. It exists to
document the derivation and to make the agreement auditable: it recomputes every
window and threshold from the FCS archive, writes the result to
``results/main/gates_rederived.json``, and prints a side-by-side comparison.

Run::

    python scripts/00_derive_gates.py

Expected outcome, with the archive as published: all six side-scatter windows
and all six V525-H thresholds reproduce exactly.

Reproducing the thresholds requires matching each session's sweep
configuration, since the two published sessions were derived differently. Those
configurations are recorded in ``nfcm.gating.SWEEP_CONFIG`` and explained in
``docs/GATING.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repository root importable so `nfcm` resolves whether or not the
# package has been pip-installed. Keeping this here means the scripts run
# straight from a fresh clone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfcm import constants as K
from nfcm.gating import (
    _SESSION_KEY,
    derive_v525_threshold,
    gate_for,
    load_gates,
    side_scatter_window,
)
from nfcm.paths import RESULTS_MAIN, require_data_root


def main() -> int:
    require_data_root()
    frozen = load_gates()

    derived: dict = {"diagnostics": {}}
    rows = []

    for session in K.SESSIONS:
        key = _SESSION_KEY[session]
        derived[f"SS_{key}"] = {}
        derived[f"V525_{key}_LO"] = {}

        for coating in K.COATINGS:
            window = side_scatter_window(session, coating)
            sweep = derive_v525_threshold(session, coating, window)

            derived[f"SS_{key}"][coating] = [round(window[0]), round(window[1])]
            derived[f"V525_{key}_LO"][coating] = round(sweep["threshold"], 1)
            derived["diagnostics"][f"{session}/{coating}"] = sweep

            frozen_window, frozen_v525 = gate_for(frozen, session, coating)
            rows.append({
                "session": session,
                "coating": coating,
                "ss_derived": (round(window[0]), round(window[1])),
                "ss_frozen": (int(frozen_window[0]), int(frozen_window[1])),
                "v525_derived": sweep["threshold"],
                "v525_frozen": frozen_v525,
                "enrichment": sweep["enrichment"],
                "retention": sweep["signal_retention"],
                "debris_pass": sweep["debris_pass"],
            })

    RESULTS_MAIN.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_MAIN / "gates_rederived.json"
    with open(out_path, "w") as fh:
        json.dump(derived, fh, indent=2)

    # ------------------------------------------------------------ report
    print(f"{'session':<13} {'coating':<10} {'SS derived':>14} {'SS frozen':>14} "
          f"{'V525 der':>9} {'V525 frz':>9} {'enrich':>10} {'retain':>8}")
    print("-" * 94)
    ss_ok = v525_ok = 0
    for r in rows:
        ss_match = r["ss_derived"] == r["ss_frozen"]
        v_match = abs(r["v525_derived"] - r["v525_frozen"]) < 1.0
        ss_ok += ss_match
        v525_ok += v_match
        print(
            f"{r['session']:<13} {r['coating']:<10} "
            f"{str(r['ss_derived']):>14} {str(r['ss_frozen']):>14} "
            f"{r['v525_derived']:>9.1f} {r['v525_frozen']:>9.1f} "
            f"{r['enrichment']:>10.4g} {r['retention']*100:>7.1f}%"
            f"{'' if (ss_match and v_match) else '   <- differs'}"
        )

    n = len(rows)
    print("-" * 94)
    print(f"side-scatter windows reproduced: {ss_ok}/{n}")
    print(f"V525-H thresholds reproduced:    {v525_ok}/{n}")
    print(f"\nwrote {out_path}")
    print("config/gates.json is unchanged; it remains the input to every other script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
