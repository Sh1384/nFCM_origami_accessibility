"""The two-component monomer gate.

Each coating changes both how much light an origami scatters and how brightly
its interior dyes read out. A single gate drawn on uncoated origami therefore
under-samples the coated populations and biases every downstream comparison. So
the gate is derived separately for each coating, from that coating's own
within-session controls, and no threshold is transferred across coatings or
across acquisition sessions.

The gate has two components, applied together:

**Side-scatter window.** The interquartile range (P25 to P75) of
:data:`~nfcm.constants.CH_SS` in that coating's pooled no-probe control, after
discarding events below :data:`~nfcm.constants.SS_NOISE_FLOOR`. The floor keeps
the lower quartile out of the electronic noise band; the upper quartile excludes
the long aggregation tail. Note the floor matters only while *deriving* the
window, since every resulting lower bound already sits above it.

**V525-H lower bound.** Chosen by a peak-enrichment sweep against a matched
coating-only debris control (:func:`peak_enrichment_threshold`). There is no
upper bound: the side-scatter window already excludes doublets and aggregates by
size, and the few bright contaminants that survive it are too rare to move a
median or a 99th percentile.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import (
    CH_ATTO,
    CH_SS,
    COATINGS,
    MIN_SIGNAL_RETENTION,
    REPLICATES,
    SESSIONS,
    SS_NOISE_FLOOR,
)
from .io import debris_path, load_channel, sample_path
from .paths import CONFIG_DIR

#: Key prefixes used inside ``config/gates.json``. ``S`` is the streptavidin
#: session and ``A`` the antibody session.
_SESSION_KEY = {"streptavidin": "S", "antibody": "A"}


# ---------------------------------------------------------------- frozen gates
def load_gates(path: str | Path | None = None) -> dict:
    """Load the gate definitions the manuscript was built on.

    These values are frozen inputs: every number in the paper was computed with
    them. :mod:`scripts.00_derive_gates` re-derives them from raw data and
    reports agreement, but does not overwrite this file.
    """
    path = Path(path) if path is not None else CONFIG_DIR / "gates.json"
    with open(path) as fh:
        return json.load(fh)


def gate_for(gates: dict, session: str, coating: str) -> tuple[tuple[float, float], float]:
    """Return ``((ss_low, ss_high), v525_low)`` for one session and coating."""
    key = _SESSION_KEY[session]
    ss_low, ss_high = gates[f"SS_{key}"][coating]
    return (float(ss_low), float(ss_high)), float(gates[f"V525_{key}_LO"][coating])


def apply_gate(df, ss_window: tuple[float, float], v525_low: float):
    """Restrict a DataFrame of events to the monomer population.

    Applies the side-scatter window and the V525-H lower bound together. This is
    the population every metric in the paper is computed on.
    """
    low, high = ss_window
    mask = (
        (df[CH_SS] >= low)
        & (df[CH_SS] <= high)
        & (df[CH_ATTO] >= v525_low)
    )
    return df[mask]


# ---------------------------------------------------------------- derivation
def side_scatter_window(
    session: str,
    coating: str,
    *,
    replicates: list[str] | None = None,
    noise_floor: float = SS_NOISE_FLOOR,
) -> tuple[float, float]:
    """Derive the side-scatter window for one coating.

    Pools the no-probe controls across replicates, drops events below
    ``noise_floor``, and returns the interquartile range.
    """
    replicates = replicates or REPLICATES
    pooled = []
    for rep in replicates:
        path = sample_path(session, "N", coating, rep, with_probe=False)
        if path.exists():
            pooled.append(load_channel(path, CH_SS, ss_floor=noise_floor))
    if not pooled:
        raise FileNotFoundError(
            f"no no-probe controls found for {coating!r} in session {session!r}"
        )
    low, high = np.percentile(np.concatenate(pooled), [25, 75])
    return float(low), float(high)


def peak_enrichment_threshold(
    signal: np.ndarray,
    debris: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    min_retention: float = MIN_SIGNAL_RETENTION,
) -> dict[str, float]:
    """Pick the V525-H lower bound that best separates origami from debris.

    Sweeps candidate thresholds and, at each one, measures what fraction of the
    origami signal survives and what fraction of the matched debris leaks
    through. Enrichment is the ratio of the two. The chosen threshold maximises
    enrichment among candidates that retain at least ``min_retention`` of the
    signal.

    The retention floor is what stops the argmax from running off to the extreme
    tail, where debris pass-rate reaches zero and enrichment diverges on a
    handful of events.

    Parameters
    ----------
    signal
        V525-H of no-biotin, probe-incubated origami, already restricted to the
        coating's side-scatter window.
    debris
        V525-H of the matched coating-only control, restricted to the same
        window so the two are directly comparable.
    grid
        Candidate thresholds. Defaults to 400 log-spaced points over 50 to 500,
        which brackets the ATTO 425 population in every condition measured here.
    min_retention
        Minimum surviving fraction of ``signal``.

    Returns
    -------
    dict
        ``threshold``, ``enrichment``, ``signal_retention``, ``debris_pass``.
    """
    if grid is None:
        grid = np.logspace(np.log10(50), np.log10(500), 400)

    retention = np.array([(signal > t).mean() for t in grid])
    leak = np.array([(debris > t).mean() for t in grid])
    # Floor the denominator so a debris-free threshold yields a large finite
    # enrichment rather than a divide-by-zero.
    enrichment = retention / np.maximum(leak, 1e-6)

    eligible = retention >= min_retention
    if not eligible.any():
        raise ValueError(
            f"no candidate threshold retains {min_retention:.0%} of the signal"
        )
    best = int(np.argmax(np.where(eligible, enrichment, -np.inf)))
    return {
        "threshold": float(grid[best]),
        "enrichment": float(enrichment[best]),
        "signal_retention": float(retention[best]),
        "debris_pass": float(leak[best]),
    }


def derive_v525_threshold(
    session: str,
    coating: str,
    ss_window: tuple[float, float],
    *,
    replicates: list[str] | None = None,
    min_retention: float = MIN_SIGNAL_RETENTION,
) -> dict[str, float]:
    """Run :func:`peak_enrichment_threshold` on one session and coating.

    Signal is the pooled no-biotin with-probe acquisitions; debris is the
    matched coating-only control. Both are restricted to ``ss_window`` so the
    two pass-rates describe the same size population.
    """
    replicates = replicates or REPLICATES
    signal = []
    for rep in replicates:
        path = sample_path(session, "N", coating, rep, with_probe=True)
        if path.exists():
            signal.append(load_channel(path, CH_ATTO, ss_window=ss_window))
    if not signal:
        raise FileNotFoundError(
            f"no with-probe controls found for {coating!r} in session {session!r}"
        )
    debris = load_channel(debris_path(session, coating), CH_ATTO, ss_window=ss_window)
    return peak_enrichment_threshold(
        np.concatenate(signal), debris, min_retention=min_retention
    )


def derive_all_gates(*, min_retention: float = MIN_SIGNAL_RETENTION) -> dict:
    """Re-derive every gate from raw data, in the layout of ``config/gates.json``.

    Also returns a ``diagnostics`` block holding the enrichment, signal
    retention, and debris pass-rate behind each threshold, which is what
    Table 1 of the manuscript reports.
    """
    out: dict = {"diagnostics": {}}
    for session in SESSIONS:
        key = _SESSION_KEY[session]
        out[f"SS_{key}"] = {}
        out[f"V525_{key}_LO"] = {}
        for coating in COATINGS:
            window = side_scatter_window(session, coating)
            sweep = derive_v525_threshold(
                session, coating, window, min_retention=min_retention
            )
            out[f"SS_{key}"][coating] = [round(window[0]), round(window[1])]
            out[f"V525_{key}_LO"][coating] = round(sweep["threshold"], 1)
            out["diagnostics"][f"{session}/{coating}"] = sweep
    return out
