"""Probe accessibility and non-specific binding metrics.

Every metric here is computed on the gated monomer population (see
:mod:`nfcm.gating`) and referenced to a control measured in the *same coating
and the same replicate*. That within-condition referencing is what makes the
coatings comparable to each other despite their very different backgrounds: a
coating that shifts the probe channel for everything, biotin or not, cancels
out of the ratio.

Three quantities appear in the manuscript.

``total_probe_association``
    How much probe lands on no-biotin origami at all. This is the non-specific
    floor, not an accessibility measure.

``method1_median_shift``
    Accessibility as a shift in the median probe signal. Sensitive to a
    population-wide change, blind to a small bright subpopulation.

``method2_positive_fraction``
    Accessibility as the fraction of monomers that cross a high threshold set
    on the no-biotin control. Sensitive to a small bright subpopulation, blind
    to a modest population-wide shift.

The two accessibility metrics answer the same question with opposite failure
modes, so the paper reports both rather than choosing.
"""
from __future__ import annotations

import numpy as np

from .constants import METHOD2_PERCENTILE


def total_probe_association(
    probe_signal: np.ndarray, no_probe_signal: np.ndarray
) -> float:
    """Log2 ratio of median probe channel, with probe versus without.

    Both populations are no-biotin monomers of the same coating and replicate,
    so the only difference is whether probe was added. A large value means the
    probe binds the surface even with no biotin present.

    Returns ``log2(median(probe) / median(no_probe))``.
    """
    return float(
        np.log2(np.median(probe_signal) / np.median(no_probe_signal))
    )


def method1_median_shift(
    biotin_signal: np.ndarray, no_biotin_signal: np.ndarray
) -> float:
    """Method 1: log2 shift in median probe signal from adding a biotin handle.

    Both inputs are probe-incubated monomers from the same coating and the same
    replicate, differing only in whether the origami carries a biotin extension.
    Referencing to the no-biotin sample of the same coating removes that
    coating's non-specific background, so the residual is biotin-driven.

    Returns ``log2(median(biotin) / median(no_biotin))``. Zero means the handle
    added nothing detectable.
    """
    return float(
        np.log2(np.median(biotin_signal) / np.median(no_biotin_signal))
    )


def method2_threshold(
    no_biotin_signal: np.ndarray, percentile: float = METHOD2_PERCENTILE
) -> float:
    """Positivity threshold for Method 2, from the no-biotin control.

    Taking the 99th percentile of the same-coating same-replicate no-biotin
    sample fixes the false-positive rate at 1% by construction, so any excess
    above 1% in a biotin-carrying sample is the signal.
    """
    return float(np.percentile(no_biotin_signal, percentile))


def method2_positive_fraction(biotin_signal: np.ndarray, threshold: float) -> float:
    """Method 2: percentage of monomers above the no-biotin threshold.

    Because :func:`method2_threshold` is a 99th percentile, 1% is the null
    value. A result near 1% means no detectable biotin-specific binding.
    """
    return float(100.0 * (biotin_signal > threshold).mean())


def nonspecific_reduction(
    coated_signal: np.ndarray, uncoated_signal: np.ndarray
) -> float:
    """Log2 change in non-specific probe binding relative to uncoated origami.

    Both inputs are no-biotin, probe-incubated monomers from the same replicate.
    Negative means the coating suppressed non-specific binding.
    """
    return float(
        np.log2(np.median(coated_signal) / np.median(uncoated_signal))
    )


def summarise(values, ddof: int = 1) -> dict[str, float]:
    """Mean, standard error, and n for a set of per-replicate values.

    Standard error uses the sample standard deviation, matching how the
    manuscript reports ``mean +/- SEM`` across biological replicates. With a
    single value the error is reported as zero rather than as NaN.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    sem = float(np.std(arr, ddof=ddof) / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": float(arr.mean()), "sem": sem, "n": n}
