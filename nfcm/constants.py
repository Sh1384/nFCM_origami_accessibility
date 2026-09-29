"""Experimental design constants: channels, conditions, labels, and colours.

Importing these from one place keeps every figure in the paper visually
consistent and stops condition names from drifting between scripts.
"""
from __future__ import annotations

# ---------------------------------------------------------------- detectors
#: Violet side scatter, height. The primary particle-size proxy and the axis the
#: monomer side-scatter window is defined on.
CH_SS = "VSSC-1-H"

#: Violet 525 nm, height. Reads the eight ATTO 425 dyes carried inside every
#: Pandora box, so it reports origami identity independently of probe binding.
CH_ATTO = "V525-H"

#: Blue 525 nm, height. Reads AF488 on whichever probe was used, so it reports
#: probe binding. Both probes are AF488-labelled and share this channel.
CH_PROBE = "B525-H"

#: All nine side-scatter axes the instrument records per event: three detectors
#: (two violet, one blue) x three pulse statistics (height, area, width). The
#: per-event classifier uses this full set and no fluorescence.
SS_AXES: list[str] = [
    "VSSC-1-H", "VSSC-1-A", "VSSC-1-W",
    "VSSC-2-H", "VSSC-2-A", "VSSC-2-W",
    "BSSC-H",   "BSSC-A",   "BSSC-W",
]

#: Fluorescence axes used only in the ablation that asks how much the classifier
#: gains from labels it would not have on an unknown sample.
FLUOR_AXES: list[str] = [CH_ATTO, CH_PROBE]

CH_TITLES: dict[str, str] = {
    CH_SS:    "VSSC-1-H (side scatter)",
    CH_ATTO:  "V525-H (ATTO 425)",
    CH_PROBE: "B525-H (AF488)",
}


# ---------------------------------------------------------------- conditions
#: Surface conditions, in the order they appear in every figure and table.
#: Internal keys are short; see :data:`COATING_LABELS` for display strings.
COATINGS: list[str] = ["NoCoating", "PLL", "Silica"]

COATING_LABELS: dict[str, str] = {
    "NoCoating": "Uncoated",
    "PLL":       "PLL-PEG",
    "Silica":    "Silica",
}

#: Biotin handle lengths. ``"N"`` carries no biotin and is the within-coating
#: negative control that both accessibility metrics reference.
EXTENSIONS: list[str] = ["N", "3", "6", "12"]

EXTENSION_LABELS: dict[str, str] = {
    "N":  "N (no biotin)",
    "3":  "3 nt",
    "6":  "6 nt",
    "12": "12 nt",
}

#: Biological replicates. Replicate C is held out when training classifiers.
REPLICATES: list[str] = ["A", "B", "C"]

#: Probe sessions. ``no_probe``/``probe`` are the filename tags that distinguish
#: a control acquisition from a probe-incubated one within each session.
SESSIONS: dict[str, dict[str, str]] = {
    "streptavidin": {
        "dataset":  "streptavidin",
        "no_probe": "NoS",
        "probe":    "S",
        "label":    "Streptavidin",
    },
    "antibody": {
        "dataset":  "antibody",
        "no_probe": "NoAB",
        "probe":    "AB",
        "label":    "Anti-biotin IgG",
    },
}

#: Coating-only debris reference for each coating, as a
#: ``(dataset key, filename)`` pair. Uncoated origami is referenced to the
#: buffer acquired in its own session, so its entry is resolved per session by
#: :func:`nfcm.io.debris_path`; the two coatings have dedicated chemistry-only
#: acquisitions that are shared across sessions.
DEBRIS_FILES: dict[str, tuple[str, str]] = {
    "PLL":    ("coating_only", "PLL_PEG_NoOrigami.fcs"),
    "Silica": ("coating_only", "SilicaChemicalsOnly.fcs"),
}


# ---------------------------------------------------------------- appearance
COATING_COLORS: dict[str, str] = {
    "NoCoating": "#1F77B4",
    "PLL":       "#2CA02C",
    "Silica":    "#FF7F0E",
}

EXTENSION_COLORS: dict[str, str] = {
    "N":  "#4D4D4D",
    "3":  "#377EB8",
    "6":  "#4DAF4A",
    "12": "#E41A1C",
}

REPLICATE_MARKERS: dict[str, str] = {"A": "o", "B": "s", "C": "^"}


# ---------------------------------------------------------------- thresholds
#: Side-scatter floor applied before deriving the monomer window. Below this the
#: record is dominated by electronic noise, which would drag the lower quartile
#: down into the noise band and make the window meaningless.
SS_NOISE_FLOOR = 300

#: Minimum fraction of origami signal a candidate V525-H threshold must retain
#: during the peak-enrichment sweep. Without it the enrichment argmax drifts onto
#: the extreme tail, where too few events survive to support stable statistics.
MIN_SIGNAL_RETENTION = 0.05

#: Percentile of the no-biotin with-probe control that defines the Method 2
#: positive threshold. A 99th percentile makes the null rate 1% by construction.
METHOD2_PERCENTILE = 99

#: Chance accuracy for the three-class coating classifiers.
CHANCE_LEVEL = 1 / 3
