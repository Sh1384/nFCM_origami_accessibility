"""Shared matplotlib styling, so every panel in the paper matches.

Call :func:`use_paper_style` once at the top of a figure script. Fonts are
embedded as TrueType (``fonttype 42``) so the vector PDFs stay editable in
Illustrator, which journals ask for.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

#: Panel letters are drawn at this size and weight throughout.
PANEL_LABEL_KW = {"fontsize": 14, "fontweight": "bold", "va": "bottom"}


def use_paper_style() -> None:
    """Apply the manuscript figure style to the global matplotlib state."""
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "axes.linewidth":    0.9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.fontsize":   9,
        "legend.frameon":    False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


def save(fig, directory: str | Path, basename: str) -> list[Path]:
    """Write a figure as both PDF and PNG and report where it went.

    The PDF is the version that goes into the manuscript; the PNG exists so the
    figure can be previewed and embedded in a Word document, which cannot place
    a PDF image.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        path = directory / f"{basename}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
        print(f"  wrote {path}")
    return written


def panel_label(fig, ax, text: str, *, dx: float = -0.055, dy: float = 0.025) -> None:
    """Place a panel letter just outside the top-left corner of an axis.

    Positioning in figure coordinates rather than axis coordinates keeps labels
    aligned across panels of different heights, and keeps them clear of tick
    labels and titles.
    """
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, ha="left", **PANEL_LABEL_KW)
