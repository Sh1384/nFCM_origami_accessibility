"""Analysis code for nano-flow cytometry of coated DNA origami.

Accompanies the manuscript "Systematic Characterization of Structure and
Function of DNA Nanostructures under Various Coatings by Nano-Flow Cytometer".

The package is deliberately thin: it holds the shared definitions that every
script needs to agree on (detector channels, experimental conditions, the
monomer gate, the accessibility metrics, and the figure style), while the
analyses themselves live as ordered scripts under ``scripts/``.

Typical use::

    from nfcm import constants as K
    from nfcm.gating import load_gates, gate_for, apply_gate
    from nfcm.io import sample_path, read_fcs

    gates = load_gates()
    window, v525_low = gate_for(gates, "streptavidin", "Silica")
    events = read_fcs(sample_path("streptavidin", "6", "Silica", "A", with_probe=True))
    monomers = apply_gate(events, window, v525_low)
"""
__version__ = "1.0.0"

from . import constants, gating, io, metrics, paths, style  # noqa: F401

__all__ = ["constants", "gating", "io", "metrics", "paths", "style"]
