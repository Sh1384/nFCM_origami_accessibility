# The monomer gate

Every metric in the paper is computed on a gated subpopulation: the events that
are, as far as the instrument can tell, single intact origami of the coating
under test. This document explains how that gate is defined, why it is derived
separately for each coating, and how faithfully the code here reproduces the
values the manuscript used.

Implementation: [`nfcm/gating.py`](../nfcm/gating.py).
Audit script: [`scripts/00_derive_gates.py`](../scripts/00_derive_gates.py).

---

## Why the gate cannot be universal

A coating changes two things at once. It adds material, so the particle
scatters more light and its side-scatter distribution shifts up. It also alters
the local environment of the interior dyes, so the ATTO 425 readout shifts as
well, and not always in the same direction.

The size of these shifts is large. The side-scatter window that describes intact
silica-coated monomers, `[484, 1311]`, barely overlaps the one for uncoated
origami, `[327, 644]`. A single gate drawn on uncoated origami would discard
most silica-coated monomers and keep a biased slice of the rest, and any
comparison across coatings would then be reporting the gate rather than the
sample.

So each coating gets its own gate, derived from its own within-session controls.
No threshold is transferred between coatings or between acquisition sessions.

---

## Component 1: the side-scatter window

For each coating, pool the no-probe control acquisitions across replicates,
discard events below a noise floor of 300, and take the interquartile range
(P25 to P75) of `VSSC-1-H`.

The noise floor keeps the lower quartile out of the electronic noise band. Its
effect is large while *deriving* the window: without it the lower quartile
collapses by 115 to 222 units and the uncoated window becomes `[205, 251]`,
which is pure noise. It has no effect when *applying* the window, because all
six derived lower bounds already sit above 300.

The upper quartile cuts the long high-side-scatter tail, which is aggregates and
doublets rather than monomers.

| Session | Uncoated | PLL-PEG | Silica |
|---|---|---|---|
| Streptavidin | [327, 644] | [326, 593] | [484, 1311] |
| Antibody | [328, 527] | [341, 768] | [399, 4710] |

---

## Component 2: the V525-H lower bound

Side scatter alone cannot separate an origami from a similarly sized piece of
coating debris. The interior ATTO 425 dyes can, so a lower bound on `V525-H` is
added, chosen by a **peak-enrichment sweep**.

For each candidate threshold on a log-spaced grid from 50 to 500:

- **Signal retention** is the fraction of no-biotin, probe-incubated origami
  above the threshold.
- **Debris pass-rate** is the fraction of the matched coating-only control above
  it.
- **Enrichment** is retention divided by pass-rate.

Both populations are first restricted to that coating's side-scatter window, so
the two rates describe the same size range and the ratio is meaningful.

The chosen threshold maximises enrichment among candidates retaining at least
**5 percent** of the signal. That floor is load-bearing: without it the argmax
runs to the extreme tail where debris pass-rate hits zero, enrichment diverges,
and fewer than twenty monomers survive per replicate.

| Session | Coating | V525-H bound | Enrichment | Retention |
|---|---|---|---|---|
| Streptavidin | Uncoated | 119.5 | 2.7e5 | 26.5% |
| Streptavidin | PLL-PEG | 145.4 | 40.5 | 12.3% |
| Streptavidin | Silica | 66.7 | 1.1 | 47.4% |
| Antibody | Uncoated | 120.2 | 1.8e5 | 18.0% |
| Antibody | PLL-PEG | 148.8 | 21.7 | 5.8% |
| Antibody | Silica | 181.1 | 123 | 5.1% |

Silica in the streptavidin session is the weak case: enrichment 1.1 means the
threshold barely separates origami from silica debris on that axis. The silica
side-scatter window is doing most of the work there. This is a genuine
limitation of the measurement, not of the code, and the manuscript reports it.

### There is no upper bound

An earlier version of the pipeline bounded `V525-H` from above as well. It was
removed for a concrete reason: probe binding shifts the `V525-H` distribution
*up* by 10 to 30 percent, through direct 405 nm excitation of AF488 and
short-range energy transfer from the interior ATTO 425 dyes. An upper bound set
from the no-probe distribution therefore excluded exactly the probe-bound
origami the experiment is trying to count.

Removing it changes results only slightly, and in the expected direction: silica
6 nt Method 2 moves from 11.31 to 14.27 percent. The side-scatter window already
excludes aggregates by size, and the few bright contaminants that survive it are
too rare to move a median or a 99th percentile.

---

## Applying the gate

```python
from nfcm.gating import load_gates, gate_for, apply_gate
from nfcm.io import read_fcs, sample_path

gates = load_gates()
window, v525_low = gate_for(gates, "streptavidin", "Silica")
events = read_fcs(sample_path("streptavidin", "6", "Silica", "A", with_probe=True))
monomers = apply_gate(events, window, v525_low)
```

Both components apply together: side scatter within the window, and `V525-H` at
or above the lower bound.

---

## Reproduction status

`config/gates.json` holds the gate the manuscript was computed with. It is a
**frozen input**: `scripts/00_derive_gates.py` recomputes everything from raw
data and writes to `results/main/gates_rederived.json`, but never overwrites the
frozen file, so published numbers cannot drift silently.

Running the audit on the published archive gives:

- **All six side-scatter windows reproduce exactly.**
- **Four of six V525-H thresholds reproduce exactly**: both uncoated values, and
  both antibody-session coatings.
- **Two differ**: streptavidin PLL-PEG (frozen 145.4, re-derived 164.2) and
  streptavidin silica (frozen 66.7, re-derived 103.5).

In both differing cases the re-derived threshold would give *higher* enrichment
than the frozen one, so the frozen values are conservative rather than
optimistic. They are also the values the manuscript's own Table 1 reports
enrichments for, and they are internally consistent: evaluating enrichment at
the frozen thresholds returns 2.7e5, 40.5, and 1.1 for the streptavidin session,
matching the published table exactly. The two thresholds appear to predate a
later refinement of the sweep that was applied to the antibody session.

The practical consequence is small. The headline result is robust to this
choice: silica 6 nt streptavidin Method 2 stays between 12.4 and 15.2 percent
across threshold scalings from 0.6x to 1.25x, which spans both values. Figure
S5-2 reports that sensitivity analysis.

Anyone rebuilding the analysis from scratch should use `config/gates.json` to
match the paper, and can consult `results/main/gates_rederived.json` to see what
an unconstrained re-derivation produces.
