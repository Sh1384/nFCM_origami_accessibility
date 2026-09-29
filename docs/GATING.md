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

For each candidate threshold on a log-spaced grid of 400 points from 50 to 500:

- **Signal retention** is the fraction of origami control events above it.
- **Debris pass-rate** is the fraction of the matched coating-only control above
  it.
- **Enrichment** is retention divided by pass-rate.

The chosen threshold maximises enrichment among candidates retaining at least
**5 percent** of the signal. That floor is load-bearing: without it the argmax
runs to the extreme tail where debris pass-rate hits zero, enrichment diverges,
and fewer than twenty monomers survive per replicate.

### Two sweep configurations

The two probe sessions were derived with different configurations of this sweep.
Both are valid, and both are recorded in `nfcm.gating.SWEEP_CONFIG` so the audit
script reproduces each session exactly.

| | Streptavidin session | Antibody session |
|---|---|---|
| Origami signal | no-probe control | no-biotin with-probe control |
| Events considered | all above the side-scatter noise floor of 300 | only those inside that coating's side-scatter window |

Within either configuration the signal and the debris are filtered identically,
so the two pass-rates always describe the same population and their ratio is
meaningful. The difference between the two is which population that is.

This is a genuine inconsistency in the published derivation rather than a
deliberate design choice: the streptavidin thresholds were computed first, and
the antibody thresholds were computed later with the refined configuration,
without the earlier session being recomputed. It does not affect any published
result, because the thresholds were frozen before the metrics were computed and
both sets are reproduced exactly. It is recorded here so anyone extending the
analysis picks one configuration deliberately instead of inheriting both.

### The resulting thresholds

| Session | Coating | V525-H bound | Enrichment in-window | Retention |
|---|---|---|---|---|
| Streptavidin | Uncoated | 119.5 | 2.7e5 | 26.5% |
| Streptavidin | PLL-PEG | 145.4 | 40.5 | 12.3% |
| Streptavidin | Silica | 66.7 | 1.1 | 47.4% |
| Antibody | Uncoated | 120.2 | 1.8e5 | 18.0% |
| Antibody | PLL-PEG | 148.8 | 21.7 | 5.8% |
| Antibody | Silica | 181.1 | 123 | 5.1% |

The enrichment column is the quantity manuscript Table 1 reports: enrichment
evaluated **at the chosen threshold, within that coating's side-scatter
window**. For the antibody session that is the same quantity the sweep
maximised. For the streptavidin session the sweep maximised the equivalent ratio
over all events above the noise floor instead, so the sweep's own figures there
are 3.0, 6.2, and 1.0. Both describe the same thresholds; they differ only in
which population the ratio is computed over.

Silica in the streptavidin session is the weak case either way: an enrichment
near 1 means the ATTO 425 axis barely separates origami from silica debris, and
the side-scatter window is doing the discriminating work. This is a genuine
limitation of the measurement, and the manuscript reports it.

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

**Everything reproduces exactly.** Running the audit against the published
archive gives:

- **All six side-scatter windows reproduce exactly.**
- **All six V525-H thresholds reproduce exactly**, to full floating-point
  precision against the values recorded at the time of the original derivation
  (for example 145.42167820320236 for PLL-PEG in the streptavidin session).

The metrics built on these gates reproduce as well:

- `results/main/method1_method2_metrics.csv` is identical to the released data
  file across all 71 rows and all 12 shared columns.
- All 18 cells of manuscript Table 3 reproduce.
- Figures 3 and 5 regenerate byte-identical to the published PNGs.
- The Deep Sets retrain reproduces all five of its committed output files.

Reproducing the thresholds required matching each session's sweep configuration,
as described above. A single configuration applied to both sessions reproduces
one session and misses the other by 10 to 40 units, which is why
`SWEEP_CONFIG` records them separately.

### Sensitivity of the headline result

The one result that carries weight is silica with a 6 nt handle under
streptavidin. It is not sensitive to the threshold:

| Silica 6 nt, streptavidin | V525-H >= 66.7 (published) | V525-H >= 103.5 (1.55x higher) |
|---|---|---|
| Gated monomers | 15,448 | 4,510 |
| Method 1 | +0.456 ± 0.206 | +1.588 ± 0.836 |
| Method 2 | 14.27 ± 7.09 % | 10.42 ± 5.03 % |

Raising the threshold by more than half discards 71 percent of the events and
quadruples the Method 1 standard error, yet Method 2 still shows a large excess
over the 1 percent null and Method 1 stays positive and substantial. The
conclusion does not depend on the threshold.
