# The metrics

Four quantities are reported. All are computed on the gated monomer population
(see [`GATING.md`](GATING.md)) and all are referenced to a control measured in
the **same coating and the same replicate**.

Implementation: [`nfcm/metrics.py`](../nfcm/metrics.py).
Computed by: [`scripts/01_compute_metrics.py`](../scripts/01_compute_metrics.py).

---

## Why within-condition referencing is the whole design

Coatings differ enormously in how much probe sticks to them with no biotin
present. On uncoated origami, streptavidin binds 7.2-fold above background even
with no handle at all. On silica it binds 1.03-fold, which is to say barely at
all.

If accessibility were measured against a shared baseline, that difference would
dominate everything and the comparison would report non-specific adsorption
rather than handle accessibility. Referencing each sample to a no-biotin sample
carrying the *same coating*, incubated with the *same probe*, in the *same
replicate*, cancels the coating-specific background. What is left is attributable
to the biotin handle.

This is also why a coating that suppresses non-specific binding is not
automatically a coating that blocks access. The two are measured separately and
can move in opposite directions.

---

## 1. Total probe association

**What it is.** How much probe lands on origami that has no biotin handle.

```
log2( median B525-H of no-biotin WITH probe
    / median B525-H of no-biotin WITHOUT probe )
```

Both samples are no-biotin monomers of the same coating and replicate; the only
difference is whether probe was added. This is the non-specific floor, **not** an
accessibility measure.

| Coating | Streptavidin | Antibody |
|---|---|---|
| Uncoated | +2.84 ± 0.18 (7.16x) | +0.98 ± 0.06 (1.98x) |
| PLL-PEG | +3.05 ± 0.02 (8.31x) | +0.70 ± 0.09 (1.63x) |
| Silica | +0.04 ± 0.00 (1.03x) | +0.67 ± 0.04 (1.59x) |

Mean ± SEM over three replicates. The probe-dependence is the point: silica
takes streptavidin from 7.2-fold down to 1.03-fold, essentially abolishing it,
while taking the antibody only from 2.0-fold to 1.6-fold.

One-way ANOVA across the three coatings gives F(2,6) = 265.6, p = 1.4e-6 for
streptavidin and F(2,6) = 6.49, p = 0.032 for the antibody.

---

## 2. Method 1, median shift

**What it is.** Accessibility as a shift in the median probe signal when a
biotin handle is added.

```
log2( median B525-H of biotin-extension monomers
    / median B525-H of no-biotin monomers )
```

Both samples carry probe, the same coating, and the same replicate. Zero means
the handle added nothing detectable.

**Sensitive to** a change that moves the whole population.
**Blind to** a small, very bright subpopulation, which a median ignores.

---

## 3. Method 2, positive fraction

**What it is.** Accessibility as the fraction of monomers crossing a high
threshold set on the no-biotin control.

```
threshold = 99th percentile of B525-H of no-biotin monomers
            (same coating, same replicate, with probe)

Method 2  = 100 x fraction of biotin-extension monomers above that threshold
```

Because the threshold is a 99th percentile, **1 percent is the null value by
construction**. A result near 1 percent means no detectable biotin-specific
binding, and only a clear excess counts.

**Sensitive to** a small bright subpopulation.
**Blind to** a modest shift of the whole population that does not cross the
threshold.

Methods 1 and 2 have opposite failure modes, which is why the paper reports both
rather than choosing. They agree on the one strong result: silica with a 6 nt
handle and streptavidin gives Method 1 = +0.456 log2 and Method 2 = 14.27
percent, against a 1 percent null. That condition is driven by two of three
replicates (16.7, 0.96, 25.2 percent), and the replicate spread is reported
honestly as ± 7.09.

---

## 4. Non-specific binding reduction

**What it is.** How much a coating changes non-specific binding relative to bare
origami.

```
log2( median B525-H of coated no-biotin monomers
    / median B525-H of uncoated no-biotin monomers )
```

Both with probe, same replicate. Negative means the coating suppressed
non-specific binding.

---

## Reading the output files

`results/main/method1_method2_metrics.csv`, one row per condition and replicate:

| Column | Meaning |
|---|---|
| `session` | `strep` or `ab` |
| `coating`, `extension`, `replicate` | The condition |
| `n_events` | Gated monomers contributing to the row |
| `B525_median` | Median probe signal of this sample |
| `N_noprobe_median` | Median of the matched no-probe control |
| `N_probe_median` | Median of the matched no-biotin with-probe control |
| `N_probe_P99` | The Method 2 threshold |
| `log2fc_total` | Total probe association |
| `M1_biotin_MFI_shift` | Method 1 |
| `M2_biotin_specific_fraction` | Method 2, percent |

`results/main/summary_by_condition.csv` collapses these to mean ± SEM across
replicates, which is the form the manuscript's Table 3 reports.

Column names match the published data files, so results are directly comparable
against the released tables.

---

## What the metrics do not establish

**Side-scatter shifts on probe binding are not usable.** Candidate probe-induced
shifts of 6 to 56 units sit inside the 10 to 193 unit replicate-to-replicate
range within a single coating, and silica flips sign between the two sessions.
These were examined and rejected as noise; Figure S5-1 documents this.

**Uncoated and PLL-PEG streptavidin cannot show accessibility.** Both are
bimodal at every extension including no-biotin. Non-specific adsorption
saturates the high-signal mode, so a biotin handle has no headroom to add
anything on top. Their Method 1 and Method 2 values near zero mean the
measurement is saturated, not that the handle is inaccessible.

**Longer is not better.** The 12 nt extension performs at or below the shorter
extensions in all three coatings for the antibody and in one of three for
streptavidin. Figure S5-4 covers this.
