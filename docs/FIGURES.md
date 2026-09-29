# Figure inventory

Which script builds which figure, and what each panel shows. Every figure is
written as both PDF (the manuscript version, with embedded TrueType fonts so it
stays editable) and PNG (for preview and for embedding in Word, which cannot
place a PDF image).

Shared styling lives in [`nfcm/style.py`](../nfcm/style.py).

---

## Main figures

Files are in `figures/main/`, named by **manuscript figure number**.

### Figure 2 — Pandora box characterisation
`Fig2_Pandora_manuscript.{pdf,png}`

Transmission electron microscopy, gel electrophoresis, oxDNA simulation, and the
size distribution of the folded structure. Assembled outside this pipeline from
microscopy and simulation outputs, so no script here regenerates it; the
published file is version-controlled.

### Figure 3 — Instrumentation and gating
`Fig3_Instrumentation_Gating_manuscript.{pdf,png}`
Built by [`scripts/03_figure3_gating.py`](../scripts/03_figure3_gating.py).

| Row | Shows |
|---|---|
| A | Median side scatter, median ATTO 425, and side-scatter CV per coating, with replicates overlaid |
| B | Spectral separation of the two fluorescence channels, using buffer, ATTO-only, and AF488-only controls |
| C | Log-log density of side scatter against ATTO 425 for each coating, with the derived monomer gate drawn on top |
| D | The V525-H peak-enrichment sweep: signal retention, debris pass-rate, and their ratio against candidate thresholds |

Row C is where the per-coating gate is easiest to see: the three gates sit in
visibly different places because the coatings scatter differently.

### Figure 4 — Label-free coating classifier
`Fig4_ML_manuscript.{pdf,png}`
Built by [`scripts/04_figure4_classifier.py`](../scripts/04_figure4_classifier.py).

| Panel | Shows |
|---|---|
| A | UMAP of the nine-dimensional side-scatter space, ground truth beside classifier prediction |
| B | One-vs-rest ROC curves on the held-out replicate |
| C | XGBoost feature importance across the nine side-scatter axes |
| D | Macro AUC by feature set: one axis, all nine, and nine plus fluorescence |

XGBoost with 200 trees, depth 5, learning rate 0.1. Trained on replicates A and
B, tested on replicate C, with classes balanced by subsampling. Macro AUC is
0.759 from side scatter alone; the blue-detector axes carry 56 percent of the
decision, with `BSSC-A` alone at 0.32.

Panel D is the point of the figure: adding fluorescence does **not** help
(0.718 against 0.759), so coating identity is genuinely readable without any
label.

### Figure 5 — Probe accessibility
`Fig5_Probe_Accessibility_manuscript.{pdf,png}` and `Fig5_PanelA_only.{pdf,png}`
Built by [`scripts/05_figure5_accessibility.py`](../scripts/05_figure5_accessibility.py).

| Panel | Shows |
|---|---|
| A | Total probe association on no-biotin monomers, both probes |
| B | Method 1, log2 median shift from adding a biotin handle |
| C | Method 2, percent of monomers above the no-biotin threshold, with the 1 percent null marked |
| D | Pooled probe-channel distributions per coating, biotin extensions overlaid |

Panel A is also written standalone, at the same aspect ratio as the other
panels, because the manuscript uses it separately.

Panels A and C share a y-axis between probes so the two are directly comparable;
panel B is scaled independently because its range is much narrower.

---

## Supporting Information figures

Files are in `figures/supplementary/`. All but one are built by
[`scripts/08_si_figures.py`](../scripts/08_si_figures.py), which accepts figure
keys as arguments (`python scripts/08_si_figures.py S3-1 S4-2`) and builds
everything when called with none.

| Figure | Key | Shows |
|---|---|---|
| S3-1 | `S3-1` | Acquisition quality control: event rates, durations, gated retention per coating |
| S3-2 | `S3-2` | Origami identity against coating-only debris on the ATTO 425 axis |
| S3-3 | `S3-3` | All nine side-scatter axes compared across coatings |
| S4-1 | `S4-1` | Per-event classifier detail: confusion matrix, pairwise separability, feature correlations |
| S4-2 | `S4-2` | Deep Sets multi-particle aggregation: accuracy against bag size, seed band, channel ablation, learning curve |
| S5-1 | `S5-1` | Alternative metrics considered and rejected, including side-scatter shift and earth-mover distance |
| S5-2 | — | Gate sensitivity: the headline result across threshold scalings. **Carried over, not rebuilt.** See note below. |
| S5-3a/b | `S5-3a`, `S5-3b` | Per-replicate probe-channel distributions, so replicate spread is visible rather than averaged away |
| S5-4 | `S5-4` | Extension length: why 12 nt does not outperform 6 nt |
| S6-1 | `S6-1` | Cross-session channel drift, and why the classifier does not transfer |

### Note on Figure S5-2

This figure shows only uncoated and silica, so removing the fourth surface
condition from the study left it unchanged and it was carried over rather than
rebuilt. Its generator predates the scripts kept in this repository, so the
published PDF and PNG are version-controlled directly. `08_si_figures.py`
confirms the files are present and warns if they are missing, rather than
silently producing an incomplete figure set.

The underlying analysis is reported in [`METRICS.md`](METRICS.md): silica 6 nt
streptavidin Method 2 stays between 12.4 and 15.2 percent across V525-H
threshold scalings from 0.6x to 1.25x.

---

## Regenerating everything

```bash
export NFCM_DATA_ROOT=/path/to/archive

python scripts/02_build_features.py        # if cache/ is empty
python scripts/03_figure3_gating.py
python scripts/04_figure4_classifier.py
python scripts/05_figure5_accessibility.py
python scripts/07_si_deepsets.py           # needed before Figure S4-2
python scripts/08_si_figures.py
```

Figures 3 and 5 and the SI figures read FCS files directly, so the raw archive is
required. Figure 4 and Figure S4-2 read only the cached feature matrices, so
they rerun without it once `cache/` is populated.
