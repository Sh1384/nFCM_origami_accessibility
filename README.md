# Probe accessibility on coated DNA origami, measured by nano-flow cytometry

Analysis code for the manuscript **"Systematic Characterization of Structure and
Function of DNA Nanostructures under Various Coatings by Nano-Flow Cytometer."**

Everything here runs on single-particle nano-flow cytometry (nFCM) records of a
DNA origami "Pandora box" carrying eight interior ATTO 425 dyes and one exterior
biotin handle. The question the code answers is narrow and practical: **once you
wrap an origami in a protective coating, can a binding partner still reach the
handle on its surface?**

The repository reproduces every number, table, and figure in the paper from the
raw instrument files, and each analysis script states which published value it
should print.

---

## What the analysis measures

Three surface conditions are compared: **uncoated**, **PLL-PEG**, and **silica**.
Each is presented to two probes that differ in size and affinity, streptavidin
(53 kDa) and an anti-biotin IgG (150 kDa), across biotin handle extensions of
0, 3, 6, and 12 nucleotides, in three biological replicates.

Two obstacles shape the whole design, and most of the code exists to handle them.

**Coatings move the measurement window.** Every coating changes how much light a
particle scatters and how bright its interior dyes read. A gate drawn on
uncoated origami therefore under-counts coated particles. So the monomer gate is
derived *per coating*, from that coating's own controls, and never transferred.
See [`docs/GATING.md`](docs/GATING.md).

**Probe sticks to origami whether or not a handle is there.** Non-specific
adsorption is large enough to swamp the specific signal, so every accessibility
metric is referenced to a no-biotin sample of the *same coating and the same
replicate*. See [`docs/METRICS.md`](docs/METRICS.md).

### Headline results the code reproduces

| Finding | Where |
|---|---|
| Silica nearly abolishes non-specific streptavidin binding, from 7.2-fold on uncoated origami to 1.03-fold, while cutting antibody binding only from 2.0-fold to 1.6-fold | `scripts/05_figure5_accessibility.py` |
| Silica with a 6 nt handle is the one condition showing clear specific capture: Method 1 = +0.456 log2, Method 2 = 14.27 percent of monomers | `scripts/01_compute_metrics.py` |
| Coating identity is readable from label-free side scatter alone, macro AUC 0.759, with the blue-detector axes carrying 56 percent of the decision | `scripts/04_figure4_classifier.py` |
| The axes the classifier relies on are also the ones that drift most between sessions, so the model does not transfer | `scripts/08_si_figures.py` (Figure S6-1) |

---

## Raw data

The FCS archive is **not** in this repository. Download it here:

**https://drive.google.com/drive/folders/1cDPtf8gMmqNC7LCTPfy_V9OBs1j3bNnz?usp=sharing**

Point the code at your copy with an environment variable:

```bash
export NFCM_DATA_ROOT=/path/to/downloaded/archive
```

or place the acquisition folders in `data/` at the repository root, which is the
default and is git-ignored. [`docs/DATA.md`](docs/DATA.md) describes the folder
layout, the filename convention, and which acquisitions each analysis uses.

Scripts that only read cached results run without the archive. Scripts that read
FCS files fail immediately with a message telling you to set the variable, rather
than part-way through.

---

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/Sh1384/nFCM_origami_accessability.git
cd nFCM_origami_accessability

python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`pip install -e .` is optional. Every script inserts the repository root on
`sys.path`, so a fresh clone runs as-is once the dependencies are present.

Results in the paper were produced with Python 3.10.10, numpy 1.26.4,
pandas 2.2.1, scikit-learn 1.7.2, xgboost 3.0.0, umap-learn 0.5.11, and
torch 2.2.1.

---

## Reproducing the paper

Run in order. Steps 00 to 02 build the inputs everything else reads.

```bash
python scripts/00_derive_gates.py         # re-derive and audit the monomer gate
python scripts/01_compute_metrics.py      # Method 1, Method 2, non-specific binding
python scripts/02_build_features.py       # per-event feature matrices (cached)

python scripts/03_figure3_gating.py       # Figure 3
python scripts/04_figure4_classifier.py   # Figure 4
python scripts/05_figure5_accessibility.py # Figure 5 and standalone panel A

python scripts/06_si_statistics.py        # Supporting Information tables
python scripts/07_si_deepsets.py          # Deep Sets aggregation (slow)
python scripts/08_si_figures.py           # the 11 Supporting Information figures
```

| Script | Reads | Writes | Runtime |
|---|---|---|---|
| `00_derive_gates.py` | FCS | `results/main/gates_rederived.json` | ~2 min |
| `01_compute_metrics.py` | FCS, `config/gates.json` | 3 CSVs in `results/main/` | ~5 min |
| `02_build_features.py` | FCS | 2 Parquet files in `cache/` | ~3 min |
| `03_figure3_gating.py` | FCS, gates | `figures/main/Fig3_*` | ~3 min |
| `04_figure4_classifier.py` | `cache/` | `figures/main/Fig4_*`, AUC and importance CSVs | ~5 min |
| `05_figure5_accessibility.py` | FCS, metrics | `figures/main/Fig5_*` | ~4 min |
| `06_si_statistics.py` | `results/` | SI tables in `results/si/` | <1 min |
| `07_si_deepsets.py` | `cache/` | Deep Sets CSVs in `results/si/` | ~30 min |
| `08_si_figures.py` | `results/`, `cache/`, FCS | 11 figures in `figures/supplementary/` | ~8 min |

Steps 02 and 07 cache their work. Delete the relevant file in `cache/` or pass
`--force` to rebuild.

Figure 2 is a microscopy, gel, and simulation composite assembled outside this
pipeline; the published file is version-controlled in `figures/main/`.

---

## Repository layout

```
nfcm/                    importable package: the definitions every script shares
  constants.py           detector channels, conditions, colours, thresholds
  paths.py               path resolution, including the raw data root
  io.py                  FCS reading and the sample naming convention
  gating.py              the two-component monomer gate and its derivation
  metrics.py             Method 1, Method 2, non-specific binding
  style.py               shared matplotlib style for publication figures

scripts/                 the analyses, numbered in dependency order
config/gates.json        the frozen monomer gate the manuscript was computed with
results/main/            metrics quoted in the main text and its tables
results/si/              tables reported in the Supporting Information
results/intermediate/    per-sample summaries several figures read
figures/main/            Figures 2 to 5, PDF and PNG
figures/supplementary/   the 11 Supporting Information figures, PDF and PNG
docs/                    method documentation, see below
cache/                   rebuildable feature matrices (git-ignored)
data/                    raw FCS archive goes here (git-ignored)
```

### Documentation

| Document | Covers |
|---|---|
| [`docs/DATA.md`](docs/DATA.md) | Acquisition folders, filename convention, controls, what the archive contains |
| [`docs/GATING.md`](docs/GATING.md) | How the monomer gate is derived, why it is per-coating, and its reproduction status |
| [`docs/METRICS.md`](docs/METRICS.md) | Exact definition of every reported metric and the control it references |
| [`docs/FIGURES.md`](docs/FIGURES.md) | Which script produces which panel of which figure |

---

## Scope

This repository covers the three surface conditions reported in the manuscript,
uncoated, PLL-PEG, and silica, measured against both probes across all four
biotin handle lengths and all three biological replicates. Every figure, table,
and statistic in the paper and its Supporting Information is reproduced here.

## License

See [`LICENSE`](LICENSE).
