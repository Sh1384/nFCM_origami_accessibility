# Analysis scripts

Numbered in dependency order. Steps 00 to 02 build the inputs everything else
reads; after those, the figure and table scripts are independent of each other
except where noted.

Every script takes its paths from `nfcm.paths`, so none contains an absolute
path. Scripts that read FCS files call `require_data_root()` first and fail
immediately with an instruction if the raw archive is not configured, rather
than part-way through a long loop.

```bash
export NFCM_DATA_ROOT=/path/to/archive
```

## Order

| # | Script | Purpose | Requires | Runtime |
|---|---|---|---|---|
| 00 | `00_derive_gates.py` | Re-derive the monomer gate and audit it against the frozen `config/gates.json` | raw FCS | ~2 min |
| 01 | `01_compute_metrics.py` | Method 1, Method 2, total probe association, non-specific binding | raw FCS | ~5 min |
| 02 | `02_build_features.py` | Per-event feature matrices for the classifiers, cached as Parquet | raw FCS | ~3 min |
| 03 | `03_figure3_gating.py` | Figure 3, instrumentation and gating | raw FCS | ~3 min |
| 04 | `04_figure4_classifier.py` | Figure 4, label-free coating classifier | step 02 cache, or raw FCS to rebuild it | ~5 min |
| 05 | `05_figure5_accessibility.py` | Figure 5 and standalone panel A | raw FCS, step 01 | ~4 min |
| 06 | `06_si_statistics.py` | Supporting Information tables | step 01, step 02 cache | <1 min |
| 07 | `07_si_deepsets.py` | Deep Sets multi-particle aggregation | step 02 cache; raw FCS optional for the cross-session block | ~30 min |
| 08 | `08_si_figures.py` | The eleven Supporting Information figures | raw FCS, steps 01, 02, 06, 07 | ~8 min |

Only steps 00 to 03 and 05 read the raw archive unconditionally. The rest work
from `cache/` and `results/`, so they rerun without it once those exist.

Step 08 builds Figure S4-2 from the output of step 07, so run 07 first for a
complete figure set. Step 08 also accepts figure keys to rebuild a subset:

```bash
python scripts/08_si_figures.py S3-1 S5-4
```

## Caching

Steps 02 and 07 write to `cache/`, which is git-ignored and rebuildable. Step 02
skips work when the cache exists; pass `--force` to rebuild. Delete the relevant
file to force a recompute of anything downstream.

## Self-checking

Scripts that reproduce a published number print it next to the expected value,
so a broken environment shows up immediately rather than silently:

- `00_derive_gates.py` prints how many of the six windows and six thresholds
  match `config/gates.json`. Expect 6 of 6 and 4 of 6; the two known
  differences are explained in [`../docs/GATING.md`](../docs/GATING.md).
- `01_compute_metrics.py` prints the silica 6 nt streptavidin result against the
  published +0.456 and 14.27 percent.
- `02_build_features.py` prints the event count per coating and replicate,
  totalling 273,913.

## A note on the figure scripts

The package under `nfcm/` holds the canonical definitions, and steps 00 to 02
use it throughout. The figure scripts (03 to 05, 07, 08) additionally keep their
own local style block and condition lists rather than importing
`nfcm.style` and `nfcm.constants`.

That is deliberate. Those literals are what produced the published figures, and
Figures 3 and 5 currently regenerate byte-identical to the released PNGs.
Routing them through the shared style would risk silent changes to panel
geometry and colour for no scientific gain. New analysis should import from
`nfcm/`; the figure scripts are kept as the record of what was published.

## What is not here

Figure 2 is a microscopy, gel, and simulation composite assembled outside this
pipeline, and Figure S5-2 predates the scripts kept here. Both published files
are version-controlled under `figures/`. See
[`../docs/FIGURES.md`](../docs/FIGURES.md).
