# The dataset

Raw flow cytometry files are archived separately from this repository:

**https://drive.google.com/drive/folders/1cDPtf8gMmqNC7LCTPfy_V9OBs1j3bNnz?usp=sharing**

Set `NFCM_DATA_ROOT` to the downloaded location, or place the acquisition
folders in `data/` at the repository root.

---

## What was measured

The particle is a DNA origami "Pandora box", roughly 30.6 x 30.6 x 32.3 nm,
folded from a p8064 scaffold. Two features make it measurable here:

- **Eight ATTO 425 dyes on the interior.** These report that a detected particle
  really is a folded origami, independent of anything bound to its outside. They
  read out on the violet 525 nm detector.
- **One biotin handle on the exterior.** This is the binding target. It is
  presented on single-stranded extensions of 0, 3, 6, or 12 nucleotides, so the
  handle is held at four different distances from the origami surface.

Each origami is measured bare, or wrapped in one of two coatings:

| Condition | Key | Preparation |
|---|---|---|
| Uncoated | `NoCoating` | Origami in buffer |
| PLL-PEG | `PLL` | Poly-L-lysine-polyethylene-glycol, N:P ratio 1.5:1 |
| Silica | `Silica` | Nucleotide:TMAPS:TEOS ratio 1:3:20 |

Two probes are used, both labelled with AF488 and so both read out on the blue
525 nm detector:

| Probe | Mass | Affinity | Session key |
|---|---|---|---|
| Streptavidin-AF488 | 53 kDa tetramer | Kd about 1e-14 M | `streptavidin` |
| Anti-biotin IgG-AF488 | 150 kDa | Kd about 1e-8 M | `antibody` |

The full design is 3 coatings x 4 extensions x 3 biological replicates, acquired
twice, once per probe, plus a matched no-probe control for every condition.

---

## Detector channels

The instrument records nine side-scatter channels per event, three detectors by
three pulse statistics, plus the fluorescence channels.

| Channel | Role |
|---|---|
| `VSSC-1-H` | Primary size proxy. The monomer side-scatter window is defined on this axis. |
| `VSSC-1-A`, `VSSC-1-W` | Violet detector 1, area and width |
| `VSSC-2-H/A/W` | Violet detector 2 |
| `BSSC-H/A/W` | Blue detector. `BSSC-A` is the single most informative axis for the coating classifier. |
| `V525-H` | ATTO 425, interior dyes. Origami identity. |
| `B525-H` | AF488, probe. Binding signal. |
| `Time` | Seconds since acquisition start. Used for event-rate statistics. |

Both probes share `B525-H`, which is why the two probe sessions are separate
acquisitions rather than a single multiplexed run.

---

## Folder layout and filenames

Keys below are the ones used in `nfcm.paths.DATASETS`. The directory names are
the original instrument folder names and are kept verbatim so a downloaded copy
needs no renaming.

| Key | Folder | Contents |
|---|---|---|
| `streptavidin` | `Shervin5_25/` | Main streptavidin session, 61 files |
| `antibody` | `Shervin4_18/Shervin4_18/` | Main antibody session, 61 files. Note this folder is nested one level deeper than the others. |
| `coating_only` | `Shervin2_13_2026/` | Coating chemistry with no origami, the matched debris references |
| `cross_session` | `Shervin5_03/` | Independent session used only to test cross-session transfer |

Acquisitions in the two main sessions follow one pattern:

```
{extension}_{coating}_{probe_tag}_{replicate}.fcs
```

For example `6_Silica_S_B.fcs` is the 6 nt extension on silica, incubated with
streptavidin, replicate B. The probe tag differs by session:

| Session | With probe | No probe |
|---|---|---|
| Streptavidin | `S` | `NoS` |
| Antibody | `AB` | `NoAB` |

Build these paths with `nfcm.io.sample_path` rather than by hand, so the tag
difference stays in one place.

---

## Controls and what each one is for

| Control | File | Used for |
|---|---|---|
| Buffer blank | `Buffer.fcs` in each main session | Debris reference for uncoated origami; spectral crosstalk baseline |
| PLL-PEG only | `PLL_PEG_NoOrigami.fcs` | Debris reference for the PLL-PEG V525-H threshold |
| Silica only | `SilicaChemicalsOnly.fcs` | Debris reference for the silica V525-H threshold |
| No-probe, per condition | `N_{coating}_{NoS or NoAB}_{rep}.fcs` | Derives the side-scatter window; baseline for total probe association |
| No-biotin with probe | `N_{coating}_{S or AB}_{rep}.fcs` | The reference both accessibility metrics are computed against |

The no-biotin with-probe control is the important one. It carries the same
coating and the same probe as the test sample and differs only in having no
biotin handle, so referencing to it removes the coating's non-specific
background rather than merely acknowledging it.

---

## Practical notes

**One condition yields no gated events.** PLL-PEG, 12 nt, antibody, replicate C
produces zero monomers inside its gate, so that condition is reported with
n = 2 in the manuscript. The metrics script drops it automatically; nothing
special needs doing.

**Gated populations are small.** The sparsest retained condition has 84 gated
monomers. This is expected: the gate is deliberately strict, and the fraction of
recorded events that are intact origami is low because coating chemistry
contributes its own particulate.

**Event counts do not equal origami counts.** Recorded events include
coating-derived debris, and most origami in the sampled volume never cross the
trigger threshold. Detected concentration is therefore orders of magnitude below
nominal sample concentration, and the gap tracks coating chemistry rather than
origami content. Event counts should not be read as origami concentrations.

**Vendor metadata warning.** The instrument writes a block into the FCS ANALYSIS
segment that `fcsparser` cannot interpret. It holds acquisition metadata only,
never event data. `nfcm.io` suppresses the resulting warning.
