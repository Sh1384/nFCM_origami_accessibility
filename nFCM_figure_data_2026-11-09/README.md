# Figure data for the 2026-11-09 draft

These folders hold the figure data for `nFCM_2026-11-09_main_text_JH.docx` (Figures 1–5) and `nFCM_2026-11-09_SI_JH.docx`
(Figures S1–S17). Table S1 and Note S1 have no figure panels.

```
Figure_<n>/
  README.txt                 caption from the .docx and the list of panels
  Panel_<letter>/
    Figure_<n>_<letter>.svg  the panel
    Figure_<n>_<letter>.csv  the plotted values (more tables as Figure_<n>_<letter>_<name>.csv)
    README.txt               what is plotted, how it was computed, source script, notes
panel_index.csv              every panel, with its files, source and notes
code/                        the scripts that compute the data (see code/README.md)
```

There are two kinds of panel:

- **Data panels (34).** The data are recomputed from the FCS files, or taken from the tables of the analysis scripts, and
  the panel is redrawn as SVG. The CSV holds the plotted values as they are, with no fitting or rescaling to match the
  manuscript artwork. The SVGs show the same data as the manuscript panels but are not styled to match them exactly.
- **Image panels (21).** These are schematics, micrographs, gels, the scadnano design and the spectra. They are cut from the
  figure images in the .docx and saved as SVG with the image embedded. They have no CSV: 1A–D, 2A, 3A, 3B, 3F, 4A, 5A, S1,
  S2, S3A–C, S5, S6, S7A–B and S8A–B.

Conventions:

- Batch 1 is the streptavidin session (05/25) and batch 2 is the antibody session (04/18).
- Replicates A to C are tubes.
- Origami events are scatter-triggered (VSSC-1-H ≥ 200).
- The antibody channel is B525-H ≥ 250.
- Acquisitions above 7,000 events/s are excluded. There are four, all silica with probe.

## Choices and differences found while rebuilding the panels

- **2C.** The histogram pools the three batch 2 bare-origami tubes in bins of 20 from the trigger. Plotting the mean of the
  three tube histograms instead differs by less than 1 percentage point per bin. The x^-3.7 guide in the inset, and the
  exponent in 3F and 3H, are the values of the 11/09 text. On the raw events, the slope at the threshold is 4.4 in batch 2
  and 5.6 in batch 1 (`nFCM Manuscript/figures_svg/checks_11_08/fig41d`).
- **2E.** Bins of 10 are centred on 0 to 200, as % of the events in that range, which reproduces the manuscript curves.
- **3C, 3D.** The dashed curves use the caption coefficients (3.49 and 0.561). A least-squares fit of (1 + a·x)² to the same
  ADDA points gives 3.45 and 0.58, so the caption coefficients were not obtained this way. Both curves are in `_curves.csv`.
- **3G.** The rates are buffer-subtracted (33.75 and 59.37 events/s). They match the Prism table you pasted.
- **4B.** The caption does not say which tube is plotted. I used batch 2, replicate A (`N_Silica_NoAB_A`), the only silica
  tube with the saturated cluster at the top right of the panel. Consider adding "(batch 2)" to the caption.
- **4C.** Batch 2, replicate C (`N_PLL_NoAB_C`), the tube in the earlier 4C density table. The guides of slope 1 and ½ pass
  through the median V525-H at VSSC-1-H 800–1,250.
- **5C.** I used bins of 20 from the trigger; the first bar is 25,808 events. The manuscript bars are narrower (first bar
  about 20,000) and their bin width is not recorded.
- **5F.** The per-origami ΔU is 0.745 ± 0.342 kT. The text rounds it to 0.75.
- **3B, S3B, S5.** The wall-profile table, the box counts and the fpbase spectra are not in these files. If you add them,
  these panels can get a CSV too.
