# Code for the figures of the 2026-11-09 draft

`scripts/` is a copy, made on 2026-11-09, of the scripts in `nFCM Manuscript/scripts` that compute the data of the figures in
`nFCM_2026-11-09_main_text_JH.docx` and `nFCM_2026-11-09_SI_JH.docx`. `build_figure_package.py` writes the figure folders of
this package. The working copies stay in `nFCM Manuscript/scripts`, so edit those and copy them again.

## Requirements

Python 3.11 with numpy 2.2, pandas 2.2, scipy 1.14, matplotlib 3.8, seaborn 0.13, statsmodels 0.14, fcsparser 0.2.8 and
Pillow 10. The simulations also need ADDA 1.4.0 (`tools/adda/adda-1.4.0/win64/adda.exe`).

## Layout

The scripts find their inputs and outputs relative to the folder that contains `scripts/`:

```
<folder>/
  ../Dilution/   FCS files and FileInfo.csv of the dilution series (or set NFCM_DILUTION_DIR)
  Data/          FCS files: Shervin4_18 (batch 2, antibody), Shervin5_25 (batch 1, streptavidin),
                 probe_only_controls, coatings_controls
  scripts/       these scripts
  figures_svg/   tables and plots written by the analysis scripts (created on first run)
  adda_runs/     ADDA geometries and results
```

In `nFCM Manuscript/` this layout already exists. To run this copy, put `Data/` next to `scripts/`. `build_figure_package.py`
writes to `NFCM_PACKAGE_DIR`, and reads the figure images from `NFCM_MAIN_DOCX` and `NFCM_SI_DOCX`. Each of these defaults to
the `ShervinCode_Data` folder.

Acquisitions above 7,000 events/s are excluded (`shared_utils.load_fcs`) everywhere except in the dilution series
(`dilution_*.py`, Figure S5), which keeps all tubes. Throughout the code, the scatter trigger is
VSSC-1-H ≥ 200, and the antibody threshold is B525-H ≥ 250.

## Run order

```bash
cd scripts
# 1  Flow-cytometry tables (median curves, quadrants, medians, rates, maps, mixture model used by fig29c)
python make_manuscript_figures.py fig06 fig08 fig10 fig11 fig13 fig20 fig21 fig22 fig27d
# 2  Two-color coincidence detection, acquisition order and delta U
python fig29_tccd_coincidence.py
python tccd_acquisition_order.py
python fig29b_tccd_summary.py
python fig29c_tccd_percent_bound.py
python fig29d_tccd_ddG.py
python fig35_antibody_length_dependence.py
python fig5g_delta_U_matched_lengths.py
# 3  Panels 5B and the run-order table (S17, S18)
python fig37_proposed_flow_panels.py 5b 5f
# 4  Light-scattering simulations (ADDA; results are already in figures_svg/fig32_adda_scattering_cases.csv)
python adda/make_geometries.py
python adda/run_adda.py
python adda/collect_adda.py
python model_coating_scattering.py
python adda/export_scattering_vs_coating.py
# 5  Serial dilution (Figure S5)
python dilution_01_summarize.py
# to redraw S5 alone: python Figure_S5_A.py in Figure_S5/Panel_A
# 6  Figure folders
python build_figure_package.py
# Figure S4B alone, without rebuilding the other folders
python fig_s4b_local_slope.py
```

## Which script makes each panel

| Panel | Data | Computed by |
|---|---|---|
| 2B, 2C, 2D, 2E | FCS files | `build_figure_package.py` `fig2` |
| 3C, 3D | `scattering_vs_pllpeg_coverage.csv`, `scattering_vs_silica_thickness.csv` | `adda/*`, `model_coating_scattering.py`, `adda/export_scattering_vs_coating.py` |
| 3E, 3G, 3H | FCS files | `build_figure_package.py` `fig3` |
| 4B, 4C | FCS files | `build_figure_package.py` `fig4` |
| 4D, 4E | `silica_side_scatter/fig06_..._median_v525_by_ss.csv` | `make_manuscript_figures.py` `fig06` (buffer curve from `fig21`) |
| 4F, 4G | `pll_peg_side_scatter/fig21_..._median_v525_by_ss.csv` | `make_manuscript_figures.py` `fig21` |
| 5B | `proposed_panels/fig5b_..._samples.csv` | `fig37_proposed_flow_panels.py` `panel_5b` |
| 5C | FCS file (antibody only) | `build_figure_package.py` `fig5` |
| 5D, 5E, S15 | `fig29d_tccd_ddG_samples.csv` | `fig29_tccd_coincidence.py`, then `fig29c`, then `fig29d` |
| 5F | `fig5g_delta_U_matched_lengths.csv` | `fig29*`, `tccd_acquisition_order.py`, `fig35_...`, `fig5g_delta_U_matched_lengths.py` |
| S4A | FCS files | `build_figure_package.py` `si` |
| S4B | FCS files | `fig_s4b_local_slope.py` (also run by `build_figure_package.py`) |
| S5 | `dilution/dilution_files.csv` | `dilution_01_summarize.py`, then `fig_s5_dilution_series.py` (saved in the panel folder as `Figure_S5_A.py`) |
| S10 | `silica_side_scatter/fig22_scatter_fluorescence_quadrants.csv` | `make_manuscript_figures.py` `fig22` |
| S11 | `fig08_median_b525_with_without_antibody.csv` | `make_manuscript_figures.py` `fig08` |
| S12 | `fig13_extension_dependence_antibody.csv` (column M1) | `make_manuscript_figures.py` `fig13` |
| S13 | `fig10_free_antibody_rates.csv` | `make_manuscript_figures.py` `fig10` |
| S14 | `fig29_tccd_coincidence_samples.csv` | `fig29_tccd_coincidence.py` |
| S16 | `fig11_free_streptavidin_maps.csv`, `_rates.csv` | `make_manuscript_figures.py` `fig11` |
| S17, S18 | `proposed_panels/fig5f_run_order_check_acquisitions.csv` | `fig29_tccd_coincidence.py`, `tccd_acquisition_order.py`, `fig37_proposed_flow_panels.py` `panel_5f` |

Panels 1A–D, 2A, 3A, 3B, 3F, 4A, 5A, S1, S2, S3, S6, S7, S8 and S9 are schematics, micrographs, gels or designs. They are cut
from the figure images in the .docx files.

## Numbers in the text and where they come from

| Statement | Script |
|---|---|
| Four acquisitions above 7,000 events/s, all silica with probe; buffer rates 34 and 59 events/s; ρ⁰ 10.9 vs 10.8 and its test; scatter-tail exponent | `fig41_caption_stat_checks.py` |
| Silica vs bare median scatter of origami-confident events, batch 1, 442 vs 229, Welch p = 0.07 | `silica_confident_origami_summary.py` |
| Coincident fraction ∝ antibody rate (log–log slope 1.00 ± 0.02), ρ CV 1.8%, antibody rate 1.7-fold lower on PLL-PEG | `tccd_statistics_check.py` |
| Conclusions unchanged with a common origami-gate upper bound (510 or 810); mixed models | `comment_checks_10_24.py` |
| Run-order trends of ρ and antibody rate within a series | `fig39_run_order_confound.py` |
| log2F and ΔU per replicate set (alternative summary) | `tccd_log2fc_delta_U.py` |

`dilution_common.py`, `dilution_01_summarize.py` and `dilution_14_replicate_series.py` were copied in on 2026-11-21 for
Figure S5. Figures S5–S17 of the 11/09 SI are S6–S18 here.

`fig_s4b_local_slope.py` was added on 2026-11-21. It writes Figure S4B with the raw events behind it.

`tccd_acquisition_order.py` was added on 2026-11-09. It regenerates `figures_svg/tccd_acquisition_order.csv` (from
the start time in each FCS acquisition record), which was previously written outside the scripts. Its output matches the
10/06 table in every column.
