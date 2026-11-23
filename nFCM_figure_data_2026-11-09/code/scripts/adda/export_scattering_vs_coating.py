"""Plotting tables: change in scattering of the open-top Pandora-box origami vs silica thickness and vs PLL-g-PEG coverage.

One row per point. I_ratio = scattering of the coated origami / bare origami (405 nm, water, orientation averaged);
pct_change = 100 x (I_ratio - 1).
  ADDA  discrete-dipole simulation (cross-section, collect_adda.py; n_silica = 1.45)
  RGD   Rayleigh-Gans-Debye model at 90 deg side scatter (model_coating_scattering.py; n_silica = 1.45, 1.40, 1.47)
Silica: conformal layer on all exposed surfaces (outer faces, rim, inner walls, cavity floor) or on the exterior only.
PLL-g-PEG: PLL(200)-g[5]-PEG(5k) (this study), fraction f of the 16,128 DNA phosphates paired with a lysine; lysines in the
DNA wall, PEG chains as a 5.9 nm brush on all exposed surfaces. Composition columns give what f means per origami.
Run after collect_adda.py:  python export_scattering_vs_coating.py
Out:  figures_svg/scattering_vs_silica_thickness.csv, figures_svg/scattering_vs_pllpeg_coverage.csv
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adda_common as A

GEOM = {'all': 'all exposed surfaces', 'exterior': 'exterior only'}


def main():
    C = pd.read_csv(os.path.join(A.FIG_OUT, 'fig32_adda_scattering_cases.csv'))
    C = C[C.dipole_nm == A.DIPOLE_NM]
    S = pd.read_csv(os.path.join(A.FIG_OUT, 'model_coating_scattering_silica.csv'))
    P = pd.read_csv(os.path.join(A.FIG_OUT, 'model_coating_scattering_pllpeg.csv'))

    rows = []
    for coat, geom in GEOM.items():
        ad = C[(C.kind == 'silica') & (C.coat == coat) & np.isclose(C.n_silica, 1.45)].set_index('t_nm').I_over_I0_adda
        for t in [0.0] + A.SILICA_T:
            r = dict(coating_geometry=geom, thickness_nm=t)
            z = S[(S.geometry == geom) & (S.n_case == 'main')]
            r['silica_volume_nm3'] = float(np.interp(t, z.thickness_nm, z.silica_volume_nm3))
            r['silica_mass_MDa'] = float(np.interp(t, z.thickness_nm, z.silica_mass_MDa))
            r['I_ratio_ADDA_n1.45'] = 1.0 if t == 0 else float(ad.get(t, np.nan))
            for case in ['main', 'low', 'high']:
                z = S[(S.geometry == geom) & (S.n_case == case)]
                r[f'I_ratio_RGD_n{z.n_silica.iloc[0]:.2f}'] = float(np.interp(t, z.thickness_nm, z.intensity_ratio))
            rows.append(r)
    Si = pd.DataFrame(rows)
    for c in [c for c in Si.columns if c.startswith('I_ratio_')]:
        Si[c.replace('I_ratio_', 'pct_change_')] = 100 * (Si[c] - 1)
    Si['ADDA_vs_RGD'] = Si['I_ratio_ADDA_n1.45'] / Si['I_ratio_RGD_n1.45']

    ad = C[C.kind == 'pllpeg'].set_index('f_bound').I_over_I0_adda
    z = P[P.architecture.str.contains('this study')]
    rows = []
    for f in [0.0] + sorted(ad.index):
        lys = f * A.NUCLEOTIDES * A.PLL_G / (A.PLL_G - 1)
        rows.append(dict(fraction_phosphates_bound=f, phosphates_bound=f * A.NUCLEOTIDES, lysines=lys,
                         PLL200_chains=lys / 200.0, PEG_chains=f * A.NUCLEOTIDES / (A.PLL_G - 1),
                         polymer_mass_MDa=float(np.interp(f, z.fraction_phosphates_bound, z.polymer_mass_MDa)),
                         polymer_to_DNA_mass=float(np.interp(f, z.fraction_phosphates_bound, z.polymer_to_DNA_mass)),
                         I_ratio_ADDA=1.0 if f == 0 else float(ad[f]),
                         I_ratio_RGD=float(np.interp(f, z.fraction_phosphates_bound, z.intensity_ratio))))
    Pp = pd.DataFrame(rows)
    Pp['pct_change_ADDA'] = 100 * (Pp.I_ratio_ADDA - 1); Pp['pct_change_RGD'] = 100 * (Pp.I_ratio_RGD - 1)
    Pp['ADDA_vs_RGD'] = Pp.I_ratio_ADDA / Pp.I_ratio_RGD

    for df, name in [(Si, 'scattering_vs_silica_thickness.csv'), (Pp, 'scattering_vs_pllpeg_coverage.csv')]:
        df.to_csv(os.path.join(A.FIG_OUT, name), index=False)
        print('wrote', os.path.join(A.FIG_OUT, name))
    pd.set_option('display.width', 250)
    print(Si.round(3).to_string(index=False)); print(); print(Pp.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
