"""Step 3: collect ADDA cross-sections, compute coated/bare ratios and compare with the RGD model.

For each case: Csca = mean of Cext (= Csca, no absorption) over incidence along x, y, z and both polarisations
(random-orientation average in the small-particle limit), I/I0 = Csca(case) / Csca(bare at the same dipole size).
Compared with the Rayleigh-Gans-Debye model (model_coating_scattering.py CSVs, same open-top box geometry; for dimers the
exact orientation-averaged Debye-formula RGD cross-section of the same voxel model, adda_common.debye_ratio).
Outputs (figures_svg/): fig32_adda_scattering.svg, _cases.csv (per case), _runs.csv (per run)
Run:  python collect_adda.py
"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adda_common as A

GEOM = {'all': 'all exposed surfaces', 'exterior': 'exterior only'}
CASES = {c['name']: c for c in A.CASES + A.CONVERGENCE}


def read_cext(path):
    for line in open(path):
        if line.startswith('Cext'):
            return float(line.split('=')[1])
    return np.nan


def main():
    man = pd.read_csv(os.path.join(A.WORK, 'manifest.csv'))
    runs = []
    for _, row in man.iterrows():
        for rdir in sorted(glob.glob(os.path.join(A.WORK, 'runs', f"{row['name']}_prop*"))):
            if not os.path.isdir(rdir):
                continue
            vals = [read_cext(p) for p in [os.path.join(rdir, 'CrossSec-X'), os.path.join(rdir, 'CrossSec-Y')] if os.path.exists(p)]
            if vals:
                runs.append(dict(name=row['name'], run=os.path.basename(rdir), n_polarisations=len(vals), Cext_um2_mean=np.mean(vals)))
    R = pd.DataFrame(runs)
    if R.empty:
        sys.exit('no finished ADDA runs found')
    C = R.groupby('name').agg(Csca_um2=('Cext_um2_mean', 'mean'), n_runs=('run', 'count')).reset_index()
    C = man.merge(C, on='name', how='inner')
    C['Csca_nm2'] = C.Csca_um2 * 1e6
    bare = C.set_index('name').Csca_um2
    C['I_over_I0_adda'] = [r.Csca_um2 / bare.get('bare_d0.25' if r.dipole_nm == 0.25 else 'bare', np.nan) for r in C.itertuples()]
    si = os.path.join(A.FIG_OUT, 'model_coating_scattering_silica.csv'); pp = os.path.join(A.FIG_OUT, 'model_coating_scattering_pllpeg.csv')
    S = pd.read_csv(si) if os.path.exists(si) else None; P = pd.read_csv(pp) if os.path.exists(pp) else None
    rgd = []
    for r in C.itertuples():
        if r.kind == 'silica' and S is not None:
            z = S[(S.geometry == GEOM[r.coat]) & np.isclose(S.n_silica, r.n_silica)]
            rgd.append(float(np.interp(r.t_nm, z.thickness_nm, z.intensity_ratio)) if len(z) else np.nan)
        elif r.kind == 'pllpeg' and P is not None:
            z = P[P.architecture.str.contains('this study')]
            rgd.append(float(np.interp(r.f_bound, z.fraction_phosphates_bound, z.intensity_ratio)))
        elif r.kind == 'dimer':
            rgd.append(A.debye_ratio(CASES[r.name])[1])
        else:
            rgd.append(1.0)
    C['I_over_I0_RGD_model'] = rgd
    C['adda_vs_RGD'] = C.I_over_I0_adda / C.I_over_I0_RGD_model

    try:
        import make_manuscript_figures as M
        save = lambda fig, name, tabs: M.save(fig, name, tabs)
    except Exception:
        def save(fig, name, tabs):
            fig.savefig(os.path.join(A.FIG_OUT, name + '.svg'), bbox_inches='tight')
            for k, v in tabs.items():
                v.to_csv(os.path.join(A.FIG_OUT, name + k + '.csv'), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    ax = axes[0]
    for coat, colr, mk in [('all', '#E6862E', 'o'), ('exterior', '#8C510A', 's')]:
        z = C[(C.kind == 'silica') & (C.coat == coat) & np.isclose(C.n_silica, 1.45) & (C.dipole_nm == A.DIPOLE_NM)]
        ax.scatter(z.t_nm, z.I_over_I0_adda, s=90, color=colr, marker=mk, edgecolor='black', zorder=3, label=f'ADDA, {GEOM[coat]}')
        if S is not None:
            zz = S[(S.geometry == GEOM[coat]) & (S.n_case == 'main') & (S.thickness_nm <= 3.2)]
            ax.plot(zz.thickness_nm, zz.intensity_ratio, color=colr, lw=1.8, label=f'RGD model, {GEOM[coat]}')
    ax.set_xlabel('Silica thickness (nm), n = 1.45'); ax.set_ylabel('Scattering cross-section / bare origami')
    ax.set_title('A  Silica', fontsize=14); ax.legend(fontsize=10, frameon=False)
    ax = axes[1]
    z = C[(C.kind == 'pllpeg')]
    ax.scatter(z.f_bound, z.I_over_I0_adda, s=90, color='#2CA02C', edgecolor='black', zorder=3, label='ADDA')
    if P is not None:
        zz = P[P.architecture.str.contains('this study') & (P.fraction_phosphates_bound <= 1.0)]
        ax.plot(zz.fraction_phosphates_bound, zz.intensity_ratio, color='#2CA02C', lw=1.8, label='RGD model')
    ax.set_xlabel('Fraction of phosphates bound, PLL(200)-g[5]-PEG(5k)'); ax.set_ylabel('Scattering cross-section / bare origami')
    ax.set_title('B  PLL-g-PEG', fontsize=14); ax.legend(fontsize=10, frameon=False)
    ax = axes[2]
    z = C[C.name != 'bare'].sort_values(['kind', 'coat', 'dipole_nm', 'n_silica', 't_nm', 'f_bound'], ascending=[False, True, False, False, True, True]).reset_index(drop=True)
    cols = ['#E6862E' if k == 'silica' else '#2CA02C' if k == 'pllpeg' else '#7F7F7F' for k in z.kind]
    ax.barh(range(len(z)), z.adda_vs_RGD - 1, left=1, color=cols, edgecolor='black')
    ax.axvline(1, color='black', lw=1)
    ax.set_yticks(range(len(z))); ax.set_yticklabels(z.name, fontsize=10); ax.invert_yaxis()
    ax.set_xlim(min(0.92, z.adda_vs_RGD.min() - 0.01), max(1.03, z.adda_vs_RGD.max() + 0.01))
    ax.set_xlabel('ADDA / RGD model (dimers: exact RGD orientation average)')
    ax.set_title('C  Agreement per case (0.25 nm voxels: convergence check)', fontsize=14)
    fig.suptitle('ADDA (discrete dipole) scattering of bare and coated open-top Pandora-box origami (405 nm in water), '
                 'orientation-averaged, relative to bare', y=1.04)
    save(fig, 'fig32_adda_scattering', {'_cases': C, '_runs': R})
    pd.set_option('display.width', 220)
    print(C[['name', 'dipole_nm', 'n_dipoles', 'n_runs', 'Csca_nm2', 'I_over_I0_adda', 'I_over_I0_RGD_model', 'adda_vs_RGD',
             'voxel_vs_exact']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
