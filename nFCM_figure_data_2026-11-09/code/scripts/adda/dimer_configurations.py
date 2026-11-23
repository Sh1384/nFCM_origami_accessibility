"""Figure 34: scattering of origami dimers in different contact geometries, relative to a single open-top box.

Configurations (adda_common.DIMERS): side to side (wall on wall), edge to edge along a vertical or a horizontal edge,
and three stackings - bottom on rim (same orientation), bottom to bottom, rim to rim (open faces together).
For each: ADDA cross-section (orientation average over incidence along x, y, z and both polarisations), the exact
orientation-averaged Rayleigh-Gans-Debye (RGD) cross-section of the same voxel model (Debye formula), and the RGD
intensity at 90 deg (side scatter), all relative to the bare monomer. The point-particle limit is 4 (twice the
polarisability, coherent); finite size lowers it through the interference between the two boxes, which depends on the
distance between their centres of mass (the bottom shifts each box's centre of mass 0.9 nm towards it).
Run after run_adda.py and collect_adda.py:  python dimer_configurations.py
Out:  figures_svg/fig34_adda_dimer_configurations.svg, _configurations.csv
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adda_common as A
from visualize_geometries import labels_grid

COLS = ['#1F77B4', '#D62728', '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B']


def com_offset():
    """Centre of mass of one box relative to its geometric centre (open face up), from the wall slabs."""
    v = [np.prod(2 * sh) for _, sh in A.slabs(np.zeros(3))]
    z = [sc[2] for sc, _ in A.slabs(np.zeros(3))]
    return np.dot(v, z) / np.sum(v)


def main():
    C = pd.read_csv(os.path.join(A.FIG_OUT, 'fig32_adda_scattering_cases.csv')).set_index('name')
    dz = com_offset()
    rows = []
    for name, (label, bx) in A.DIMERS.items():
        case = dict(name=name, kind='dimer')
        i90, cs, curve = A.debye_ratio(case, curve=True)
        coms = [np.array(c) + np.array([0, 0, u * dz]) for c, u in bx]
        rows.append(dict(name=name, configuration=label, centre_of_mass_separation_nm=float(np.linalg.norm(coms[1] - coms[0])),
                         adda_Csca_ratio=C.loc[name, 'I_over_I0_adda'] if name in C.index else np.nan,
                         rgd_Csca_ratio=cs, rgd_I90_ratio=i90, adda_vs_rgd=(C.loc[name, 'I_over_I0_adda'] / cs) if name in C.index else np.nan,
                         **{f'rgd_ratio_{t:g}deg': v for t, v in zip(A.THETA_GRID, curve)}))
    D = pd.DataFrame(rows)

    n = len(D)
    fig = plt.figure(figsize=(4.2 * n, 12))
    gs = fig.add_gridspec(2, n, height_ratios=[1.1, 1], hspace=0.02)
    for j, r in D.iterrows():
        case = dict(name=r['name'], kind='dimer')
        lab, ax3 = labels_grid(case, 1.0)
        X, Y, Z = np.meshgrid(*ax3, indexing='ij')
        keep = np.ones(lab.shape, bool)
        bx = A.boxes(case)
        same_plane = len({round(c[1], 3) for c, _ in bx}) == 1     # a half-cut would separate boxes touching along a vertical edge
        for (c, u) in (bx if same_plane else []):    # cut each box in half through its own centre (front half removed)
            inside = (np.abs(X - c[0]) <= A.BOX_NM[0] / 2 + 1) & (np.abs(Y - c[1]) <= A.BOX_NM[1] / 2 + 1) &                      (np.abs(Z - c[2]) <= A.BOX_NM[2] / 2 + 1)
            keep &= ~(inside & (Y < c[1]))
        ax = fig.add_subplot(gs[0, j], projection='3d')
        ax.voxels((lab == 1) & keep, facecolors=COLS[j], edgecolor=None)
        ax.view_init(elev=18, azim=-62); ax.set_box_aspect(lab.shape); ax.set_axis_off()
        ax.set_title(f"{'ABCDEF'[j]}  {r.configuration}\nCOM separation {r.centre_of_mass_separation_nm:.1f} nm", fontsize=12)
    gsb = gs[1, :].subgridspec(1, 2, wspace=0.25)
    ax = fig.add_subplot(gsb[0, 0])
    x = np.arange(n)
    ax.axhspan(3, 5, color='0.92', zorder=0, label='dimer factor range used in Fig. 30 (3–5)')
    ax.axhline(4, color='black', ls='--', lw=1.2, label='point-particle limit (4)')
    ax.bar(x, D.adda_Csca_ratio, color=COLS[:n], edgecolor='black', width=0.6, label='ADDA cross-section')
    ax.scatter(x, D.rgd_Csca_ratio, marker='o', s=70, facecolor='white', edgecolor='black', zorder=4, label='RGD cross-section (exact orientation average)')
    ax.scatter(x, D.rgd_I90_ratio, marker='D', s=70, color='black', zorder=4, label='RGD at 90° (side scatter)')
    for xi, v in zip(x, D.rgd_I90_ratio):
        ax.text(xi, 3.02, f'{v:.2f}', ha='center', va='bottom', fontsize=11, color='white', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([c.replace(' (', '\n(').replace(': ', ':\n') for c in D.configuration], fontsize=10)
    ax.set_ylim(3, 5.2); ax.set_ylabel('Dimer scattering / single box')
    ax.set_title('G  Dimer scatter factor by contact geometry (405 nm, water)', fontsize=14)
    ax.legend(fontsize=10, frameon=False, loc='upper left', ncol=2)
    ax = fig.add_subplot(gsb[0, 1])
    for j, r in D.iterrows():
        ax.plot(A.THETA_GRID, [r[f'rgd_ratio_{t:g}deg'] for t in A.THETA_GRID], color=COLS[j], lw=2, label=r.configuration)
    ax.axvline(90, color='black', ls=':', lw=1.2); ax.axhline(4, color='black', ls='--', lw=1.2)
    ax.set_xlabel('Scattering angle (deg)'); ax.set_ylabel('Dimer / single box (RGD, orientation averaged)')
    ax.set_xlim(0, 180); ax.set_title('H  Dimer factor vs scattering angle (dotted: 90°)', fontsize=14)
    ax.legend(fontsize=10, frameon=False, loc='lower left')
    fig.suptitle('Origami dimers in different contact geometries (boxes cut in half to show the cavities; B uncut): scattering '
                 'relative to one open-top box (ADDA discrete dipole and Rayleigh-Gans-Debye)', y=0.99)
    tab = D[[c for c in D.columns if not c.startswith('rgd_ratio_')]]
    try:
        import make_manuscript_figures as M
        M.save(fig, 'fig34_adda_dimer_configurations', {'_configurations': D})
    except Exception:
        fig.savefig(os.path.join(A.FIG_OUT, 'fig34_adda_dimer_configurations.svg'), bbox_inches='tight')
        D.to_csv(os.path.join(A.FIG_OUT, 'fig34_adda_dimer_configurations_configurations.csv'), index=False)
    pd.set_option('display.width', 220)
    print(tab.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
