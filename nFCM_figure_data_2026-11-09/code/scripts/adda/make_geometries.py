"""Step 1: build ADDA geometry files for every case in adda_common.CASES (+ convergence cases).

Writes adda_runs/geometry/<case>.geom and adda_runs/manifest.csv (one row per case: grid size, ADDA -size in um,
relative refractive index of each material, number of dipoles, and a check of the total excess optical volume of the
voxel model against the exact geometry).
Run:  python make_geometries.py            (all cases)
      python make_geometries.py bare dimer_side
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adda_common as A


def main(names=None):
    gdir = os.path.join(A.WORK, 'geometry'); os.makedirs(gdir, exist_ok=True)
    rows = []
    for case in A.CASES + A.CONVERGENCE:
        if names and case['name'] not in names:
            continue
        coords, vals, nx, dn = A.build_geometry(case)
        dom, levels = A.quantize(vals)
        path = os.path.join(gdir, case['name'] + '.geom')
        A.write_geometry(path, coords, dom, len(levels), case['name'])
        m = np.sqrt(1 + levels / A.N_MEDIUM ** 2)
        vox = float(levels[dom].sum() * dn ** 3); ana = A.analytic_optical_volume(case)
        rows.append(dict(name=case['name'], kind=case['kind'], coat=case.get('coat', ''), t_nm=case.get('t', 0.0),
                         n_silica=case.get('n_silica', np.nan), f_bound=case.get('f', np.nan), dipole_nm=dn, nx=nx,
                         size_um=nx * dn / 1000.0, n_dipoles=len(coords), n_materials=len(levels),
                         m_values=' '.join(f'{x:.8f}' for x in m), geometry=os.path.relpath(path, A.WORK),
                         voxel_excess_optical_volume=vox, exact_excess_optical_volume=ana, voxel_vs_exact=vox / ana))
        print(f"{case['name']:24s} d={dn} nm  grid x {nx}  dipoles {len(coords):8d}  materials {len(levels):2d}  "
              f"voxel/exact optical volume {vox / ana:.4f}", flush=True)
    man = os.path.join(A.WORK, 'manifest.csv')
    new = pd.DataFrame(rows)
    if names and os.path.exists(man):
        old = pd.read_csv(man)
        valid = {c['name'] for c in A.CASES + A.CONVERGENCE}
        new = pd.concat([old[~old.name.isin(new.name) & old.name.isin(valid)], new], ignore_index=True)
    new.to_csv(man, index=False)
    print('wrote', man)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
