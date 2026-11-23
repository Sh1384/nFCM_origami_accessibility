"""Visualise the particle models given to ADDA (bare and coated open-top Pandora-box origami).

Outputs
  figures_svg/fig33_adda_geometries.svg   static figure, one column per case:
      row 1  3D cutaway (a vertical quarter removed) at 1 nm display resolution: blue = DNA wall, orange = silica,
             green (translucent) = PEG brush; the cavity (open at the top) is water
      row 2  vertical slice (y = 0, x-z plane) of the exact ADDA input: relative refractive index m = n / n_water of every
             dipole (0.5 nm dipoles, partial-volume voxels, quantised to the ADDA materials) - shows the open top,
             the 5 nm side walls and the 2.5 nm bottom
      row 3  index profiles through the centre along x (side walls) and along z (bottom and open top)
  figures_svg/adda_geometry_viewer.html   interactive 3D viewer (Plotly from cdn.jsdelivr.net): voxel surfaces at 1 nm,
                                          single boxes cut away, dimer configurations (fig34) whole; rotate/zoom, pick case
  adda_runs/vtk/<case>.vtk                 full-resolution input grids (relative index) as legacy VTK for ParaView
Run:  python visualize_geometries.py            (after make_geometries.py)
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adda_common as A

SHOW = ['bare', 'silica_all_1nm', 'silica_ext_1nm', 'pllpeg_f0.1']
TITLES = {'bare': 'Bare origami (open top)', 'silica_all_1nm': 'Silica 1 nm, all exposed surfaces',
          'silica_ext_1nm': 'Silica 1 nm, exterior only', 'pllpeg_f0.1': 'PLL-g-PEG, 10% of phosphates',
          **{k: 'Dimer, ' + v[0] for k, v in A.DIMERS.items()}}
MAT_COL = {1: '#4C8ED9', 2: '#E6862E', 3: '#2CA02C'}
MAT_NAME = {1: 'DNA wall', 2: 'silica', 3: 'PEG brush'}
CASES = {c['name']: c for c in A.CASES}


def labels_grid(case, res=1.0):
    """Material label at voxel centres on a res-nm grid: 0 water, 1 DNA wall, 2 silica, 3 PEG brush."""
    H = np.array(A.BOX_NM) / 2
    t = case.get('t', 0.0) if case['kind'] == 'silica' else 0.0
    h = A.PEG_BRUSH_NM if case['kind'] == 'pllpeg' else 0.0
    pad = max(t, h) + res
    cens = A.box_centers(case)
    lo = np.min([c - H - pad for c in cens], 0); hi = np.max([c + H + pad for c in cens], 0)
    nvox = np.ceil((hi - lo) / res).astype(int)
    ax = [(lo[i] + hi[i]) / 2 + (np.arange(nvox[i]) - (nvox[i] - 1) / 2) * res for i in range(3)]   # centred grid
    P = np.stack(np.meshgrid(*ax, indexing='ij'), -1)
    wall, dist, col = A.membership(P, case)
    lab = np.zeros(wall.shape, int); lab[wall] = 1
    if t > 0:
        sel = ~wall & (dist <= t)
        if case.get('coat') == 'exterior':
            sel &= ~col
        lab[sel] = 2
    if h > 0:
        lab[~wall & (dist <= h)] = 3
    return lab, ax


def adda_input_grid(case):
    """Dense grid of the relative refractive index exactly as written to the ADDA geometry file."""
    coords, vals, nx, dn = A.build_geometry(case)
    dom, levels = A.quantize(vals)
    grid = np.ones(coords.max(0) + 1)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = np.sqrt(1 + levels[dom] / A.N_MEDIUM ** 2)
    axes = [(np.arange(s) - (s - 1) / 2) * dn for s in grid.shape]
    return grid, axes, dn


def voxel_mesh(mask, ax, res):
    """Triangle mesh (vertices (V, 3) in nm, triangles (T, 3)) of the exposed faces of a boolean voxel grid."""
    m = np.pad(mask, 1)
    quads = []
    for d in range(3):
        u, v = [a for a in range(3) if a != d]
        for s in (-1, 1):
            idx = np.argwhere(m & ~np.roll(m, -s, axis=d)).astype(float)
            idx[:, d] += s > 0
            c = np.repeat(idx[:, None, :], 4, 1)
            c[:, 1, u] += 1; c[:, 2, u] += 1; c[:, 2, v] += 1; c[:, 3, v] += 1
            quads.append(c)
    q = np.concatenate(quads)
    verts, inv = np.unique(q.reshape(-1, 3), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1, 4)
    origin = np.array([a[0] for a in ax]) - 1.5 * res
    return origin + verts * res, np.concatenate([inv[:, [0, 1, 2]], inv[:, [0, 2, 3]]])


def cutaway(lab, ax3, case):
    """Remove the vertical quarter x > 0, y > 0 of a single box; dimers are shown whole."""
    if case['kind'] == 'dimer':
        return lab
    X, Y, _ = np.meshgrid(*ax3, indexing='ij')
    return lab * ~((X > 0) & (Y > 0))


def write_vtk(path, grid, dn):
    nx, ny, nz = grid.shape
    with open(path, 'w') as fh:
        fh.write('# vtk DataFile Version 3.0\nADDA input: relative refractive index (1 = water)\nASCII\n'
                 f'DATASET STRUCTURED_POINTS\nDIMENSIONS {nx} {ny} {nz}\n'
                 f'ORIGIN {-(nx - 1) / 2 * dn:.4f} {-(ny - 1) / 2 * dn:.4f} {-(nz - 1) / 2 * dn:.4f}\n'
                 f'SPACING {dn} {dn} {dn}\nPOINT_DATA {nx * ny * nz}\nSCALARS relative_index float 1\nLOOKUP_TABLE default\n')
        np.savetxt(fh, grid.transpose(2, 1, 0).ravel(), fmt='%.6f')


def main():
    vdir = os.path.join(A.WORK, 'vtk'); os.makedirs(vdir, exist_ok=True)
    n = len(SHOW)
    fig = plt.figure(figsize=(6 * n, 16))
    gs = fig.add_gridspec(3, n, height_ratios=[0.9, 1, 0.8])
    prof_x, prof_z = [], []
    for j, name in enumerate(SHOW):
        case = CASES[name]
        # ---- 3D cutaway
        lab, ax3 = labels_grid(case, 1.0)
        cut = cutaway(lab, ax3, case)
        ax = fig.add_subplot(gs[0, j], projection='3d')
        for mat in [1, 2, 3]:
            filled = cut == mat
            if filled.any():
                ax.voxels(filled, facecolors=matplotlib.colors.to_rgba(MAT_COL[mat], 0.12 if mat == 3 else 1.0), edgecolor=None)
        ax.view_init(elev=28, azim=40); ax.set_box_aspect(lab.shape); ax.set_axis_off()
        ax.set_title(f"{'ABCDE'[j]}1  {TITLES[name]}", fontsize=13)
        # ---- vertical slice (y = 0) of the exact ADDA input
        grid, axn, dn = adda_input_grid(case)
        j0 = grid.shape[1] // 2
        a2 = fig.add_subplot(gs[1, j])
        im = a2.imshow(grid[:, j0, :].T, origin='lower', cmap='magma', vmin=1.0, vmax=1.09,
                       extent=[axn[0][0] - dn / 2, axn[0][-1] + dn / 2, axn[2][0] - dn / 2, axn[2][-1] + dn / 2])
        H = np.array(A.BOX_NM) / 2
        for cen in A.box_centers(case):
            a2.add_patch(Rectangle((cen[0] - H[0], -H[2]), 2 * H[0], 2 * H[2], fill=False, ls='--', lw=0.8, color='white'))
        a2.set_xlabel('x (nm)'); a2.set_ylabel('z (nm)  (open face at top)')
        a2.set_title(f'{"ABCDE"[j]}2  ADDA input, y = 0 slice ({int((grid > 1).sum()):,} dipoles)', fontsize=11)
        if j == n - 1:
            fig.colorbar(im, ax=a2, fraction=0.046, label='relative index m = n / n$_{water}$')
        i0, k0 = grid.shape[0] // 2, grid.shape[2] // 2
        prof_x.append((name, axn[0], (grid[:, j0, k0] - 1) * A.N_MEDIUM))
        prof_z.append((name, axn[2], (grid[i0, j0, :] - 1) * A.N_MEDIUM))
        write_vtk(os.path.join(vdir, f'{name}.vtk'), grid, dn)
    # ---- profiles
    dl0 = A.material_deltas(CASES['bare']); dlp = A.material_deltas(CASES['pllpeg_f0.1'])
    dn_wall = np.sqrt(A.N_MEDIUM ** 2 + dl0['wall']) - A.N_MEDIUM
    dn_peg = np.sqrt(A.N_MEDIUM ** 2 + dlp['peg']) - A.N_MEDIUM
    pcol = ['black', '#E6862E', '#8C510A', '#2CA02C']
    gsb = gs[2, :].subgridspec(1, 2)
    for k, (prof, xl) in enumerate([(prof_x, 'x through the centre (nm): side walls'),
                                    (prof_z, 'z through the centre (nm): bottom (left), open top (right)')]):
        ap = fig.add_subplot(gsb[0, k])
        for (name, x, d), colr in zip(prof, pcol):
            bare = name == 'bare'                    # drawn dashed on top: the coated profiles coincide with it in the walls
            ap.step(x, d, where='mid', color='black' if bare else colr, lw=1.4 if bare else 1.8, ls='--' if bare else '-',
                    zorder=5 if bare else 2, label=TITLES[name])
        ap.set_xlabel(xl); ap.set_ylabel('n − n$_{water}$ of ADDA dipoles')
        ap.set_title(f"{'FG'[k]}  Index profile" + (f': wall Δn ≈ {dn_wall:.3f}, silica 0.107, PEG brush (10%) ≈ {dn_peg:.3f}'
                                                     if k == 0 else ' along z'), fontsize=13)
        if k == 1:
            ap.legend(fontsize=11, frameon=False, loc='center right')
    fig.suptitle('Particle models given to ADDA: open-top box, side walls 2 helices (5 nm), bottom 1 helix (2.5 nm); '
                 '0.5 nm dipoles; 405 nm in water', y=1.0)
    out = os.path.join(A.FIG_OUT, 'fig33_adda_geometries.svg')
    fig.savefig(out, bbox_inches='tight')
    prev = os.environ.get('FIG_PREVIEW_DIR')
    if prev:
        fig.savefig(os.path.join(prev, 'fig33_adda_geometries.png'), dpi=60, bbox_inches='tight')
    print('wrote', out)
    # ---- interactive viewer: voxel surfaces (1 nm)
    traces, vis = [], []
    names = SHOW + list(A.DIMERS)
    for i, nm in enumerate(names):
        case = CASES[nm]
        lab, ax3 = labels_grid(case, 1.0)
        cut = cutaway(lab, ax3, case)
        for mat in [1, 2, 3]:
            if not (cut == mat).any():
                continue
            vtx, tri = voxel_mesh(cut == mat, ax3, 1.0)
            traces.append(dict(type='mesh3d', x=np.round(vtx[:, 0], 2).tolist(), y=np.round(vtx[:, 1], 2).tolist(),
                               z=np.round(vtx[:, 2], 2).tolist(), i=tri[:, 0].tolist(), j=tri[:, 1].tolist(), k=tri[:, 2].tolist(),
                               name=MAT_NAME[mat], showlegend=True, color=MAT_COL[mat], opacity=0.18 if mat == 3 else 1.0,
                               flatshading=True, lighting=dict(ambient=0.45, diffuse=0.9, specular=0.05), visible=(i == 0)))
            vis.append(i)
    sub = lambda nm: ' (quarter cut away; nm)' if CASES[nm]['kind'] != 'dimer' else ' (nm)'
    buttons = [dict(label=TITLES[nm], method='update',
                    args=[{'visible': [v == i for v in vis]}, {'title': {'text': TITLES[nm] + sub(nm)}}])
               for i, nm in enumerate(names)]
    layout = dict(title={'text': TITLES[names[0]] + sub(names[0]), 'x': 0.5},
                  scene=dict(aspectmode='data', xaxis_title='x (nm)', yaxis_title='y (nm)', zaxis_title='z (nm)'),
                  updatemenus=[dict(type='dropdown', x=0, y=1.0, xanchor='left', yanchor='top', buttons=buttons)],
                  margin=dict(l=0, r=0, t=60, b=0))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ADDA origami geometries</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.27.0/plotly.min.js"></script>
<style>body{{margin:0;font-family:Arial,sans-serif;background:#fff}} #p{{width:100vw;height:92vh}} p{{margin:6px 12px;font-size:13px;color:#444}}</style>
</head><body><p>Particle models given to ADDA: open-top box, side walls 2 helices (5 nm), bottom 1 helix (2.5 nm); 1 nm display
resolution; single boxes have a vertical quarter removed, dimers are shown whole. Blue: DNA; orange: silica; green (translucent):
PEG brush. Pick a case from the menu; drag to rotate, scroll to zoom.</p><div id="p"></div>
<script>Plotly.newPlot('p', {json.dumps(traces)}, {json.dumps(layout)}, {{responsive: true}});</script></body></html>"""
    hpath = os.path.join(A.FIG_OUT, 'adda_geometry_viewer.html')
    open(hpath, 'w', encoding='utf8').write(html)
    print('wrote', hpath, f'({os.path.getsize(hpath) / 1e6:.1f} MB)')
    print('wrote VTK files to', vdir)


if __name__ == '__main__':
    main()
