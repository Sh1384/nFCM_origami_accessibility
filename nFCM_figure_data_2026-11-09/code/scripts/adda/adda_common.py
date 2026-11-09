"""Shared geometry and settings for the scattering models of bare and coated Pandora-box origami.

Used by the ADDA scripts in this folder and by ../model_coating_scattering.py, so both describe the same particle.
Units: nm internally; ADDA takes um (wavelength in the medium, refractive indices relative to the medium).

Origami (open-top box): outer 30.6 x 30.6 x 32.3 nm (x, y, z), open at +z.
  side walls  two helices thick  (SIDE_WALL_NM = 5.0 nm)
  bottom      one helix thick    (BOTTOM_NM = 2.5 nm)
  The DNA (16,128 nt x 330 Da) is spread uniformly over the wall volume: n_wall = n_m + (dn/dc) c_wall.
Coatings (thickness measured from the nearest DNA surface, so edges and corners are rounded):
  silica 'all'        conformal layer on every DNA surface in contact with water: outer faces, rim, inner side walls
                      and cavity floor (the open top gives access to the interior)
  silica 'exterior'   same, but nothing inside the cavity column (outer faces and rim only)
  PLL-g-PEG           PLL(200)-g[5]-PEG(5k) (this study): a fraction f of phosphates paired with a lysine amine; the
                      lysines are added to the wall index and the PEG chains fill a conformal brush layer of thickness =
                      PEG Flory radius on all exposed surfaces (inside and out)
  dimers              two bare boxes in contact, DIMERS below: side to side, edge to edge (vertical or horizontal edge),
                      and stacked (bottom on rim, bottom to bottom, rim to rim); 'up' = direction of the open face
Partial-volume voxels: each voxel's excess permittivity is the mean over SUBSAMPLE^3 sub-points, so layers thinner than
a voxel keep the right polarizability; voxel values are grouped into at most MAX_DOMAINS ADDA materials.
"""
import os
from functools import lru_cache
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MANUSCRIPT = os.path.abspath(os.path.join(HERE, '..', '..'))
ADDA_EXE = os.path.join(MANUSCRIPT, 'tools', 'adda', 'adda-1.4.0', 'win64', 'adda.exe')
WORK = os.path.join(MANUSCRIPT, 'adda_runs')
FIG_OUT = os.path.join(MANUSCRIPT, 'figures_svg')

# ---------------------------------------------------------------- optics
LAMBDA_VAC_NM = 405.0
N_MEDIUM = 1.343
LAMBDA_MED_UM = LAMBDA_VAC_NM / N_MEDIUM / 1000.0

# ---------------------------------------------------------------- origami
BOX_NM = (30.6, 30.6, 32.3)          # outer dimensions; open face at +z
SIDE_WALL_NM = 5.0                   # two helices (2 x 2.5 nm)
BOTTOM_NM = 2.5                      # one helix
NUCLEOTIDES = 16128
DA_PER_NT = 330.0
DNDC_DNA, DNDC_PEPTIDE, DNDC_PEG = 0.180, 0.200, 0.140
AVOG = 6.02214076e23

# ---------------------------------------------------------------- PLL-g-PEG used in this study
PLL_G = 5.0
PEG_DA = 5000.0
LYS_DA = 128.17
PEG_BRUSH_NM = 5.9

# ---------------------------------------------------------------- discretisation
DIPOLE_NM = 0.5
SUBSAMPLE = 4
MAX_DOMAINS = 14
PROPS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

# ---------------------------------------------------------------- cases
CASES = [dict(name='bare', kind='bare')]
SILICA_T = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
for t in SILICA_T:
    CASES.append(dict(name=f'silica_all_{t:g}nm', kind='silica', coat='all', t=t, n_silica=1.45))
for t in SILICA_T:
    CASES.append(dict(name=f'silica_ext_{t:g}nm', kind='silica', coat='exterior', t=t, n_silica=1.45))
CASES.append(dict(name='silica_all_1nm_n1.40', kind='silica', coat='all', t=1.0, n_silica=1.40))
for f in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 1.0]:
    CASES.append(dict(name=f'pllpeg_f{f:g}', kind='pllpeg', f=f))
_hx, _hy, _hz = BOX_NM[0] / 2, BOX_NM[1] / 2, BOX_NM[2] / 2
DIMERS = {   # name: (label, [(box centre, up = +1 open at +z / -1 open at -z), ...])
    'dimer_side': ('side to side (wall on wall)', [((-_hx, 0, 0), 1), ((_hx, 0, 0), 1)]),
    'dimer_edge_vertical': ('edge to edge (vertical edges)', [((-_hx, -_hy, 0), 1), ((_hx, _hy, 0), 1)]),
    'dimer_edge_horizontal': ('edge to edge (rim edge on bottom edge)', [((-_hx, 0, -_hz), 1), ((_hx, 0, _hz), 1)]),
    'dimer_stack_bottom_on_rim': ('stacked: bottom on rim (same orientation)', [((0, 0, -_hz), 1), ((0, 0, _hz), 1)]),
    'dimer_stack_bottom_to_bottom': ('stacked: bottom to bottom', [((0, 0, -_hz), -1), ((0, 0, _hz), 1)]),
    'dimer_stack_rim_to_rim': ('stacked: rim to rim (open faces together)', [((0, 0, -_hz), 1), ((0, 0, _hz), -1)]),
}
for _n in DIMERS:
    CASES.append(dict(name=_n, kind='dimer'))
CONVERGENCE = [dict(c, name=c['name'] + '_d0.25', dipole_nm=0.25) for c in CASES if c['name'] in ('bare', 'silica_all_1nm')]


# ---------------------------------------------------------------- geometry
def boxes(case):
    """[(centre, up), ...] for the boxes of a case; up = +1 if the open face points to +z."""
    if case.get('kind') == 'dimer':
        return [(np.array(c, float), u) for c, u in DIMERS[case['name'].replace('_d0.25', '')][1]]
    return [(np.zeros(3), 1)]


def box_centers(case):
    return [c for c, _ in boxes(case)]


def slabs(center, up=1):
    """Axis-aligned slabs (centre, half-sizes) whose union is the open box walls."""
    a, b, c = BOX_NM; w, fb = SIDE_WALL_NM, BOTTOM_NM
    hx, hy, hz = a / 2, b / 2, c / 2
    return [(center + np.array([-(hx - w / 2), 0, 0]), np.array([w / 2, hy, hz])),
            (center + np.array([+(hx - w / 2), 0, 0]), np.array([w / 2, hy, hz])),
            (center + np.array([0, -(hy - w / 2), 0]), np.array([hx - w, w / 2, hz])),
            (center + np.array([0, +(hy - w / 2), 0]), np.array([hx - w, w / 2, hz])),
            (center + np.array([0, 0, -up * (hz - fb / 2)]), np.array([hx - w, hy - w, fb / 2]))]


def membership(p, case):
    """For points p (..., 3): (in DNA wall, distance to nearest DNA, inside a cavity column)."""
    a, b, c = BOX_NM; w, fb = SIDE_WALL_NM, BOTTOM_NM
    dist = np.full(p.shape[:-1], np.inf); in_col = np.zeros(p.shape[:-1], bool)
    for cen, up in boxes(case):
        for sc, sh in slabs(cen, up):
            dist = np.minimum(dist, np.sqrt((np.clip(np.abs(p - sc) - sh, 0, None) ** 2).sum(-1)))
        r = p - cen
        in_col |= (np.abs(r[..., 0]) < a / 2 - w) & (np.abs(r[..., 1]) < b / 2 - w) & (up * r[..., 2] > -c / 2 + fb)
    return dist <= 0, dist, in_col


def wall_volume():
    a, b, c = BOX_NM
    return a * b * c - (a - 2 * SIDE_WALL_NM) * (b - 2 * SIDE_WALL_NM) * (c - BOTTOM_NM)


@lru_cache(maxsize=None)
def coating_volume(t, coat='all', n=6_000_000, seed=1):
    """Monte Carlo volume (nm^3) of a coating of thickness t on one box ('all' or 'exterior')."""
    if t <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    lo = -np.array(BOX_NM) / 2 - t; hi = np.array(BOX_NM) / 2 + t
    hit = 0; chunk = 500_000
    for i in range(0, n, chunk):
        p = rng.uniform(lo, hi, (min(chunk, n - i), 3))
        wall, dist, col = membership(p, {'kind': 'bare'})
        sel = ~wall & (dist <= t)
        if coat == 'exterior':
            sel &= ~col
        hit += sel.sum()
    return hit / n * np.prod(hi - lo)


def material_deltas(case):
    """Excess permittivity (eps - eps_m) of each material."""
    em = N_MEDIUM ** 2
    v_wall_ml = wall_volume() * 1e-21
    n_wall = N_MEDIUM + DNDC_DNA * (NUCLEOTIDES * DA_PER_NT / AVOG) / v_wall_ml
    d = {'peg': 0.0, 'silica': 0.0}
    if case['kind'] == 'pllpeg':
        f = case['f']
        lys_g = f * NUCLEOTIDES * PLL_G / (PLL_G - 1) * LYS_DA / AVOG
        peg_g = f * NUCLEOTIDES / (PLL_G - 1) * PEG_DA / AVOG
        n_wall += DNDC_PEPTIDE * lys_g / v_wall_ml
        n_peg = N_MEDIUM + DNDC_PEG * peg_g / (coating_volume(PEG_BRUSH_NM, 'all') * 1e-21)
        d['peg'] = n_peg ** 2 - em
    if case['kind'] == 'silica':
        d['silica'] = case['n_silica'] ** 2 - em
    d['wall'] = n_wall ** 2 - em
    return d


def build_geometry(case, dipole_nm=None):
    """Integer voxel coordinates (N, 3), per-voxel excess permittivity (N,), grid size along x, dipole size."""
    dn = dipole_nm or case.get('dipole_nm', DIPOLE_NM)
    H = np.array(BOX_NM) / 2
    t = case.get('t', 0.0) if case['kind'] == 'silica' else 0.0
    h = PEG_BRUSH_NM if case['kind'] == 'pllpeg' else 0.0
    pad = max(t, h)
    cens = box_centers(case)
    lo = np.min([c - H - pad for c in cens], axis=0); hi = np.max([c + H + pad for c in cens], axis=0)
    n = np.ceil((hi - lo) / dn).astype(int)
    origin = (lo + hi) / 2 - n * dn / 2
    dl = material_deltas(case)
    ss = SUBSAMPLE
    off = ((np.arange(ss) + 0.5) / ss - 0.5) * dn
    sub = np.stack(np.meshgrid(off, off, off, indexing='ij'), -1).reshape(-1, 3)
    ii, jj = np.meshgrid(np.arange(n[0]), np.arange(n[1]), indexing='ij'); ii, jj = ii.ravel(), jj.ravel()
    coords, vals = [], []
    for k in range(n[2]):
        vc = np.stack([origin[0] + (ii + 0.5) * dn, origin[1] + (jj + 0.5) * dn,
                       np.full(ii.shape, origin[2] + (k + 0.5) * dn)], -1)
        p = vc[:, None, :] + sub[None, :, :]
        wall, dist, col = membership(p, case)
        de = wall * dl['wall']
        if t > 0:
            sel = ~wall & (dist <= t)
            if case.get('coat') == 'exterior':
                sel &= ~col
            de = de + sel * dl['silica']
        if h > 0:
            de = de + (~wall & (dist <= h)) * dl['peg']
        v = de.mean(1); keep = v > 0
        if keep.any():
            coords.append(np.stack([ii[keep], jj[keep], np.full(keep.sum(), k)], -1)); vals.append(v[keep])
    coords = np.concatenate(coords); vals = np.concatenate(vals)
    coords = coords - coords.min(0)
    return coords, vals, int(coords[:, 0].max() + 1), dn


def analytic_optical_volume(case):
    """sum (eps - eps_m) x volume (nm^3) from the exact geometry (coating volumes by Monte Carlo)."""
    dl = material_deltas(case)
    tot = dl['wall'] * wall_volume() * (2 if case['kind'] == 'dimer' else 1)
    if case['kind'] == 'silica':
        tot += dl['silica'] * coating_volume(case['t'], case['coat'])
    if case['kind'] == 'pllpeg':
        tot += dl['peg'] * coating_volume(PEG_BRUSH_NM, 'all')
    return tot


def quantize(vals, max_domains=MAX_DOMAINS):
    """Group voxel values into <= max_domains ADDA materials (domain value = mean of its voxels, so the total excess
    optical volume is preserved). Pure materials (values shared by >= 2% of voxels) keep their own domain; the
    partial-volume voxels are split into equal-count bins with the remaining domains."""
    r = np.round(vals, 10)
    u, inv, cnt = np.unique(r, return_inverse=True, return_counts=True)
    if len(u) <= max_domains:
        return inv, u
    pure = np.argsort(cnt)[::-1][:max_domains // 2]
    pure = pure[cnt[pure] >= 0.02 * len(vals)]
    is_pure = np.isin(inv, pure)
    dom = np.full(len(vals), -1)
    dom[is_pure] = np.searchsorted(np.sort(pure), inv[is_pure])
    rest = vals[~is_pure]
    nb = max_domains - len(pure)
    edges = np.unique(np.quantile(rest, np.linspace(0, 1, nb + 1)))
    dom[~is_pure] = len(pure) + np.clip(np.searchsorted(edges, rest, side='right') - 1, 0, len(edges) - 2)
    present = np.unique(dom); remap = np.zeros(dom.max() + 1, int); remap[present] = np.arange(len(present))
    dom = remap[dom]
    return dom, np.array([vals[dom == i].mean() for i in range(len(present))])


def write_geometry(path, coords, dom, nmat, case_name):
    with open(path, 'w') as fh:
        fh.write(f'#Pandora-box origami (open top) case: {case_name}\n#generated by adda_common.py\nNmat={nmat}\n')
        np.savetxt(fh, np.column_stack([coords, dom + 1]), fmt='%d')


# ---------------------------------------------------------------- Rayleigh-Gans-Debye (exact orientation average)
def q_at(theta_deg):
    return 4 * np.pi / (LAMBDA_MED_UM * 1000.0) * np.sin(np.radians(theta_deg) / 2)


def debye(case, res=1.0, bw=0.1):
    """Orientation-averaged RGD intensity of the voxel model (Debye formula on res-nm partial-volume voxels).
    Returns (I at 90 deg, I integrated over angle with the unpolarised dipole factor (1 + cos^2)/2 = cross-section,
    I at THETA_GRID), all proportional to (sum of excess permittivity x volume)^2 x form factor."""
    coords, vals, nx, dn = build_geometry(case, dipole_nm=res)
    x = coords * dn; w = vals * dn ** 3
    nb = int(np.ceil(np.sqrt(((x.max(0) - x.min(0)) ** 2).sum()) / bw)) + 2
    h = np.zeros(nb)
    for i in range(0, len(x), 1500):
        r = np.sqrt(((x[i:i + 1500, None, :] - x[None, :, :]) ** 2).sum(-1))
        h += np.bincount(np.rint(r / bw).astype(int).ravel(), (w[i:i + 1500, None] * w[None, :]).ravel(), nb)
    r = np.arange(nb) * bw
    th = np.linspace(0, np.pi, 721)
    I = np.array([(h * np.sinc(q_at(np.degrees(t)) * r / np.pi)).sum() for t in th])
    i90 = (h * np.sinc(q_at(90.0) * r / np.pi)).sum()
    return i90, np.trapz(I * (1 + np.cos(th) ** 2) / 2 * np.sin(th), th), I[::12]


THETA_GRID = np.linspace(0, 180, 61)          # debye() angular output (3 deg steps)


def debye_ratio(case, res=1.0, curve=False):
    """(I90 / I90_bare, Csca / Csca_bare[, I(theta) / I_bare(theta) at THETA_GRID]) from debye(), cached in
    adda_runs/rgd_debye.json."""
    import json
    path = os.path.join(WORK, 'rgd_debye.json')
    cache = json.load(open(path)) if os.path.exists(path) else {}
    for c in [CASES[0], case]:
        key = f"{c['name']}|{res}"
        if key not in cache:
            i90, cs, ith = debye(c, res)
            cache[key] = [float(i90), float(cs)] + list(map(float, ith))
            json.dump(cache, open(path, 'w'), indent=1)
    b = np.array(cache[f"{CASES[0]['name']}|{res}"]); v = np.array(cache[f"{case['name']}|{res}"])
    # I(theta = 0) = (sum of voxel polarisabilities)^2: rescale it to the exact optical volume so voxel-grid alignment
    # (which differs between the monomer and boxes offset by 32.3 nm) does not enter the ratio
    norm = (analytic_optical_volume(case) / analytic_optical_volume(CASES[0])) ** 2 / (v[2] / b[2])
    out = (v[0] / b[0] * norm, v[1] / b[1] * norm)
    return out + (v[2:] / b[2:] * norm,) if curve else out
