"""Theoretical side-scatter intensity of coated Pandora-box origami relative to bare origami.

Physics
  The origami (~31 nm) is much smaller than the 405 nm side-scatter wavelength (VSSC-1) and every component has low
  optical contrast against water, so scattering is in the Rayleigh-Gans-Debye (RGD) regime. The scattered field is
  proportional to the particle's integrated excess polarizability (excess optical volume)
        A = sum_i (n_i - n_m) V_i  =  sum_i (dn/dc)_i m_i        [nm^3]
  and I = K A^2 P(q), P(q) ~ exp(-q^2 Rg^2 / 3) (Guinier), Rg the excess-polarizability-weighted radius of gyration,
  q at 90 deg side scatter. Relative to bare origami: I / I0 = (A / A0)^2 * P(q) / P0(q). The ratio is independent of
  the collection optics to within ~2 % and is checked against ADDA (discrete-dipole) calculations (scripts/adda).

Bare origami (this paper): open-top box, outer 30.6 x 30.6 x 32.3 nm, side walls two helices thick (5.0 nm), bottom one
  helix thick (2.5 nm), ~16,128 nt (330 Da/nt -> 5.3 MDa). Geometry shared with the ADDA scripts (scripts/adda/
  adda_common.py). Coating volumes and Rg are computed by Monte Carlo sampling (20 million points) of the exact shape.

Silica: sol-gel silica of thickness t (nm) measured from the nearest DNA surface, n_silica 1.45 (1.40 porous - 1.47 fused).
  all exposed surfaces     conformal on outer faces, rim, inner side walls and cavity floor (the open top exposes the
                           interior); primary
  exterior only            outer faces and rim only (nothing inside the cavity)
  Stoichiometric ceiling: nucleotide:TMAPS:TEOS = 1:3:20 gives at most 20 SiO2 per nucleotide (19.4 MDa, 2.0 g/cm^3)
  if every TEOS condensed onto the origami; the thickness this corresponds to is marked.

PLL-PEG: fraction f of DNA phosphates paired with a lysine amine (f = 1: full neutralisation, the N:P = 1 limit). Grafted
  lysines carry PEG and are not charged, so per bound amine there are g/(g-1) lysines and 1/(g-1) PEG chains (graft ratio
  g). Architectures: PLL(200)-g[5]-PEG(5k) (this study), PLL(20k)-g[3.5]-PEG(2k), PLL(20k)-g[3.5]-PEG(5k), K10-PEG5k,
  PLL alone. Lysines sit on the DNA; PEG fills a conformal brush layer (thickness = Flory radius) on all exposed surfaces
  (affects only P(q)).

Outputs (figures_svg/): model_coating_scattering.svg, _silica.csv, _pllpeg.csv, _summary.csv, _parameters.csv
Run: python model_coating_scattering.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adda'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import make_manuscript_figures as M
import adda_common as G

# ------------------------------------------------------------------ parameters
P = dict(
    box_a_nm=G.BOX_NM[0], box_b_nm=G.BOX_NM[1], box_c_nm=G.BOX_NM[2], open_face='+z (top)',
    side_wall_nm=G.SIDE_WALL_NM, bottom_wall_nm=G.BOTTOM_NM,
    nucleotides=G.NUCLEOTIDES, dalton_per_nt=G.DA_PER_NT,
    wavelength_nm=G.LAMBDA_VAC_NM, n_medium=G.N_MEDIUM, angle_deg=90.0,
    dndc_DNA=G.DNDC_DNA, dndc_peptide=G.DNDC_PEPTIDE, dndc_PEG=G.DNDC_PEG,
    n_silica=1.45, n_silica_low=1.40, n_silica_high=1.47,
    silica_density=2.0, TEOS_per_nt=20.0,
    lysine_residue_Da=G.LYS_DA,
)
AVOG = 6.02214076e23
NM3_PER_ML = 1e21
q = 4 * np.pi * P['n_medium'] / P['wavelength_nm'] * np.sin(np.radians(P['angle_deg']) / 2)
N_P = P['nucleotides']
M_DNA = P['nucleotides'] * P['dalton_per_nt']


def optical_volume(mass_Da, dndc):
    """Excess optical volume (n - n_m) V in nm^3 for a macromolecule of given mass and dn/dc."""
    return mass_Da / AVOG * dndc * NM3_PER_ML


A0 = optical_volume(M_DNA, P['dndc_DNA'])

# ------------------------------------------------------------------ Monte Carlo moments of the exact geometry
TMAX, DB, NMC = 8.2, 0.01, 20_000_000
H = np.array(G.BOX_NM) / 2
LO, HI = -H - TMAX, H + TMAX
VBOX = np.prod(HI - LO)
NB = int(round(TMAX / DB))
acc = {k: dict(n=np.zeros(NB), s1=np.zeros((NB, 3)), s2=np.zeros(NB)) for k in ['all', 'exterior']}
dna = dict(n=0, s1=np.zeros(3), s2=0.0)
rng = np.random.default_rng(0)
for i in range(0, NMC, 1_000_000):
    p = rng.uniform(LO, HI, (1_000_000, 3))
    wall, dist, col = G.membership(p, {'kind': 'bare'})
    dna['n'] += wall.sum(); dna['s1'] += p[wall].sum(0); dna['s2'] += (p[wall] ** 2).sum()
    for key, sel in [('all', ~wall & (dist < TMAX)), ('exterior', ~wall & (dist < TMAX) & ~col)]:
        b = np.minimum((dist[sel] / DB).astype(int), NB - 1); pp = p[sel]
        acc[key]['n'] += np.bincount(b, minlength=NB)
        acc[key]['s2'] += np.bincount(b, weights=(pp ** 2).sum(1), minlength=NB)
        for j in range(3):
            acc[key]['s1'][:, j] += np.bincount(b, weights=pp[:, j], minlength=NB)
DV = VBOX / NMC
for key in acc:
    for k in ['n', 's1', 's2']:
        acc[key][k] = np.concatenate([np.zeros((1,) + acc[key][k].shape[1:]), np.cumsum(acc[key][k], axis=0)])
MU_D = dna['s1'] / dna['n']; M2_D = dna['s2'] / dna['n']
RG2_DNA = M2_D - MU_D @ MU_D


def layer(t, coat='all'):
    """(volume nm^3, mean position, mean r^2) of a coating layer of thickness t."""
    k = int(round(min(t, TMAX - DB) / DB)); a = acc[coat]
    if a['n'][k] == 0:
        return 0.0, np.zeros(3), 0.0
    return a['n'][k] * DV, a['s1'][k] / a['n'][k], a['s2'][k] / a['n'][k]


def v_coat(t, coat='all'):
    return layer(t, coat)[0]


def intensity_ratio(components):
    """components: list of (optical volume nm^3, mean position, mean r^2); DNA is added automatically."""
    comps = [(A0, MU_D, M2_D)] + [c for c in components if c[0] > 0]
    A = sum(c[0] for c in comps)
    mu = sum(c[0] * c[1] for c in comps) / A; m2 = sum(c[0] * c[2] for c in comps) / A
    rg2_tot = m2 - mu @ mu
    Pq = np.exp(-q ** 2 * rg2_tot / 3); P0 = np.exp(-q ** 2 * RG2_DNA / 3)
    return (A / A0) ** 2 * Pq / P0, A / A0, np.sqrt(rg2_tot), Pq / P0


# ------------------------------------------------------------------ silica
GEOMS = {'all exposed surfaces': 'all', 'exterior only': 'exterior'}
T = np.round(np.arange(0, 8.01, 0.05), 2)
rows = []
for t in T:
    for geom, coat in GEOMS.items():
        v, mu, m2 = layer(t, coat)
        for lab, n_si in [('main', P['n_silica']), ('low', P['n_silica_low']), ('high', P['n_silica_high'])]:
            dn = n_si - P['n_medium']
            r, amp, rg, pr = intensity_ratio([(dn * v, mu, m2)])
            rows.append(dict(thickness_nm=t, geometry=geom, n_silica=n_si, n_case=lab, silica_volume_nm3=v,
                             silica_mass_MDa=v * P['silica_density'] / NM3_PER_ML * AVOG / 1e6,
                             amplitude_ratio=amp, Rg_nm=rg, form_factor_ratio=pr, intensity_ratio=r))
Si = pd.DataFrame(rows)

m_si_max = N_P * P['TEOS_per_nt'] * 60.08
v_si_max = m_si_max / AVOG / P['silica_density'] * NM3_PER_ML
t_max = {g: float(np.interp(v_si_max, [v_coat(t, c) for t in T], T)) for g, c in GEOMS.items()}

# ------------------------------------------------------------------ PLL-PEG
ARCH = {'PLL(200)-g[5]-PEG(5k) (this study)': dict(g=5.0, peg=5000, h=5.9),
        'PLL(20k)-g[3.5]-PEG(2k)': dict(g=3.5, peg=2000, h=3.4),
        'PLL(20k)-g[3.5]-PEG(5k)': dict(g=3.5, peg=5000, h=5.9),
        'K10-PEG(5k)': dict(g=None, peg=5000, per_amine_peg=0.1, h=5.9),
        'PLL only': dict(g=None, peg=0, per_amine_peg=0.0, h=0.0)}
F = np.r_[np.linspace(0, 0.2, 81), np.linspace(0.21, 1, 80)]
rows = []
for name, a in ARCH.items():
    if a['g'] is not None:
        lys_per_amine, peg_per_amine = a['g'] / (a['g'] - 1), 1 / (a['g'] - 1)
    else:
        lys_per_amine, peg_per_amine = 1.0, a['per_amine_peg']
    v_peg, mu_peg, m2_peg = layer(a['h'], 'all') if a['h'] > 0 else (0.0, np.zeros(3), 0.0)
    for f in F:
        m_lys = f * N_P * lys_per_amine * P['lysine_residue_Da']
        m_peg = f * N_P * peg_per_amine * a['peg']
        comps = [(optical_volume(m_lys, P['dndc_peptide']), MU_D, M2_D)]   # lysines sit on the DNA wall
        if m_peg > 0:
            comps.append((optical_volume(m_peg, P['dndc_PEG']), mu_peg, m2_peg))
        r, amp, rg, pr = intensity_ratio(comps)
        rows.append(dict(architecture=name, fraction_phosphates_bound=f, polymer_mass_MDa=(m_lys + m_peg) / 1e6,
                         polymer_to_DNA_mass=(m_lys + m_peg) / M_DNA, amplitude_ratio=amp, Rg_nm=rg,
                         form_factor_ratio=pr, intensity_ratio=r))
Pp = pd.DataFrame(rows)

# ------------------------------------------------------------------ summary
summ = []
for geom in GEOMS:
    for case in ['main', 'low', 'high']:
        z = Si[(Si.geometry == geom) & (Si.n_case == case)]
        d = dict(model='silica', variant=geom, n_silica=z.n_silica.iloc[0])
        for t in [0.5, 1, 2, 3, 5]:
            d[f'I/I0 at {t} nm'] = float(np.interp(t, z.thickness_nm, z.intensity_ratio))
        for target in [1.25, 1.5, 2, 4]:
            d[f'thickness (nm) for {target}x'] = float(np.interp(target, z.intensity_ratio, z.thickness_nm))
        d['stoichiometric max thickness (nm)'] = t_max[geom]
        d['I/I0 at stoichiometric max'] = float(np.interp(t_max[geom], z.thickness_nm, z.intensity_ratio))
        summ.append(d)
for name in ARCH:
    z = Pp[Pp.architecture == name]
    d = dict(model='PLL-PEG', variant=name)
    for f in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
        d[f'I/I0 at f={f}'] = float(np.interp(f, z.fraction_phosphates_bound, z.intensity_ratio))
    for target in [1.1, 1.25, 1.5, 2]:
        d[f'f for {target}x'] = (float(np.interp(target, z.intensity_ratio, z.fraction_phosphates_bound))
                                 if z.intensity_ratio.max() >= target else np.nan)
    d['polymer/DNA mass at f=1'] = float(z.polymer_to_DNA_mass.iloc[-1])
    summ.append(d)
S = pd.DataFrame(summ)
Par = pd.DataFrame([dict(parameter=k, value=v) for k, v in P.items()] +
                   [dict(parameter='q at side scatter (1/nm)', value=q),
                    dict(parameter='bare DNA mass (MDa)', value=M_DNA / 1e6),
                    dict(parameter='bare excess optical volume A0 (nm^3)', value=A0),
                    dict(parameter='bare Rg (nm)', value=np.sqrt(RG2_DNA)),
                    dict(parameter='bare form factor P0', value=np.exp(-q ** 2 * RG2_DNA / 3)),
                    dict(parameter='DNA wall volume, exact (nm^3)', value=G.wall_volume()),
                    dict(parameter='DNA wall volume, Monte Carlo (nm^3)', value=dna['n'] * DV),
                    dict(parameter='exposed DNA surface, all (nm^2, ~V(0.05 nm)/0.05)', value=v_coat(0.05) / 0.05),
                    dict(parameter='exposed DNA surface, exterior (nm^2)', value=v_coat(0.05, 'exterior') / 0.05),
                    dict(parameter='max silica mass if all TEOS deposits (MDa)', value=m_si_max / 1e6),
                    dict(parameter='max silica volume (nm^3)', value=v_si_max)])


# ------------------------------------------------------------------ figure
def plain_log_ticks(ax):
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ticks = [1, 1.5, 2, 3, 5, 10, 20, 50]
    ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())


gcol = {'all exposed surfaces': '#E6862E', 'exterior only': '#8C510A'}
acol = dict(zip(ARCH, ['#000000', '#2CA02C', '#1B7837', '#66C2A5', '#7F7F7F']))
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
ax = axes[0, 0]
for geom in gcol:
    z = Si[(Si.geometry == geom) & (Si.n_case == 'main')]
    lo = Si[(Si.geometry == geom) & (Si.n_case == 'low')]; hi = Si[(Si.geometry == geom) & (Si.n_case == 'high')]
    ax.plot(z.thickness_nm, z.intensity_ratio, color=gcol[geom], lw=2.5, label=f'{geom} (n = {P["n_silica"]})')
    ax.fill_between(z.thickness_nm, lo.intensity_ratio.values, hi.intensity_ratio.values, color=gcol[geom], alpha=0.2)
    ax.axvline(t_max[geom], color=gcol[geom], ls=':', lw=1.8)
    ax.text(t_max[geom], 1.05 if geom == 'all exposed surfaces' else 25, f' max if all TEOS\n deposits: {t_max[geom]:.1f} nm',
            color=gcol[geom], fontsize=10, va='bottom')
for yv in [1.5, 2]:
    ax.axhline(yv, color='0.6', ls='--', lw=1)
ax.set_yscale('log'); ax.set_xlim(0, 8); plain_log_ticks(ax)
ax.set_xlabel('Silica thickness (nm)'); ax.set_ylabel('Side-scatter intensity / bare origami')
ax.set_title(f'A  Silica coating (band: n = {P["n_silica_low"]}–{P["n_silica_high"]})', fontsize=14)
ax.legend(fontsize=11, frameon=False, loc='upper left')
ax = axes[0, 1]
ns = np.linspace(1.36, 1.48, 61)
for t, ls in [(0.5, ':'), (1, '-.'), (2, '--'), (3, '-')]:
    v, mu, m2 = layer(t, 'all')
    ax.plot(ns, [intensity_ratio([((n - P['n_medium']) * v, mu, m2)])[0] for n in ns], ls=ls, color='#E6862E',
            lw=2, label=f'{t} nm, all exposed surfaces')
ax.axvline(P['n_medium'], color='0.6', lw=1)
ax.axvspan(P['n_silica_low'], P['n_silica_high'], color='0.85', zorder=0)
ax.set_xlabel('Silica refractive index at 405 nm'); ax.set_ylabel('Side-scatter intensity / bare origami')
ax.set_title('B  Sensitivity to silica density / porosity', fontsize=14); ax.legend(fontsize=11, frameon=False)
for ax, xmax, title, logy in [(axes[1, 0], 1.0, 'C  PLL-PEG: fraction of phosphates bound (full range)', True),
                              (axes[1, 1], 0.15, 'D  PLL-PEG: low coverage (detection range)', False)]:
    for name in ARCH:
        z = Pp[(Pp.architecture == name) & (Pp.fraction_phosphates_bound <= xmax + 1e-9)]
        ax.plot(z.fraction_phosphates_bound, z.intensity_ratio, color=acol[name], lw=2.5, label=name)
    for yv in ([1.1, 1.25] if not logy else [1.5, 2]):
        ax.axhline(yv, color='0.6', ls='--', lw=1)
        if not logy:
            ax.text(xmax, yv, f' +{round((yv - 1) * 100)}%', va='center', fontsize=10, color='0.4')
    if logy:
        ax.set_yscale('log'); plain_log_ticks(ax)
    ax.set_xlim(0, xmax)
    ax.set_xlabel('Fraction of DNA phosphates paired with a lysine (1 = N:P 1, fully bound)')
    ax.set_ylabel('Side-scatter intensity / bare origami'); ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, frameon=False, loc='upper left')
fig.suptitle('Rayleigh-Gans-Debye model: side scatter (405 nm, 90°) of coated Pandora-box origami relative to bare\n'
             f'(open-top box 30.6 × 30.6 × 32.3 nm, side walls 2 helices, bottom 1 helix; {M_DNA / 1e6:.1f} MDa DNA)', y=1.04)
M.save(fig, 'model_coating_scattering', {'_silica': Si, '_pllpeg': Pp, '_summary': S, '_parameters': Par})

pd.set_option('display.width', 250)
print(Par.to_string(index=False))
print(S.T.to_string())
