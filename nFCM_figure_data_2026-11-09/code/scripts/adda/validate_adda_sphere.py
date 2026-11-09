"""Validation: ADDA (built-in sphere) vs exact Mie theory for a sphere of the origami's equal volume.

Mie series implemented after Bohren & Huffman (1983), BHMIE. Sphere diameter 38.6 nm (equal volume to the
30.6 x 30.6 x 32.3 nm envelope); relative indices 1.096 (DNA wall) and 1.080 (silica); 405 nm in water.
Agreement of the extinction (= scattering) cross-section within ~1 % confirms the ADDA set-up and units.
Run:  python validate_adda_sphere.py
"""
import os, sys, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adda_common as A


def bhmie_q(x, m):
    nstop = int(np.ceil(x + 4 * x ** (1 / 3) + 2))
    y = m * x
    nmx = int(max(nstop, abs(y)) + 15)
    D = np.zeros(nmx + 1, complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / y - 1 / (D[n] + n / y)
    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = complex(psi1, -chi1)
    qext = qsca = 0.0
    for n in range(1, nstop + 1):
        psi = (2 * n - 1) * psi1 / x - psi0
        chi = (2 * n - 1) * chi1 / x - chi0
        xi = complex(psi, -chi)
        an = ((D[n] / m + n / x) * psi - psi1) / ((D[n] / m + n / x) * xi - xi1)
        bn = ((D[n] * m + n / x) * psi - psi1) / ((D[n] * m + n / x) * xi - xi1)
        qsca += (2 * n + 1) * (abs(an) ** 2 + abs(bn) ** 2)
        qext += (2 * n + 1) * (an + bn).real
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = complex(psi1, -chi1)
    return 2 * qext / x ** 2, 2 * qsca / x ** 2


def main():
    D_um = 2 * (3 * np.prod(A.BOX_NM) / (4 * np.pi)) ** (1 / 3) / 1000
    x = np.pi * D_um / A.LAMBDA_MED_UM
    print(f'sphere diameter {D_um * 1000:.1f} nm, size parameter {x:.3f}')
    os.makedirs(os.path.join(A.WORK, 'validation'), exist_ok=True)      # ADDA does not create parent folders
    for m in [1.096, 1.080]:
        qext, qsca = bhmie_q(x, m)
        c_mie = qsca * np.pi * (D_um / 2) ** 2
        for grid in [32, 64]:
            rdir = os.path.join(A.WORK, 'validation', f'sphere_m{m}_grid{grid}')
            if not os.path.exists(os.path.join(rdir, 'CrossSec-Y')):
                subprocess.run([A.ADDA_EXE, '-shape', 'sphere', '-size', f'{D_um:.8f}', '-lambda', f'{A.LAMBDA_MED_UM:.8f}',
                                '-m', str(m), '0', '-grid', str(grid), '-dir', rdir], capture_output=True, text=True)
            c_adda = float([l.split('=')[1] for l in open(os.path.join(rdir, 'CrossSec-Y')) if l.startswith('Cext')][0])
            print(f'  m = {m}: Mie Csca = {c_mie:.6e} um^2;  ADDA grid {grid}: {c_adda:.6e} um^2  '
                  f'({100 * (c_adda / c_mie - 1):+.2f} %)')


if __name__ == '__main__':
    main()
