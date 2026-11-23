"""Step 2: run ADDA for every case in adda_runs/manifest.csv, for incidence along x, y and z.

For each case and propagation direction ADDA is called as
  adda.exe -shape read <geom> -size <nx*d um> -lambda <405 nm / n_water, um> -m <m1> 0 <m2> 0 ... -prop px py pz
           -no_vol_cor -dir adda_runs/runs/<case>_prop<xyz>
Both incident polarisations are computed by ADDA (one is skipped automatically if symmetry makes them equal).
Averaging the 3 directions x 2 polarisations gives the random-orientation average of the scattering cross-section in
the dipole (small-particle) limit, which is what the coated/bare ratio needs (independent of collection optics to
within ~2 %). Finished runs are skipped, so the script can be restarted.
Run:  python run_adda.py                 (all cases)
      python run_adda.py bare silica_all_1nm
"""
import os, sys, subprocess, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adda_common as A


def run_case(row):
    out = []
    for p in A.PROPS:
        tag = ''.join('xyz'[i] for i, v in enumerate(p) if v)
        rdir = os.path.join(A.WORK, 'runs', f"{row['name']}_prop{tag}")
        if os.path.exists(os.path.join(rdir, 'CrossSec-Y')) or os.path.exists(os.path.join(rdir, 'CrossSec-X')):
            out.append((tag, 'skipped (done)')); continue
        m = [s for v in str(row['m_values']).split() for s in (v, '0')]
        cmd = [A.ADDA_EXE, '-shape', 'read', os.path.join(A.WORK, row['geometry']), '-size', f"{row['size_um']:.8f}",
               '-lambda', f'{A.LAMBDA_MED_UM:.8f}', '-m', *m, '-prop', *map(str, p), '-no_vol_cor', '-dir', rdir]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        with open(rdir + '.stdout.txt', 'w') as fh:
            fh.write(' '.join(cmd) + '\n\n' + res.stdout + '\n' + res.stderr)
        out.append((tag, f'exit {res.returncode}, {time.time() - t0:.0f} s'))
    return out


def main(names=None):
    if not os.path.exists(A.ADDA_EXE):
        sys.exit(f'ADDA not found at {A.ADDA_EXE}')
    man = pd.read_csv(os.path.join(A.WORK, 'manifest.csv'))
    os.makedirs(os.path.join(A.WORK, 'runs'), exist_ok=True)
    for _, row in man.iterrows():
        if names and row['name'] not in names:
            continue
        print(row['name'], run_case(row), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
