"""Shared settings and loaders for the serial-dilution analysis (dilution_*.py).

Data: ShervinCode_Data/Dilution/*.fcs and FileInfo.csv (Sample, Concentration, Replicate); concentration is taken to be
the nominal origami concentration in nM at the sample port. Override the folder with NFCM_DILUTION_DIR.
Outputs: figures_svg/dilution/ (summary tables from dilution_01_summarize.py, then one SVG + CSV per figure).
"""
import os, re, json
import numpy as np, pandas as pd
from fcsparser import parse
import warnings; warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
MANUSCRIPT_DIR = os.path.abspath(os.path.join(HERE, '..'))
DIL = os.environ.get('NFCM_DILUTION_DIR') or os.path.join(MANUSCRIPT_DIR, '..', 'Dilution')
OUT = os.path.join(MANUSCRIPT_DIR, 'figures_svg', 'dilution'); os.makedirs(OUT, exist_ok=True)
PREVIEW = os.environ.get('FIG_PREVIEW_DIR')

SS, F, B, W = 'VSSC-1-H', 'V525-H', 'B525-H', 'VSSC-1-W'
SS_TRIG, B_TRIG = 200, 250                       # recorded trigger: VSSC-1 >= 200 OR B525 >= 250
AVOGADRO = 6.02214076e23
FLOW_UL_PER_MIN = 1.0                            # checked per file from the acquisition record (Volume / Duration)
TICK = 2.0 ** -21                                # Time channel resolution (s)


def nominal_flux(conc_nM, flow_ul_per_min=FLOW_UL_PER_MIN):
    """Particles per second passing the laser if every particle in the sample stream crosses it."""
    return np.asarray(conc_nM) * 1e-9 * AVOGADRO * flow_ul_per_min * 1e-6 / 60


def file_info():
    fi = pd.read_csv(os.path.join(DIL, 'FileInfo.csv'), index_col=0)
    fi.columns = [c.strip() for c in fi.columns]
    fi = fi.rename(columns={'Sample': 'sample', 'Concentration': 'conc_nM', 'Replicate': 'replicate'})
    fi['series'] = fi['sample'].str[0]
    buf = sorted(f[:-4] for f in os.listdir(DIL) if f.lower().startswith('buffer') and f.endswith('.fcs'))
    fi = pd.concat([fi, pd.DataFrame(dict(sample=buf, conc_nM=0.0, replicate=[int(re.sub(r'\D', '', b) or 0) for b in buf],
                                          series='buffer'))], ignore_index=True)
    return fi


def record(path):
    """Acquisition record (NANOFLOWINFO JSON in the ANALYSIS segment)."""
    meta = parse(path, meta_data_only=True, reformat_meta=True)
    raw = open(path, 'rb').read()[int(meta['$BEGINANALYSIS']):int(meta['$ENDANALYSIS']) + 1].decode('latin1')
    info = json.loads(bytes.fromhex(re.search(r'NANOFLOWINFO\n([0-9A-Fa-f]+)', raw).group(1)).decode('utf8'))
    a = info['Acquisition']
    ch = a['ExtraProperties']['Channels']; ch = json.loads(ch) if isinstance(ch, str) else ch
    return dict(start=a['StartTime'], end=a['EndTime'], duration_s=a['Duration'], volume_ul=a['Volume'],
                events_recorded=a['Events'], aborted=a['AbortedEvents'], trig_aborted=a['TrigAbortedEvents'],
                scan_aborted=a['ScanAbortedEvents'], flow_setting=a['FlowRateValue'], sequence=info['Tube']['Sequence'],
                tube_name=info['Tube']['TubeName'], primary=f"ch{a['PrimaryChannel']}>={a['PrimaryThreshold']}",
                secondary=f"ch{a['SecondaryChannel']}>={a['SecondaryThreshold']}", trig_logic=a['ThresholdTrigLogic'],
                voltages=';'.join(f"{c['ChannelName']}={c['Voltage']}" for c in ch),
                channel_trig=';'.join(f"{c['ChannelName']}={c['Trig']}" for c in ch))


def load(sample, cols=(SS, F, B, W, 'VSSC-1-A', 'Time')):
    _, df = parse(os.path.join(DIL, f'{sample}.fcs'), reformat_meta=True)
    return df[list(cols)].astype('float64')


def read(name):
    return pd.read_csv(os.path.join(OUT, name))


def save(fig, name, tables=None):
    fig.savefig(os.path.join(OUT, f'{name}.svg'), bbox_inches='tight')
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, f'{name}.png'), dpi=130, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    for suffix, df in (tables or {}).items():
        df.to_csv(os.path.join(OUT, f'{name}{suffix}.csv'), index=False)
    print(f'  wrote {name}.svg' + (f' (+{len(tables)} csv)' if tables else ''))


def style():
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'legend.frameon': False, 'lines.linewidth': 1.4})


INK, MUTED, RED = '#222222', '#6b6b6b', '#C0392B'
REP_COL = {1: '#3354D1', 2: '#2E9E4F', 3: '#F28E2B', 0: '#8C8C8C'}
REP_MK = {1: 'o', 2: 's', 3: '^', 0: 'D'}
