"""Figure data package for the 2026-11-09 main text and SI (nFCM_2026-11-09_main_text_JH.docx, nFCM_2026-11-09_SI_JH.docx).

One folder per figure and one subfolder per panel. Each data panel gets <Figure>_<panel>.svg (the plot redrawn from the
data), <Figure>_<panel>.csv (the plotted values, as they are; extra tables as <Figure>_<panel>_<name>.csv) and README.txt
(what is plotted, how it was computed, source script). Panels that are schematics, micrographs, gels or designs are cut from
the figure image in the .docx and saved as SVG (embedded PNG) without a CSV. Each figure folder has README.txt with the
caption from the .docx. PACKAGE/panel_index.csv lists every panel.

Data panels are computed from the FCS files (shared_utils.load_fcs; acquisitions above 7,000 events/s excluded) or read
from the tables written by the analysis scripts (make_manuscript_figures.py, fig29*_tccd*.py, fig5g_delta_U_matched_lengths.py,
fig37_proposed_flow_panels.py, adda/export_scattering_vs_coating.py, fig41_caption_stat_checks.py, dilution_01_summarize.py),
which must be run first
(code/README.md in the package lists the order). Batch 1 = streptavidin session (05/25), batch 2 = antibody session (04/18).

SI numbering: the dilution series was inserted as Figure S5 on 2026-11-21, so S5 to S17 of the 11/09 SI are S6 to S18 here.
Figure S5 is drawn by fig_s5_dilution_series.py (D18 without tube labels) and keeps all 36 tubes: it is the one exception to the 7,000
events/s exclusion.

Environment (optional): NFCM_PACKAGE_DIR (output folder), NFCM_MAIN_DOCX, NFCM_SI_DOCX (figure images).
Run:  python build_figure_package.py
"""
import sys, os, re, io, base64, zipfile, html, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MPLBACKEND', 'Agg')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from scipy import stats, optimize
import fig37_proposed_flow_panels as P          # sets the plot style
import fig_s4b_local_slope as S4B
import fig_s5_dilution_series as S5
import make_manuscript_figures as M
from shared_utils import acq_time

ROOT = os.path.dirname(M.MANUSCRIPT_DIR)                        # ShervinCode_Data
PKG = os.environ.get('NFCM_PACKAGE_DIR') or os.path.join(ROOT, 'nFCM_figure_data_2026-11-09')
DOCX = {'main': os.environ.get('NFCM_MAIN_DOCX') or os.path.join(ROOT, 'nFCM_2026-11-09_main_text_JH.docx'),
        'si': os.environ.get('NFCM_SI_DOCX') or os.path.join(ROOT, 'nFCM_2026-11-09_SI_JH.docx')}
FS = M.OUT                                                      # tables written by the analysis scripts
SS, F, B = 'VSSC-1-H', 'V525-H', 'B525-H'
TRIG, AB_T = 200, 250
B1, B2 = P.B1, P.B2
BATCH = {B1: 'batch 1', B2: 'batch 2'}
COL, INK, MUTED = P.COL, P.INK, P.MUTED
RED = '#C0392B'
LBL = {'NoCoating': 'bare', 'PLL': 'PLL-PEG', 'Silica': 'silica'}
CO = {'Uncoated': 'NoCoating', 'PLL-PEG': 'PLL', 'Silica': 'Silica'}
MK = {B1: '^', B2: 'o'}
INDEX = []
SI_INSERTED = 5                                                 # SI figure number added after the 11/09 draft
S5_CAPTION = ('Figure S5. Scatter-triggered event rate (VSSC-1-H >= 200) of serially diluted bare DNA origami as a function of '
              'nominal origami concentration, for three independent dilution chains (replicates 1 to 3) and for all three '
              'together. Solid lines are fits of R = R_buffer + kC^alpha to each chain by least squares '
              'on ln R; alpha is given as the slope with its 95% confidence interval. The dotted line is the mean rate of three '
              'buffer acquisitions. All acquisitions are shown, including five above 7,000 events/s (3.3 and 1.65 nM).')
rng = np.random.default_rng(7)
_cache = {}


# ------------------------------------------------------------------------------------------------ helpers
def L(folder, f):
    if (folder, f) not in _cache:
        _cache[(folder, f)] = M.L(folder, f)
    return _cache[(folder, f)]


def tube(sess, coat, rep, tag=None):
    s = M.SESSIONS[sess]
    return L(s['folder'], f"N_{coat}_{tag or s['noprobe']}_{rep}.fcs")


def buffer(sess):
    return L(M.SESSIONS[sess]['folder'], 'Buffer.fcs')


def read(name):
    return pd.read_csv(os.path.join(FS, name))


def jitter(n, w=0.08):
    return rng.uniform(-w, w, n)


def panel_dir(fig, p):
    d = os.path.join(PKG, f'Figure_{fig}', f'Panel_{p}')
    os.makedirs(d, exist_ok=True)
    return d


def write_panel(fig, p, title, how, source, fig_obj=None, tables=None, note='', image=None, svg_writer=None, script=None):
    """Save one panel folder: SVG (matplotlib figure, writer function or embedded image), CSV tables, script and README."""
    d = panel_dir(fig, p); stem = f'Figure_{fig}_{p}'
    svg = os.path.join(d, stem + '.svg')
    if svg_writer is not None:
        svg_writer(svg)
    elif fig_obj is not None:
        fig_obj.savefig(svg, bbox_inches='tight')
        if P.PREVIEW:
            fig_obj.savefig(os.path.join(P.PREVIEW, stem + '.png'), dpi=110, bbox_inches='tight')
        plt.close(fig_obj)
    else:
        buf = io.BytesIO(); image.save(buf, 'PNG')
        w, h = image.size
        with open(svg, 'w', encoding='utf8') as fh:
            fh.write(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" '
                     f'height="{h}" viewBox="0 0 {w} {h}"><image width="{w}" height="{h}" xlink:href="data:image/png;base64,'
                     f'{base64.b64encode(buf.getvalue()).decode()}"/></svg>')
    names = []
    for k, (name, df) in enumerate((tables or {}).items()):
        fn = stem + ('.csv' if k == 0 else f'_{name}.csv')
        df.to_csv(os.path.join(d, fn), index=False); names.append(fn)
    kind = 'data' if tables else 'image'
    if script:
        shutil.copyfile(script, os.path.join(d, stem + '.py'))
    txt = [f'Figure {fig}, panel {p}: {title}', '', 'Plotted: ' + how, '', 'Source: ' + source]
    if names:
        txt += ['', 'Files: ' + stem + '.svg, ' + ', '.join(names + ([stem + '.py'] if script else []))]
    else:
        txt += ['', f'Files: {stem}.svg (cut from the figure image in the 11/09 .docx; no plotted data)']
    if note:
        txt += ['', 'Note: ' + note]
    open(os.path.join(d, 'README.txt'), 'w', encoding='utf8').write('\n'.join(txt) + '\n')
    INDEX.append(dict(figure=fig, panel=p, title=title, kind=kind, svg=os.path.relpath(svg, PKG),
                      csv=';'.join(os.path.relpath(os.path.join(d, n), PKG) for n in names), source=source, note=note))
    print(f'  {stem} ({kind})')


# ------------------------------------------------------------------------------------------------ figure images
def si_number(text):
    """Figure numbers of the 11/09 SI (S5 to S17) as numbered in this package (S6 to S18)."""
    return re.sub(r'(?<![A-Za-z0-9])S(\d+)(?=[A-D]?(?![A-Za-z0-9]))',
                  lambda m: f'S{int(m.group(1)) + 1}' if int(m.group(1)) >= SI_INSERTED else m.group(0), text)


def docx_parts(which):
    z = zipfile.ZipFile(DOCX[which])
    x = z.read('word/document.xml').decode('utf8')
    rels = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', z.read('word/_rels/document.xml.rels').decode('utf8')))
    paras = re.findall(r'<w:p[ >].*?</w:p>', x[x.index('<w:body>'):], re.S)
    text = lambda p: ''.join(html.unescape(m) for m in re.findall(r'<w:t(?:\s[^>]*)?>([^<]*)</w:t>', re.sub(r'<w:del\b.*?</w:del>', '', p, flags=re.S)))
    images, captions, last = {}, {}, None
    for p in paras:
        ids = [rels[r] for r in re.findall(r'r:embed="(rId\d+)"', p) if rels.get(r, '').startswith('media/')]
        if ids:
            last = [i for i in ids if not i.endswith('.svg')][0]
        t = text(p).strip()
        m = re.match(r'(Figure (S?\d+))\.', t)
        if m and last and si_number(m.group(2)) not in captions:
            key = si_number(m.group(2)); captions[key] = si_number(t); images[key] = last; last = None
    return z, images, captions


def figure_image(z, name):
    im = Image.open(io.BytesIO(z.read('word/' + name)))
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA'); bg = Image.new('RGBA', im.size, 'white'); bg.alpha_composite(im); im = bg
    return im.convert('RGB')


# crop boxes as fractions of the figure image (left, top, right, bottom)
CROPS = {
    '1': {'A': (0, 0, .362, .46), 'B': (.362, 0, 1, .46), 'C': (0, .466, .442, 1), 'D': (.442, .466, 1, 1)},
    '2': {'A': (0, 0, .385, .494)},
    '3': {'A': (0, 0, .503, .352), 'B': (.518, 0, 1, .352), 'F': (0, .65, .335, 1)},
    '4': {'A': (0, 0, .325, .464)},
    '5': {'A': (0, 0, .35, .453)},
    'S1': {'A': (0, 0, 1, 1)}, 'S2': {'A': (0, 0, 1, 1)}, 'S6': {'A': (0, 0, 1, 1)}, 'S7': {'A': (0, 0, 1, 1)},
    'S3': {'A': (0, 0, .485, .351), 'B': (.485, 0, 1, .351), 'C': (0, .351, 1, 1)},
    'S8': {'A': (0, 0, 1, .497), 'B': (0, .497, 1, 1)},
    'S9': {'A': (0, 0, 1, .497), 'B': (0, .497, 1, 1)},
}
IMAGE_TITLES = {
    ('1', 'A'): 'DNA origami box (schematic)', ('1', 'B'): 'Surface coatings (schematic)',
    ('1', 'C'): 'Biotin display and IgG probe (schematic)', ('1', 'D'): 'Nano-flow cytometry (schematic)',
    ('2', 'A'): 'Negative-stain TEM of bare origami, raw field and class averages',
    ('3', 'A'): 'TEM class averages of bare, PLL-PEG- and silica-coated origami',
    ('3', 'B'): 'Intensity profiles across the side walls of the side-view class averages',
    ('3', 'F'): 'Threshold truncation (schematic)', ('4', 'A'): 'Reading scatter vs fluorescence (schematic)',
    ('5', 'A'): 'Two-color coincidence detection (schematic)', ('S1', 'A'): 'scadnano design',
    ('S2', 'A'): 'Agarose gel of origami folding', ('S3', 'A'): 'TEM of bare origami at three magnifications',
    ('S3', 'B'): 'Aggregation on the grid', ('S3', 'C'): 'Validation of the bare class averages',
    ('S6', 'A'): 'ATTO 425 spectra and the 525/50 band', ('S7', 'A'): 'PLL-PEG gel retardation assay',
    ('S8', 'A'): 'TEM of PLL-PEG-coated origami', ('S8', 'B'): 'Validation of the PLL-PEG class averages',
    ('S9', 'A'): 'TEM of silica-coated origami', ('S9', 'B'): 'Validation of the silica class averages',
}
IMAGE_NOTES = {
    ('3', 'B'): 'The wall profiles and the FWHM / cavity / spacing table come from the class-average analysis (Methods, '
                'TEM imaging), which is not among the files used here; ask for the profile table to add a CSV.',
    ('S3', 'B'): 'Box counts (1,112 boxes, six micrographs) come from the TEM picking analysis, not among the files used here.',
    ('S6', 'A'): 'Spectra from fpbase.org; the spectra table is not among the files used here.',
}


def image_panels(captions_out):
    for which in ['main', 'si']:
        z, images, captions = docx_parts(which)
        captions_out.update(captions)
        for fig, boxes in CROPS.items():
            if fig not in images:
                continue
            im = figure_image(z, images[fig]); w, h = im.size
            for p, (l, t, r, b) in boxes.items():
                crop = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
                kind = 'TEM / gel / design image' if fig in ('2', '3', 'S1', 'S2', 'S3', 'S7', 'S8', 'S9') and (fig, p) != ('3', 'F') else 'schematic'
                write_panel(fig, p, IMAGE_TITLES[(fig, p)], f'{kind}, cut from {os.path.basename(DOCX[which])} ({images[fig]})',
                            'figure image in the .docx', image=crop, note=IMAGE_NOTES.get((fig, p), ''))


# ------------------------------------------------------------------------------------------------ Figure 2
def fig2():
    # B  density of scatter-triggered events, buffer and bare origami (batch 2, replicate B)
    samples = [('Buffer', buffer(B2), 'Shervin4_18/Buffer.fcs'), ('Bare origami', tube(B2, 'NoCoating', 'B'), 'Shervin4_18/N_NoCoating_NoAB_B.fcs')]
    trig = [(lab, M.scat(d), d, fn) for lab, d, fn in samples]
    xb = P.span_edges([z[SS] for _, z, _, _ in trig], 2); xb = np.logspace(2, np.log10(xb[-1]), 101)
    yb = P.span_edges([z[F] for _, z, _, _ in trig], 101)
    rows, summ, hs = [], [], []
    for lab, z, d, fn in trig:
        t = acq_time(d); ok = z[F] > 0
        h = np.histogram2d(z[SS][ok], z[F][ok], [xb, yb])[0] / t; hs.append(h)
        rows += [dict(sample=lab, file=fn, ss_bin_lo=xb[i], ss_bin_hi=xb[i + 1], v525_bin_lo=yb[j], v525_bin_hi=yb[j + 1],
                      events_per_s=h[i, j]) for i, j in zip(*np.nonzero(h))]
        summ.append(dict(sample=lab, file=fn, acq_time_s=t, scatter_triggered_events=len(z), scatter_triggered_events_per_s=len(z) / t,
                         events_with_V525_le_0_not_plotted=int((~ok).sum())))
    vmin = min(h[h > 0].min() for h in hs); vmax = max(h.max() for h in hs)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
    for ax, (lab, *_), h, sm in zip(axes, trig, hs, summ):
        pc = ax.pcolormesh(xb, yb, np.where(h > 0, h, np.nan).T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='Blues', rasterized=True)
        ax.axvline(TRIG, color=RED, ls='--', lw=1); ax.text(TRIG * 1.1, yb[-1] / 4, 'trigger', color=RED, fontsize=7)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e2, xb[-1]); ax.set_ylim(yb[0], yb[-1])
        ax.set_xlabel('VSSC-1-H (side scatter)'); ax.set_title(f"{lab}: {sm['scatter_triggered_events_per_s']:,.0f} events/s", fontsize=8)
    axes[0].set_ylabel('V525-H (ATTO 425)')
    fig.colorbar(pc, ax=axes, pad=0.02, fraction=0.05).set_label('events/s per bin', fontsize=8)
    write_panel('2', 'B', 'Side scatter vs ATTO 425, scatter-triggered events (batch 2)',
                'events per second in 100 x 100 log bins of VSSC-1-H and V525-H, scatter-triggered events (VSSC-1-H >= 200) of '
                'the batch 2 buffer and bare origami replicate B, no probe; events with V525-H <= 0 cannot be shown on log axes',
                'build_figure_package.py fig2 (same as fig37_proposed_flow_panels.py panel_2b)', fig,
                {'bins': pd.DataFrame(rows), 'summary': pd.DataFrame(summ)})

    # C  recorded side scatter of bare origami, batch 2 (three tubes pooled); inset: fraction >= x
    reps = [(r, M.scat(tube(B2, 'NoCoating', r))[SS].values) for r in 'ABC']
    pool = np.concatenate([v for _, v in reps]); e = np.arange(200, 661, 20)
    H = pd.DataFrame(dict(bin_lo=e[:-1], bin_hi=e[1:], pct_of_events_pooled=np.histogram(pool, e)[0] / len(pool) * 100))
    for r, v in reps:
        H[f'pct_of_events_replicate_{r}'] = np.histogram(v, e)[0] / len(v) * 100
    xs = np.arange(210, 491, 20.0); fr = np.array([(pool >= x).mean() for x in xs])
    I = pd.DataFrame(dict(x=xs, fraction_ge_x=fr, guide_x_pow_minus_3p7=fr[0] * (xs / xs[0]) ** -3.7))
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.bar(H.bin_lo, H.pct_of_events_pooled, width=19, align='edge', color=COL['NoCoating'], alpha=0.85)
    ax.axvline(TRIG, color=RED, ls='--', lw=1); ax.text(TRIG + 5, H.pct_of_events_pooled.max() * 1.02, 'trigger (200)', color=RED, fontsize=7)
    ax.set_xlim(150, 650); ax.set_xlabel('VSSC-1-H (side scatter)'); ax.set_ylabel('Recorded events (%)')
    ins = ax.inset_axes([0.45, 0.35, 0.5, 0.55])
    ins.plot(I.x, I.fraction_ge_x, 'o', ms=3, color=COL['NoCoating']); ins.plot(I.x[:6], I.guide_x_pow_minus_3p7[:6], 'k--', lw=0.8)
    ins.set_yscale('log'); ins.set_xlim(200, 500); ins.set_title('fraction ≥ x', fontsize=7); ins.tick_params(labelsize=6)
    ins.text(330, fr[1], r'$\propto x^{-3.7}$', fontsize=7)
    write_panel('2', 'C', 'Recorded side scatter, bare origami (batch 2)',
                'histogram of VSSC-1-H of scatter-triggered events of the three batch 2 bare-origami tubes without probe (pooled), '
                'bins of 20 from the trigger (200), as % of all scatter-triggered events; inset: fraction of events >= x and the '
                'x^-3.7 guide of the manuscript', 'build_figure_package.py fig2', fig, {'histogram': H, 'inset': I},
                note='The guide exponent 3.7 is the value in the 11/09 text; refitting on the raw events gives 4.4 (batch 2) to '
                     '5.6 (batch 1) at the threshold (checks_11_08/fig41d).')

    # D  scatter-triggered event rate, buffer and bare origami without probe (raw)
    rows = []
    for sess in [B1, B2]:
        d = buffer(sess); t = acq_time(d)
        rows.append(dict(batch=BATCH[sess], sample='Buffer', replicate='', scatter_events_per_s=(d[SS] >= TRIG).sum() / t))
        for r in 'ABC':
            d = tube(sess, 'NoCoating', r); t = acq_time(d)
            rows.append(dict(batch=BATCH[sess], sample='Origami', replicate=r, scatter_events_per_s=(d[SS] >= TRIG).sum() / t))
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(2.4, 2.7))
    for i, (s, c) in enumerate([('Buffer', COL['Buffer']), ('Origami', COL['NoCoating'])]):
        v = D[D['sample'] == s].scatter_events_per_s.values
        ax.bar(i, v.mean(), 0.7, color=c, alpha=0.3); ax.errorbar(i, v.mean(), v.std(ddof=1), color=INK, capsize=3, lw=1)
        ax.scatter(i + jitter(len(v)), v, s=18, color=c, edgecolor='white', lw=0.5, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Buffer', 'Origami']); ax.set_ylabel('Scatter-triggered events/s')
    write_panel('2', 'D', 'Event rate, buffer and bare origami without probe',
                'scatter-triggered events per second (VSSC-1-H >= 200, raw), one point per acquisition: buffer of each batch '
                '(n = 2) and bare origami without probe (n = 6, batches 1 and 2); bar mean, error bar SD',
                'build_figure_package.py fig2 (same values as fig01_event_rate_origami_vs_buffer.csv)', fig, {'rates': D})

    # E  V525-H of scatter-triggered events, batch 2
    e = np.arange(-5, 206, 10); c = (e[:-1] + e[1:]) / 2
    rows = []
    for lab, v in [('Buffer', M.scat(buffer(B2))[F].values),
                   ('Bare origami', np.concatenate([M.scat(tube(B2, 'NoCoating', r))[F].values for r in 'ABC']))]:
        h = np.histogram(v, e)[0]
        rows += [dict(sample=lab, bin_center=cc, bin_lo=a, bin_hi=b, events=n, pct_of_events_in_range=100 * n / h.sum())
                 for cc, a, b, n in zip(c, e[:-1], e[1:], h)]
    E = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for lab, colr, mk in [('Bare origami', COL['NoCoating'], 's'), ('Buffer', COL['Buffer'], 'o')]:
        z = E[E['sample'] == lab]; ax.plot(z.bin_center, z.pct_of_events_in_range, marker=mk, ms=3.5, color=colr, label=lab)
    ax.set_xlim(0, 200); ax.set_xlabel('V525-H (ATTO 425)'); ax.set_ylabel('Recorded events (%)'); ax.legend()
    write_panel('2', 'E', 'ATTO 425 signal of recorded events (batch 2)',
                'histogram of V525-H of scatter-triggered events, bins of 10 centred on 0, 10, ..., 200, as % of the events in '
                'that range; buffer and the three bare-origami tubes without probe (pooled), batch 2',
                'build_figure_package.py fig2', fig, {'histograms': E})


# ------------------------------------------------------------------------------------------------ Figure 3
def tubes_table():
    rows = []
    for sess in [B1, B2]:
        b = buffer(sess); bt = acq_time(b); bss = (b[SS] >= TRIG).sum() / bt
        rows.append(dict(batch=BATCH[sess], coating='Buffer', replicate='', file=f"{M.SESSIONS[sess]['folder']}/Buffer.fcs",
                         median_VSSC1H=float(M.scat(b)[SS].median()), scatter_events_per_s=bss, scatter_events_per_s_minus_buffer=np.nan))
        for c in ['NoCoating', 'PLL', 'Silica']:
            for r in 'ABC':
                d = tube(sess, c, r); t = acq_time(d); z = M.scat(d)
                rows.append(dict(batch=BATCH[sess], coating=LBL[c], replicate=r, file=f"{M.SESSIONS[sess]['folder']}/N_{c}_{M.SESSIONS[sess]['noprobe']}_{r}.fcs",
                                 median_VSSC1H=float(z[SS].median()), scatter_events_per_s=len(z) / t,
                                 scatter_events_per_s_minus_buffer=len(z) / t - bss))
    return pd.DataFrame(rows)


def coat_bars(ax, T, col, groups=('bare', 'PLL-PEG', 'silica')):
    for i, c in enumerate(groups):
        for k, sess in enumerate([B1, B2]):
            v = T[(T.coating == c) & (T.batch == BATCH[sess])][col].values; x = i + (k - 0.5) * 0.36
            colr = COL[{'bare': 'NoCoating', 'PLL-PEG': 'PLL', 'silica': 'Silica'}[c]]
            ax.bar(x, v.mean(), 0.32, color=colr, alpha=0.3)
            ax.scatter(x + jitter(len(v), 0.05), v, s=16, marker=MK[sess], color=colr, edgecolor='white', lw=0.4, zorder=3)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(['Bare', 'PLL-PEG', 'Silica'][:len(groups)])
    ax.legend(handles=[plt.Line2D([], [], ls='', marker=MK[s], color=INK, label=BATCH[s]) for s in [B1, B2]], ncol=2, loc='upper left')


def fig3():
    # C, D  ADDA simulations
    Pp = read('scattering_vs_pllpeg_coverage.csv'); Si = read('scattering_vs_silica_thickness.csv')
    Si = Si[Si.coating_geometry == 'all exposed surfaces']
    for p, df, x, y, a_cap, colr, xlab, title in [
            ('C', Pp, 'fraction_phosphates_bound', 'I_ratio_ADDA', 3.49, COL['PLL'], 'Fraction of phosphates bound, f', 'Simulated scatter, PLL-PEG'),
            ('D', Si, 'thickness_nm', 'I_ratio_ADDA_n1.45', 0.561, COL['Silica'], 'Silica thickness, t (nm)', 'Simulated scatter, silica')]:
        X, Y = df[x].values, df[y].values
        a_fit = optimize.curve_fit(lambda t, a: (1 + a * t) ** 2, X, Y, p0=[1])[0][0]
        xx = np.linspace(0, X.max(), 101)
        Cv = pd.DataFrame({x: xx, 'caption_curve': (1 + a_cap * xx) ** 2, 'least_squares_curve': (1 + a_fit * xx) ** 2})
        fig, ax = plt.subplots(figsize=(3.2, 2.7))
        ax.plot(X, Y, 'o', ms=4, color=colr, label='ADDA'); ax.plot(xx, Cv.caption_curve, '--', color=colr, lw=1, label=f'(1 + {a_cap} {x[0]})²')
        ax.axhline(1, color=MUTED, ls=':', lw=0.8); ax.set_xlabel(xlab); ax.set_ylabel('Scatter / bare'); ax.set_ylim(0, None); ax.legend()
        write_panel('3', p, title, f'ADDA ratio of the scattering cross-section of the coated to the bare box (405 nm, water, '
                    f'orientation averaged; points) and the dashed curve of the manuscript caption, (1 + {a_cap} {x[0]})²',
                    'adda/collect_adda.py -> adda/export_scattering_vs_coating.py (figures_svg/' +
                    ('scattering_vs_pllpeg_coverage.csv)' if p == 'C' else 'scattering_vs_silica_thickness.csv, all exposed surfaces, n = 1.45)'),
                    fig, {'adda': df[[x, y] + [c for c in df.columns if c not in (x, y)]], 'curves': Cv},
                    note=f'A least-squares fit of (1 + a {x[0]})² to these points gives a = {a_fit:.3f}, not {a_cap}; the caption '
                         'coefficient could not be reproduced from the ADDA table.')
    # E, G, H  medians and event rates of the no-probe tubes
    T = tubes_table(); Tc = T[T.coating != 'Buffer']
    fig, ax = plt.subplots(figsize=(3.3, 2.7)); coat_bars(ax, Tc, 'median_VSSC1H'); ax.set_ylabel('Median VSSC-1-H'); ax.set_ylim(0, 700)
    write_panel('3', 'E', 'Median side scatter', 'median VSSC-1-H of the scatter-triggered events of each no-probe tube (raw); bars '
                'show the mean of the three tubes of each batch', 'build_figure_package.py fig3', fig, {'tubes': T})
    fig, ax = plt.subplots(figsize=(3.3, 2.7)); coat_bars(ax, Tc, 'scatter_events_per_s_minus_buffer')
    ax.set_yscale('log'); ax.set_ylim(100, 1e4); ax.set_ylabel('Scatter-triggered events/s')
    write_panel('3', 'G', 'Event rate', 'scatter-triggered events per second of each no-probe tube minus the buffer of the same '
                'session (33.75 events/s in batch 1, 59.37 events/s in batch 2); bars show the mean of the three tubes',
                'build_figure_package.py fig3 (same values as checks_11_08/fig41a and the Prism table of Figure 3G)', fig, {'tubes': T})
    mean = Tc.groupby(['batch', 'coating']).scatter_events_per_s_minus_buffer.mean()
    R = pd.DataFrame([dict(batch=b, coating=c, mean_rate_coated=mean[(b, c)], mean_rate_bare=mean[(b, 'bare')],
                           ratio_coated_to_bare=mean[(b, c)] / mean[(b, 'bare')],
                           scatter_gain_k_from_R_eq_k_pow_3p7=(mean[(b, c)] / mean[(b, 'bare')]) ** (1 / 3.7))
                      for b in ['batch 1', 'batch 2'] for c in ['PLL-PEG', 'silica']])
    k = np.linspace(1, 2.5, 151); Cv = pd.DataFrame(dict(k=k, R_eq_k_pow_3p7=k ** 3.7))
    kmax = 2 ** (1 / 3.7)
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.axhspan(0.5, 2, color='#E5E8EC'); ax.plot(k, k ** 3.7, color=INK, lw=1.3)
    for r in R.itertuples():
        ax.plot(1.02, r.ratio_coated_to_bare, marker='^' if r.batch == 'batch 1' else 'o', ls='', ms=5,
                color=COL['PLL'] if r.coating == 'PLL-PEG' else COL['Silica'])
    ax.axvline(kmax, ymax=0.5, color=INK, ls=':', lw=0.8); ax.text(kmax + 0.02, 0.62, f'k ≤ {kmax:.2f}', fontsize=7)
    s1 = R[(R.batch == 'batch 1') & (R.coating == 'silica')].iloc[0]
    ax.plot([1, s1.scatter_gain_k_from_R_eq_k_pow_3p7], [s1.ratio_coated_to_bare] * 2, color=COL['Silica'], ls='--', lw=0.8)
    ax.text(s1.scatter_gain_k_from_R_eq_k_pow_3p7, s1.ratio_coated_to_bare * 1.15, f'k ≈ {s1.scatter_gain_k_from_R_eq_k_pow_3p7:.1f}',
            color=COL['Silica'], fontsize=7, ha='center')
    ax.set_yscale('log'); ax.set_ylim(0.4, 40); ax.set_xlim(1, 2.5); ax.set_xlabel('Scatter gain per particle, k')
    ax.set_ylabel('Event rate, coated / bare'); ax.text(2.0, 4.5, 'R = k$^{3.7}$', fontsize=8)
    write_panel('3', 'H', 'Scatter gain allowed by the event rate',
                'ratio of the mean buffer-subtracted event rate (Figure 3G) of coated to bare origami for each batch (symbols at '
                'k = 1), the curve R = k^3.7 and the band of ratios within twofold of bare (0.5 to 2), which limits k to 2^(1/3.7)',
                'build_figure_package.py fig3', fig, {'observed_ratios': R, 'curve': Cv},
                note='Uses the exponent 3.7 of the 11/09 text; see Figure 2C note.')


# ------------------------------------------------------------------------------------------------ Figure 4
def dot_panel(fig_id, p, d, title, how, guides=False):
    z = M.scat(d); ok = z[F] > 0
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.scatter(z[SS][ok], z[F][ok], s=0.6, color=COL['NoCoating'], alpha=0.25, lw=0, rasterized=True)
    ax.axvline(TRIG, color=RED, ls='--', lw=1)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(10, 1.2e6); ax.set_ylim(0.1, 1e6)
    ax.set_xlabel('VSSC-1-H (side scatter)'); ax.set_ylabel('V525-H (ATTO 425)')
    tables = {'events': z[[SS, F]].reset_index(drop=True)}
    if guides:
        x0 = 1000.0; y0 = float(z[(z[SS] >= 800) & (z[SS] < 1250)][F].median()); xs = np.logspace(3, 5, 21)
        G = pd.DataFrame(dict(VSSC1H=xs, slope_1=y0 * xs / x0, slope_half=y0 * (xs / x0) ** 0.5))
        ax.plot(G.VSSC1H, G.slope_1, color=INK, lw=1.2); ax.plot(G.VSSC1H, G.slope_half, color=INK, lw=1.2, ls='--')
        ax.text(xs[-1], G.slope_1.iloc[-1] * 1.4, 'slope 1', fontsize=7, ha='right'); ax.text(xs[-1], G.slope_half.iloc[-1] / 2.2, 'slope ½', fontsize=7, ha='right')
        tables['guides'] = G
    write_panel(fig_id, p, title, how + f'; {int((~ok).sum())} events with V525-H <= 0 are in the CSV but cannot be shown on log axes',
                'build_figure_package.py fig4', fig, tables)


def fig4():
    dot_panel('4', 'B', tube(B2, 'Silica', 'A'), 'Silica-coated origami (batch 2, replicate A)',
              'VSSC-1-H vs V525-H of every scatter-triggered event of Shervin4_18/N_Silica_NoAB_A.fcs (no probe)')
    dot_panel('4', 'C', tube(B2, 'PLL', 'C'), 'PLL-PEG-coated origami (batch 2, replicate C)',
              'VSSC-1-H vs V525-H of every scatter-triggered event of Shervin4_18/N_PLL_NoAB_C.fcs (no probe), with guides of '
              'slope 1 and 1/2 through the median V525-H of events at VSSC-1-H 800 to 1,250', guides=True)
    S = read('fig06_silica_scatter_vs_v525_median_v525_by_ss.csv')
    Pl = read(os.path.join('pll_peg_side_scatter', 'fig21_pll_side_scatter_confident_origami_median_v525_by_ss.csv'))
    name = {'Uncoated': 'Bare', 'Uncoated origami (buffer-subtracted)': 'Bare', 'Silica': 'Silica-coated',
            'Silica chemicals only (raw)': 'Silica precursors only', 'PLL-PEG origami (buffer-subtracted)': 'PLL-PEG-coated',
            'PLL-PEG only (raw)': 'PLL-PEG only', 'Buffer (raw)': 'Buffer'}
    cl = {'Bare': COL['NoCoating'], 'Silica-coated': COL['Silica'], 'PLL-PEG-coated': COL['PLL'], 'Silica precursors only': '#8E6BB0',
          'PLL-PEG only': '#8E6BB0', 'Buffer': COL['Buffer']}
    for p, sess, coat in [('D', B1, 'Silica'), ('E', B2, 'Silica'), ('F', B1, 'PLL'), ('G', B2, 'PLL')]:
        if coat == 'Silica':
            z = pd.concat([S[S.session == sess], Pl[(Pl.session == sess) & (Pl['sample'] == 'Buffer (raw)')]])
        else:
            z = Pl[Pl.session == sess]
        z = z.assign(curve=z['sample'].map(name), batch=BATCH[sess])
        fig, ax = plt.subplots(figsize=(2.8, 2.6))
        for c, q in z.groupby('curve', sort=False):
            ax.plot(q.ss_bin_center, q.median_V525, color=cl[c], lw=1.4, ls='--' if c == 'Buffer' else '-', label=c)
        ax.set_xscale('log'); ax.set_xlim(150, 1e5); ax.set_ylim(0, 350); ax.set_xlabel('VSSC-1-H bin'); ax.set_ylabel('Median V525-H')
        ax.legend(fontsize=6.5, loc='upper left')
        write_panel('4', p, f"{'Silica' if coat == 'Silica' else 'PLL-PEG'}, {BATCH[sess]}",
                    'median V525-H of scatter-triggered events in 14 log bins of VSSC-1-H (200 to 100,000) for bare and coated '
                    'origami without probe (three tubes pooled, buffer-subtracted), the coating reagent without origami (02/13, raw) '
                    'and the session buffer (raw)', 'make_manuscript_figures.py fig06 (silica) and fig21 (PLL-PEG; buffer curve)',
                    fig, {'medians': z[['batch', 'curve', 'sample', 'ss_bin_center', 'median_V525']]})


# ------------------------------------------------------------------------------------------------ Figure 5
def stem_panel(fig_id, p, smp, col, ylab, title, how):
    fig, ax = plt.subplots(figsize=(3.3, 2.7)); rows = []
    for k, (c, mk) in enumerate([('Uncoated', 'o'), ('PLL-PEG', 's')]):
        z = smp[smp.coating == c]; xs, ms, sd = [], [], []
        for i, e in enumerate(['3', '6', '12']):
            v = z[z.extension.astype(str) == e][col].dropna().values
            rows.append(dict(coating=c, stem_bp=e, n=len(v), mean=v.mean(), sd=v.std(ddof=1)))
            xs.append(i + (k - 0.5) * 0.12); ms.append(v.mean()); sd.append(v.std(ddof=1))
        colr = COL[CO[c]]
        ax.errorbar(xs, ms, sd, marker=mk, ms=5, color=colr, capsize=3, lw=1, label='bare' if c == 'Uncoated' else c)
    ax.set_xticks(range(3)); ax.set_xticklabels(['3', '6', '12']); ax.set_xlabel('Stem length (bp)'); ax.set_ylabel(ylab)
    ax.set_ylim(0, None); ax.legend(loc='upper left')
    write_panel(fig_id, p, title, how, 'fig29_tccd_coincidence.py -> fig29d_tccd_ddG.py (figures_svg/fig29d_tccd_ddG_samples.csv)',
                fig, {'summary': pd.DataFrame(rows), 'samples': smp[['coating', 'replicate', 'extension', 'sample', col]]})


def fig5():
    S = read(os.path.join('proposed_panels', 'fig5b_origami_vs_free_antibody_signal_samples.csv'))
    lr = stats.linregress(S.low_SS_median_B525, S.gated_median_B525)
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    for r in S.itertuples():
        colr = COL[CO[r.coating]]
        ax.scatter(r.low_SS_median_B525, r.gated_median_B525, s=22, marker='s' if r.biotin == 'no biotin' else 'o',
                   facecolor='white' if r.biotin == 'no biotin' else colr, edgecolor=colr, lw=1, zorder=3)
    xx = np.array([S.low_SS_median_B525.min() - 1, S.low_SS_median_B525.max() + 1])
    ax.plot(xx, lr.intercept + lr.slope * xx, 'k--', lw=0.8); ax.text(0.65, 0.12, f'r = {lr.rvalue:.2f}', transform=ax.transAxes)
    ax.set_xlabel('Median B525-H, low-scatter events'); ax.set_ylabel('Median B525-H, origami events')
    ax.legend(handles=[plt.Line2D([], [], ls='', marker='s', mfc='white', mec=INK, label='no biotin'),
                       plt.Line2D([], [], ls='', marker='o', color=INK, label='biotin'),
                       plt.Line2D([], [], ls='', marker='o', color=COL['NoCoating'], label='bare'),
                       plt.Line2D([], [], ls='', marker='o', color=COL['PLL'], label='PLL-PEG')], ncol=2, fontsize=6.5, loc='upper left')
    write_panel('5', 'B', 'Median signal tracks free antibody',
                'per antibody acquisition (bare and PLL-PEG, all stems and no biotin): median B525-H of origami-gate events vs that '
                'of low-scatter (VSSC-1-H < 50) events with B525-H >= 250; dashed line, least-squares fit',
                'fig37_proposed_flow_panels.py panel_5b (proposed_panels/fig5b_origami_vs_free_antibody_signal_samples.csv)', fig,
                {'samples': S, 'fit': pd.DataFrame([dict(slope=lr.slope, intercept=lr.intercept, pearson_r=lr.rvalue, p=lr.pvalue, n=len(S))])})

    d = L('probe_only_controls', 'AB_Only__Shervin03_13_2026.fcs'); v = d[B].values
    e = np.arange(250, 601, 20); h = np.histogram(v, e)[0]
    Hh = pd.DataFrame(dict(B525_bin_lo=e[:-1], B525_bin_hi=e[1:], events=h, events_per_s=h / acq_time(d)))
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.bar(Hh.B525_bin_lo, Hh.events, width=19, align='edge', color=COL['Silica'])
    ax.axvline(AB_T, color=RED, ls='--', lw=1); ax.text(AB_T + 6, h.max(), 'trigger (250)', color=RED, fontsize=7)
    ax.set_xlim(0, 600); ax.set_xlabel('B525-H (AF488), antibody only'); ax.set_ylabel('Events')
    write_panel('5', 'C', 'Single antibodies sit at the trigger',
                f'histogram (bins of 20 from 250) of B525-H of all recorded events with B525-H >= 250 in the antibody-only control '
                f'(probe_only_controls/AB_Only__Shervin03_13_2026.fcs, {acq_time(d):.0f} s)', 'build_figure_package.py fig5', fig, {'histogram': Hh},
                note='The bars of the 11/09 panel are narrower than 20 (first bar about 20,000 events); the bin width used there is not '
                     'recorded. Bins of 10 are in figures_svg/antibody_only_B525_histogram.csv.')

    smp = read('fig29d_tccd_ddG_samples.csv'); smp = smp[smp.coating != 'Silica']
    stem_panel('5', 'D', smp, 'log2_fold_change_simple', 'log$_2$ F', 'Coincidences per antibody',
               'log2 fold change F of the observed-to-chance ratio of each biotin sample relative to its matched no-biotin sample; '
               'mean ± SD of the technical replicates (n = 3, n = 2 for PLL-PEG with the 12 bp stem)')
    stem_panel('5', 'E', smp, 'excess_simple', 'Coincidence excess, ε (% of origami)', 'Excess origami with an antibody',
               'biotin-specific coincidence excess ε (Equation 4) of each biotin sample; mean ± SD of the technical replicates')

    U = read('fig5g_delta_U_matched_lengths.csv')
    rows = []
    for m, lab in [('From log2FC', 'per antibody'), ('From excess', 'per origami')]:
        r = U[U.method == m].iloc[0]
        for s in 'ABC':
            rows.append(dict(estimate=lab, method=m, replicate_set=s, delta_U_kT=r[f'set_{s}_kT'], stems_bp=r[f'set_{s}_spacer_lengths_nt']))
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(2.8, 2.7))
    for i, lab in enumerate(['per antibody', 'per origami']):
        v = D[D.estimate == lab].delta_U_kT.values
        ax.bar(i, v.mean(), 0.75, color=COL['PLL'], alpha=0.35); ax.errorbar(i, v.mean(), v.std(ddof=1), color=INK, capsize=3, lw=1)
        ax.scatter(i + jitter(3), v, s=18, color=COL['PLL'], edgecolor='white', lw=0.5, zorder=3)
        ax.text(i, v.max() + 0.12, f'{v.mean():.3f} ± {v.std(ddof=1):.3f} kT', ha='center', fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['per antibody', 'per origami']); ax.set_ylabel('ΔU (kT)'); ax.set_ylim(0, 1.45)
    write_panel('5', 'F', 'Apparent coating penalty',
                'ΔU of the PLL-PEG coating for each technical replicate set from the per-antibody fold change (Equation 6) and the '
                'per-origami excess (Equation 5), using the stem lengths measured for both coatings; bar mean, error bar SD',
                'fig5g_delta_U_matched_lengths.py (figures_svg/fig5g_delta_U_matched_lengths.csv)', fig, {'replicate_sets': D, 'summary': U})


# ------------------------------------------------------------------------------------------------ SI figures
def fig_s5():
    A = pd.read_csv(os.path.join(FS, 'dilution', 'dilution_files.csv'), float_precision='round_trip')
    A = A.assign(file='Dilution/' + A['sample'] + '.fcs', above_7000_events_per_s=A.events_per_s > 7000)
    A = A[['sample', 'file', 'conc_nM', 'replicate', 'acq_order', 'acq_time_s', 'events', 'events_per_s', 'ss_events',
           'ss_events_per_s', 'above_7000_events_per_s']]
    buf, T = S5.analyze(pd.read_csv(io.StringIO(A.to_csv(index=False))))         # parsed as by Figure_S5_A.py
    write_panel('S5', 'A', 'Serial dilution: event rate of each replicate series with its fit',
                'scatter-triggered events/s (raw) against nominal concentration for each of the three dilution chains and for '
                'all three together, with fits R = buffer + k C^alpha (least squares on ln R, buffer fixed at the mean of three '
                'acquisitions); this is D18 without the tube labels. To redraw it, run python Figure_S5_A.py in this folder',
                'dilution_01_summarize.py -> Figure_S5_A.py (fig_s5_dilution_series.py in code/scripts)',
                svg_writer=lambda path: S5.draw(buf, T, path), script=S5.__file__,
                tables={**T, 'acquisitions': A},
                note='All 36 tubes are kept, by the decision of 2026-11-21, so this figure is exempt from the 7,000 events/s '
                     'exclusion. a1, a2 (3.3 nM), a3, b1 and b2 (1.65 nM) exceed it; they are marked in _acquisitions.csv.')


def si():
    T = tubes_table()
    z = T[T.coating.isin(['Buffer', 'bare'])]
    fig, ax = plt.subplots(figsize=(2.6, 2.7))
    for i, (c, colr) in enumerate([('Buffer', COL['Buffer']), ('bare', COL['NoCoating'])]):
        v = z[z.coating == c].median_VSSC1H.values
        ax.bar(i, v.mean(), 0.7, color=colr, alpha=0.3); ax.errorbar(i, v.mean(), v.std(ddof=1), color=INK, capsize=3, lw=1)
        ax.scatter(i + jitter(len(v)), v, s=18, color=colr, edgecolor='white', lw=0.5, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Buffer', 'Bare origami']); ax.set_ylabel('Median VSSC-1-H')
    write_panel('S4', 'A', 'Median side scatter of bare origami', 'median VSSC-1-H of scatter-triggered events (raw) of the buffer '
                '(n = 2) and the bare-origami tubes without probe (n = 6, batches 1 and 2); bar mean, error bar SD',
                'build_figure_package.py si (same as fig02_median_ss_origami_vs_buffer.csv, column median_SS)', fig,
                {'acquisitions': z[['batch', 'coating', 'replicate', 'file', 'median_VSSC1H']]})
    S4B.write(write_panel)
    fig_s5()

    Q = read(os.path.join('silica_side_scatter', 'fig22_scatter_fluorescence_quadrants.csv'))
    Q['quadrant'] = np.where(Q.fluorescence == 'High', 'F+', 'F−') + ' ' + np.where(Q.scatter == 'High', 'SS+', 'SS−')
    qc = {'F+ SS−': '#9BC2E6', 'F+ SS+': '#E7A6CF', 'F− SS−': '#F4C17A', 'F− SS+': '#A8D5A2'}
    for p, coated in [('A', 'Silica origami'), ('B', 'PLL-PEG origami')]:
        z = Q[Q['sample'].isin(['Uncoated origami', coated])]
        fig, ax = plt.subplots(figsize=(3.0, 2.7)); xt = []
        for i, (day, smpl) in enumerate([(d_, s_) for d_ in ['Day 1 (05/25)', 'Day 2 (04/18)'] for s_ in ['Uncoated origami', coated]]):
            q = z[(z.day == day) & (z['sample'] == smpl)].set_index('quadrant').pct_of_events; bottom = 0
            for qd in ['F− SS+', 'F+ SS+', 'F− SS−', 'F+ SS−']:
                ax.bar(i, q.get(qd, 0), 0.7, bottom=bottom, color=qc[qd], label=qd if i == 0 else None); bottom += q.get(qd, 0)
            xt.append(f"{'bare' if smpl.startswith('Unc') else coated.split()[0]}\n{'batch 1' if '05/25' in day else 'batch 2'}")
        ax.set_xticks(range(4)); ax.set_xticklabels(xt, fontsize=7); ax.set_ylabel('% of events'); ax.legend(fontsize=6.5, bbox_to_anchor=(1, 1), loc='upper left')
        write_panel('S10', p, f"Quadrants, bare and {coated.split()[0]}-coated origami",
                    'percentage of scatter-triggered events in each scatter-fluorescence quadrant (SS+ above the bare-origami 95th '
                    'percentile of VSSC-1-H, F+ above the bare-origami median V525-H of the batch), buffer-subtracted, tubes pooled',
                    'make_manuscript_figures.py fig22 (silica_side_scatter/fig22_scatter_fluorescence_quadrants.csv)', fig,
                    {'quadrants': z.drop(columns=['quadrant']).assign(quadrant=z.quadrant)})

    G = read('fig08_median_b525_with_without_antibody.csv')
    rows = []
    for c, lab in [('Uncoated', 'Bare origami'), ('PLL-PEG', 'PLL-PEG coated')]:
        for cond, nm in [('No probe', 'No antibody'), ('With anti-biotin IgG', 'With antibody')]:
            for r in G[(G.coating == c) & (G.condition == cond)].itertuples():
                rows.append(dict(group=lab, condition=nm, replicate=r.replicate, median_B525=r.median_B525_buffer_subtracted, value='buffer-subtracted'))
    rows += [dict(group='Buffer', condition='No antibody', replicate='', median_B525=G[G.coating == 'Buffer (04/18)'].median_B525_raw.iloc[0], value='raw, Buffer (04/18)'),
             dict(group='Buffer', condition='With antibody', replicate='', median_B525=G[G.coating == 'Antibody only (03/13)'].median_B525_raw.iloc[0], value='raw, antibody only (03/13)')]
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    for i, g in enumerate(['Bare origami', 'PLL-PEG coated', 'Buffer']):
        for k, cond in enumerate(['No antibody', 'With antibody']):
            v = D[(D.group == g) & (D.condition == cond)].median_B525.values; x = i + (k - 0.5) * 0.36
            colr = {'Bare origami': COL['NoCoating'], 'PLL-PEG coated': COL['PLL'], 'Buffer': COL['Buffer']}[g]
            ax.bar(x, v.mean(), 0.32, color=colr, alpha=0.3 if k == 0 else 0.7, label=cond if i == 0 else None)
            if len(v) > 1:
                ax.errorbar(x, v.mean(), v.std(ddof=1), color=INK, capsize=2, lw=0.8)
            ax.scatter(x + jitter(len(v), 0.04), v, s=12, color=colr, edgecolor='white', lw=0.4, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(['Bare origami', 'PLL-PEG coated', 'Buffer'], fontsize=7)
    ax.set_ylabel('Median B525-H'); ax.legend(fontsize=6.5)
    write_panel('S11', 'A', 'Antibody signal with and without antibody',
                'median B525-H of origami-gate scatter-triggered events without and with anti-biotin IgG (no-biotin origami, '
                'antibody session; buffer-subtracted; points are tubes), and for buffer (04/18) and antibody only (03/13) in the '
                'uncoated-origami gate (raw)', 'make_manuscript_figures.py fig08 (figures_svg/fig08_median_b525_with_without_antibody.csv)',
                fig, {'medians': D})

    Mx = read('fig13_extension_dependence_antibody.csv'); Mx = Mx[Mx.coating.isin(['NoCoating', 'PLL'])]
    fig, ax = plt.subplots(figsize=(3.2, 2.7)); rows = []
    for k, c in enumerate(['NoCoating', 'PLL']):
        for i, e in enumerate([3, 6, 12]):
            v = Mx[(Mx.coating == c) & (Mx.extension.astype(int) == e)].M1_biotin_MFI_shift.values; x = i + (k - 0.5) * 0.36
            rows.append(dict(coating=LBL[c], stem_bp=e, n=len(v), mean=v.mean(), sd=v.std(ddof=1)))
            ax.bar(x, v.mean(), 0.32, color=COL[c], alpha=0.35, label=LBL[c] if i == 0 else None)
            ax.errorbar(x, v.mean(), v.std(ddof=1), color=INK, capsize=2, lw=0.8)
            ax.scatter(x + jitter(len(v), 0.04), v, s=12, color=COL[c], edgecolor='white', lw=0.4, zorder=3)
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(['3', '6', '12']); ax.set_xlabel('Stem length (bp)')
    ax.set_ylabel('log$_2$ FC, median B525-H'); ax.legend()
    write_panel('S12', 'A', 'Median antibody signal shift',
                'log2 of the median B525-H of origami-gate events of each biotin sample over that of the matched no-biotin sample '
                '(same coating and replicate), antibody session; bar mean, error bar SD, points replicates',
                'make_manuscript_figures.py fig13 (figures_svg/fig13_extension_dependence_antibody.csv, column M1_biotin_MFI_shift)',
                fig, {'summary': pd.DataFrame(rows), 'samples': Mx})

    R = read('fig10_free_antibody_rates.csv')
    cat = {'All recorded events': 'All events', 'Scatter-triggered (SS ≥ 200)': 'SS+',
           'Fluorescence-triggered (SS < 200 and B525 ≥ 250)': 'SS− and AB+', 'SS ≥ 200 and B525 ≥ 250': 'SS+ and AB+'}
    R['category'] = R.events.map(cat); R['group'] = R['sample'].replace({'Uncoated origami, no antibody (04/18, A-C)': 'Bare origami only'})
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    gc = {'Buffer': COL['Buffer'], 'Antibody only': '#A55FB0', 'Bare origami only': COL['NoCoating']}
    for i, c in enumerate(['All events', 'SS+', 'SS− and AB+', 'SS+ and AB+']):
        for k, g in enumerate(['Buffer', 'Antibody only', 'Bare origami only']):
            v = R[(R.category == c) & (R.group == g)].eps.values; y = i + (k - 1) * 0.26
            ax.barh(y, v.mean(), 0.24, color=gc[g], alpha=0.5, label=g if i == 0 else None)
            ax.scatter(v, y + jitter(len(v), 0.03), s=10, color=gc[g], zorder=3)
    ax.set_xscale('log'); ax.set_xlim(0.01, 1e4); ax.set_yticks(range(4)); ax.set_yticklabels(['All events', 'SS+', 'SS− and AB+', 'SS+ and AB+'])
    ax.set_xlabel('Events per second'); ax.legend(fontsize=6.5, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    write_panel('S13', 'A', 'Event rates of buffer, antibody and origami',
                'events per second of buffer (03/13), antibody only (03/13) and bare origami without antibody (04/18, tubes A to C; '
                'bar mean, points tubes) for all events, scatter-triggered (SS+), fluorescence-triggered (SS− and AB+) and '
                'SS >= 200 with B525-H >= 250 (SS+ and AB+)', 'make_manuscript_figures.py fig10 (figures_svg/fig10_free_antibody_rates.csv)',
                fig, {'rates': R})

    q = read('fig29_tccd_coincidence_samples.csv')
    q = q[q.probe.str.contains('IgG') & q.coating.isin(['Uncoated', 'PLL-PEG']) & q.valid]
    fig, ax = plt.subplots(figsize=(3.3, 2.7)); rows = []
    for k, c in enumerate(['Uncoated', 'PLL-PEG']):
        for i, e in enumerate(['N', '3', '6', '12']):
            v = q[(q.coating == c) & (q.extension.astype(str) == e)].obs_exp_deficit.values; x = i + (k - 0.5) * 0.36
            rows.append(dict(coating=c, stem=e, n=len(v), mean=v.mean(), sd=v.std(ddof=1)))
            ax.bar(x, v.mean(), 0.32, color=COL[CO[c]], alpha=0.4, label=('Bare origami' if c == 'Uncoated' else 'PLL-PEG coated') if i == 0 else None)
            ax.errorbar(x, v.mean(), v.std(ddof=1), color=INK, capsize=2, lw=0.8)
            ax.scatter(x + jitter(len(v), 0.04), v, s=12, color=COL[CO[c]], edgecolor='white', lw=0.4, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(['N', '3', '6', '12']); ax.set_xlabel('Stem length (bp; N, no biotin)')
    ax.set_ylabel('Observed / chance coincidences'); ax.legend(fontsize=6.5, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    write_panel('S14', 'A', 'Observed-to-chance coincidence of origami and antibody',
                'observed coincident events over the chance expectation (Equation 1) for each antibody sample, bare and PLL-PEG, '
                'without biotin (N) and with 3, 6 or 12 bp stems; bar mean, error bar SD, points replicates (one PLL-PEG 12 bp sample '
                'without detectable origami excluded)', 'fig29_tccd_coincidence.py (figures_svg/fig29_tccd_coincidence_samples.csv, column obs_exp_deficit)',
                fig, {'summary': pd.DataFrame(rows), 'samples': q[['coating', 'replicate', 'extension', 'sample', 'R_A', 'R_B', 'N_coinc', 'two_tau_deficit_us', 'obs_exp_deficit']]})

    smp = read('fig29d_tccd_ddG_samples.csv'); s = smp[smp.coating == 'Silica']
    fig, ax = plt.subplots(figsize=(3.2, 2.6)); rows = []
    for i, e in enumerate(['3', '6', '12']):
        v = s[s.extension.astype(str) == e].excess_simple.values
        rows.append(dict(stem_bp=e, n=len(v), mean=v.mean(), sd=v.std(ddof=1)))
        ax.barh(i, v.mean(), 0.6, color=COL['Silica'], alpha=0.4); ax.errorbar(v.mean(), i, xerr=v.std(ddof=1), color=COL['Silica'], capsize=3, lw=1)
        ax.scatter(v, i + jitter(len(v), 0.05), s=16, color=COL['Silica'], zorder=3)
    ax.axvline(0, color=INK, lw=0.8); ax.set_yticks(range(3)); ax.set_yticklabels(['3', '6', '12']); ax.set_ylabel('Stem length (bp)')
    ax.set_xlabel('Coincidence excess (% of origami)')
    write_panel('S15', 'A', 'Silica-coated origami: coincidence excess',
                'biotin-specific coincidence excess (Equation 4) of each silica-coated antibody sample relative to its matched no-biotin '
                'sample; bar mean, error bar SD, points replicates (3_Silica_AB_C excluded at > 7,000 events/s)',
                'fig29_tccd_coincidence.py -> fig29d_tccd_ddG.py (figures_svg/fig29d_tccd_ddG_samples.csv)', fig,
                {'summary': pd.DataFrame(rows), 'samples': s[['coating', 'replicate', 'extension', 'sample', 'excess_simple']]})

    Mp = read('fig11_free_streptavidin_maps.csv'); z = Mp[Mp['sample'] == 'Streptavidin only']
    xe = np.unique(np.r_[Mp.ss_bin_lo, Mp.ss_bin_hi]); ye = np.unique(np.r_[Mp.b525_bin_lo, Mp.b525_bin_hi])
    h = np.full((len(xe) - 1, len(ye) - 1), np.nan)
    for r in z.itertuples():
        h[np.searchsorted(xe, r.ss_bin_lo), np.searchsorted(ye, r.b525_bin_lo)] = r.events_per_s
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    pc = ax.pcolormesh(xe, ye, h.T, norm=LogNorm(), cmap='Blues', rasterized=True)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('VSSC-1-H'); ax.set_ylabel('B525-H (AF488)')
    fig.colorbar(pc, ax=ax).set_label('events/s per bin', fontsize=8)
    write_panel('S16', 'A', 'Streptavidin only: B525-H vs side scatter',
                'events per second per log bin of VSSC-1-H and B525-H for streptavidin without origami (probe_only_controls/'
                'Streptavidin_Only__Shervin5_03.fcs; all recorded events)', 'make_manuscript_figures.py fig11 (figures_svg/fig11_free_streptavidin_maps.csv)',
                fig, {'bins': Mp})
    Rs = read('fig11_free_streptavidin_rates.csv')
    cat = {'All recorded events': 'All events', 'SS ≥ 200 and B525 ≥ 256': 'SS+ and SA+', 'Scatter-triggered (SS ≥ 200)': 'SS+',
           'Fluorescence-triggered (SS < 200 and B525 ≥ 256)': 'SS− and SA+'}
    Rs['category'] = Rs.events.map(cat)
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    for i, c in enumerate(['All events', 'SS+ and SA+', 'SS+', 'SS− and SA+']):
        for k, (g, colr) in enumerate([('Streptavidin only', '#A55FB0'), ('Buffer', COL['Buffer'])]):
            v = Rs[(Rs.category == c) & (Rs['sample'] == g)].eps.values
            ax.bar(i + (k - 0.5) * 0.36, v.mean(), 0.32, color=colr, label=g if i == 0 else None)
    ax.set_yscale('log'); ax.set_xticks(range(4)); ax.set_xticklabels(['All events', 'SS+ and SA+', 'SS+', 'SS− and SA+'], rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Events per second'); ax.legend(fontsize=6.5)
    write_panel('S16', 'B', 'Streptavidin only: event rates',
                'events per second of streptavidin only and buffer (05/03) for all events, SS >= 200 with B525-H >= 256 (SS+ and SA+), '
                'scatter-triggered (SS+) and fluorescence-triggered (SS− and SA+); 256 is the B525 trigger read from the file',
                'make_manuscript_figures.py fig11 (figures_svg/fig11_free_streptavidin_rates.csv)', fig, {'rates': Rs})

    A = read(os.path.join('proposed_panels', 'fig5f_run_order_check_acquisitions.csv')); A = A[A.valid]
    for fig_id, sub, title in [('S17', A[A.extension.astype(str) == 'N'], 'no-biotin origami over the session'),
                               ('S18', A, 'all antibody samples versus acquisition time')]:
        for p, col, ylab in [('A', 'observed_over_chance', 'Observed / chance'), ('B', 'antibody_events_per_s', 'Antibody events/s')]:
            fig, ax = plt.subplots(figsize=(4.2, 2.4))
            for (c, r), q in sub.groupby(['coating', 'replicate']):
                q = q.sort_values('hours_since_first_acquisition'); colr = COL[CO[c]]
                if fig_id == 'S18':
                    ax.plot(q.hours_since_first_acquisition, q[col], color=colr, lw=1)
                    for mk, e in zip('osD^', ['N', '3', '6', '12']):
                        w = q[q.extension.astype(str) == e]; ax.plot(w.hours_since_first_acquisition, w[col], marker=mk, ls='', ms=4, color=colr)
                else:
                    ax.plot(q.hours_since_first_acquisition, q[col], marker='o', ls='', ms=4, color=colr)
            hs = [plt.Line2D([], [], color=COL[CO[c]], marker='o', ls='', label=c if c != 'Uncoated' else 'bare') for c in ['Uncoated', 'PLL-PEG', 'Silica']]
            if fig_id == 'S18':
                hs += [plt.Line2D([], [], color=MUTED, marker=mk, ls='', label=lab) for mk, lab in zip('osD^', ['no biotin', '3 bp', '6 bp', '12 bp'])]
            ax.legend(handles=hs, fontsize=6, ncol=len(hs), loc='upper center', bbox_to_anchor=(0.5, 1.22), frameon=False)
            ax.set_xlabel('Hours since the start of acquisition'); ax.set_ylabel(ylab)
            write_panel(fig_id, p, f"{ylab}, {title}",
                        f"{'observed-to-chance coincidence ratio' if p == 'A' else 'antibody event rate (B525-H >= 250, buffer-subtracted)'} of each "
                        f"{'no-biotin ' if fig_id == 'S17' else ''}antibody sample against its start time (hours after the first acquisition of the "
                        f"session){'; lines join the samples of one coating and replicate set in acquisition order' if fig_id == 'S18' else ''}",
                        'fig37_proposed_flow_panels.py panel_5f (proposed_panels/fig5f_run_order_check_acquisitions.csv)', fig, {'acquisitions': sub})


# ------------------------------------------------------------------------------------------------ main
def main():
    if os.path.isdir(PKG):
        for d in os.listdir(PKG):
            if d.startswith('Figure_'):
                shutil.rmtree(os.path.join(PKG, d))
    os.makedirs(PKG, exist_ok=True)
    captions = {}
    print('image panels'); image_panels(captions)
    print('Figure 2'); fig2()
    print('Figure 3'); fig3()
    print('Figure 4'); fig4()
    print('Figure 5'); fig5()
    print('SI'); si()
    Ix = pd.DataFrame(INDEX)
    order = lambda f: (f.startswith('S'), int(f.lstrip('S')))
    Ix = Ix.iloc[sorted(range(len(Ix)), key=lambda i: (order(Ix.figure.iloc[i]), Ix.panel.iloc[i]))]
    Ix.to_csv(os.path.join(PKG, 'panel_index.csv'), index=False)
    for fig, z in Ix.groupby('figure'):
        head, cap = 'Caption (11/09 draft):', captions.get(fig, '(not found)')
        n = int(fig[1:]) if fig.startswith('S') else 0
        if n == SI_INSERTED:
            head, cap = 'Suggested caption (new figure, 2026-11-21):', S5_CAPTION
        elif n > SI_INSERTED:
            head = f'Caption (11/09 draft, where this was Figure S{n - 1}; figure numbers updated):'
        lines = [f'Figure {fig}', '', head, cap, '', 'Panels:']
        lines += [f'  {r.panel}  {r.title}  [{r.kind}]' for r in z.itertuples()]
        open(os.path.join(PKG, f'Figure_{fig}', 'README.txt'), 'w', encoding='utf8').write('\n'.join(lines) + '\n')
    print(f'{len(Ix)} panels written to {PKG}')


if __name__ == '__main__':
    main()
