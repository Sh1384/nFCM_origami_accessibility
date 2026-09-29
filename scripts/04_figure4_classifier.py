"""Figure 4: label-free coating classifier.

Asks whether a coating can be identified from one particle's scatter signature
alone, with no fluorescent label. XGBoost (200 trees, depth 5, learning rate
0.1) is trained on replicates A and B and tested on the held-out replicate C,
with classes balanced by subsampling.

Panels
  A  UMAP of the nine-axis side-scatter space, ground truth beside prediction
  B  One-vs-rest ROC curves on the held-out replicate
  C  Feature importance (normalised gain) across the nine side-scatter axes
  D  Macro AUC by feature set: one axis, all nine, and nine plus fluorescence

Panel D carries the result: adding fluorescence does not improve on side scatter
alone, so coating identity is genuinely readable without a label.

Inputs   cache/coating_features_3coat_fluor.parquet (built on demand)
Outputs  figures/main/Fig4_ML_manuscript.{pdf,png}
         results/si/ml_ss_axes_{metrics,auc_per_class}.csv, feature_importance.csv
Usage    python scripts/04_figure4_classifier.py
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from fcsparser import parse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import umap

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,'axes.linewidth':0.9,
    'axes.spines.top':False,'axes.spines.right':False,
    'xtick.major.width':0.9,'ytick.major.width':0.9,
})


# --- repository bootstrap -------------------------------------------------
# Resolve `nfcm` whether or not the package has been pip-installed, and take
# every path from nfcm.paths so this script holds no absolute path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from nfcm.gating import load_gates
from nfcm.paths import (
    CACHE_DIR, FIGURES_MAIN, RESULTS_SI,
    data_root, require_data_root,
)

#: Root of the raw FCS archive, resolved without asserting it exists.
#: Only needed when the feature cache has to be rebuilt from raw FCS.
PROJ = str(data_root())
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT  = str(FIGURES_MAIN)
RES  = str(RESULTS_SI); os.makedirs(RES, exist_ok=True)
FEAT = str(CACHE_DIR / 'coating_features_3coat_fluor.parquet')

COATINGS    = ['NoCoating','PLL','Silica']
COAT_LABELS = {'NoCoating':'NoCoating','PLL':'PLL','Silica':'Silica'}
COAT_COLORS = {'NoCoating':'#1F77B4','PLL':'#2CA02C','Silica':'#FF7F0E'}
REPS = ['A','B','C']
SS = ['VSSC-1-H','VSSC-1-A','VSSC-1-W','VSSC-2-H','VSSC-2-A','VSSC-2-W',
      'BSSC-H','BSSC-A','BSSC-W']
FLUOR = ['V525-H','B525-H']
F_SS = [f'log_{c}' for c in SS]
F_FL = [f'log_{c}' for c in FLUOR]

# ---------------------------------------------------------------- features
if os.path.exists(FEAT):
    df = pd.read_parquet(FEAT)
    print(f'loaded cached features: {len(df):,} events')
else:
    # Rebuilding the cache reads raw FCS, so the archive is required here.
    require_data_root()
    rows = []
    for coat in COATINGS:
        for rep in REPS:
            fp = f'{PROJ}/Shervin5_25/N_{coat}_NoS_{rep}.fcs'
            if not os.path.exists(fp): continue
            _, d = parse(fp, reformat_meta=True)
            f = d[d['VSSC-1-H'] >= 300].copy()
            ok = (f[SS+FLUOR] > 0).all(axis=1)
            f = f[ok]
            out = {f'log_{c}': np.log10(f[c].values) for c in SS+FLUOR}
            out['Coating'] = coat; out['Replicate'] = rep
            rows.append(pd.DataFrame(out))
            print(f'  {coat:<10} {rep}: {len(f):>7,} events')
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(FEAT, index=False)
    print(f'\ntotal: {len(df):,} events -> cached')

tr_all = df[df.Replicate.isin(['A','B'])]; te_all = df[df.Replicate == 'C']
def bal(x, seed):
    n = int(x.Coating.value_counts().min())
    return pd.concat([g.sample(n, random_state=seed) for _, g in x.groupby('Coating')])
tr, te = bal(tr_all, 0), bal(te_all, 1)
print(f'\ntrain {len(tr):,} ({len(tr)//3:,}/class)   test {len(te):,} ({len(te)//3:,}/class)')

le = LabelEncoder().fit(COATINGS)
ytr, yte = le.transform(tr.Coating), le.transform(te.Coating)

def fit(feats):
    m = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                      objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                      random_state=0, n_jobs=-1, verbosity=0)
    m.fit(tr[feats], ytr)
    pr = m.predict_proba(te[feats])
    return m, pr

SCEN = [('1 axis\n(VSSC-1-H only)', ['log_VSSC-1-H'],          '#1F77B4'),
        ('9 axes\n(all SS)',        F_SS,                       '#2CA02C'),
        ('3 fluor\n(SS + V525 + B525)', ['log_VSSC-1-H']+F_FL,  '#FF7F0E')]
scen_res, m9, pr9 = [], None, None
for name, feats, col in SCEN:
    m, pr = fit(feats)
    yp = pr.argmax(1)
    r = dict(scenario=name.replace('\n',' '), n_features=len(feats),
             accuracy=accuracy_score(yte, yp), macro_f1=f1_score(yte, yp, average='macro'),
             macro_auc=roc_auc_score(yte, pr, multi_class='ovr', average='macro'))
    scen_res.append((name, col, r))
    print(f"  {r['scenario']:<30} acc={r['accuracy']:.3f}  macroAUC={r['macro_auc']:.3f}")
    if feats is F_SS: m9, pr9 = m, pr
pd.DataFrame([r for _,_,r in scen_res]).to_csv(f'{RES}/ml_ss_axes_metrics.csv', index=False)

# per-class ROC
roc = {}
for i, c in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve((yte == i).astype(int), pr9[:, i])
    roc[c] = (fpr, tpr, roc_auc_score((yte == i).astype(int), pr9[:, i]))
Yb = np.eye(3)[yte]
fpr_mi, tpr_mi, _ = roc_curve(Yb.ravel(), pr9.ravel())
auc_mi = roc_auc_score(Yb.ravel(), pr9.ravel())
grid = np.linspace(0, 1, 1000)
tpr_ma = np.mean([np.interp(grid, roc[c][0], roc[c][1]) for c in le.classes_], axis=0)
auc_ma = float(np.mean([roc[c][2] for c in le.classes_]))
pd.DataFrame([{'class':c,'AUC':roc[c][2]} for c in le.classes_]
             + [{'class':'micro-average','AUC':auc_mi},{'class':'macro-average','AUC':auc_ma}]
            ).to_csv(f'{RES}/ml_ss_axes_auc_per_class.csv', index=False)
print('\nper-class AUC: ' + '  '.join(f'{c}={roc[c][2]:.3f}' for c in le.classes_)
      + f'  micro={auc_mi:.3f}  macro={auc_ma:.3f}')

# feature importance (normalised gain, matching the original figure)
imp = pd.DataFrame({'feature': SS, 'importance': m9.feature_importances_}) \
        .sort_values('importance', ascending=False).reset_index(drop=True)
imp.to_csv(f'{RES}/feature_importance.csv', index=False)
blue = imp[imp.feature.str.contains('BSSC')].importance.sum()
print(f'\nfeature importance:\n{imp.to_string(index=False)}')
print(f'  blue-detector share = {blue*100:.1f}%')

cm = confusion_matrix(yte, pr9.argmax(1))
pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(f'{RES}/confusion_matrix.csv')

# ---------------------------------------------------------------- UMAP
# Rebuildable: the UMAP embedding is a cache, not a reported result.
UMAP_CACHE = str(CACHE_DIR / 'umap_embedding.npz')
N_PER = 5000
sub = df.groupby('Coating', observed=True).sample(n=N_PER, random_state=42)
if os.path.exists(UMAP_CACHE):
    z = np.load(UMAP_CACHE, allow_pickle=True); emb, y_sub = z['emb'], z['y']
    print('\nloaded cached UMAP')
else:
    print('\nrunning UMAP on 9D side-scatter space...')
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean',
                    random_state=42, n_jobs=1).fit_transform(sub[F_SS].values)
    y_sub = sub.Coating.values.astype(str)
    np.savez(UMAP_CACHE, emb=emb, y=y_sub)
pred_sub = le.inverse_transform(m9.predict(sub[F_SS]))

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(15.5, 15.5))
gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.05, 0.58],
                      hspace=0.42, top=0.945, bottom=0.045, left=0.070, right=0.985)

# Panel A
rowA = gs[0].subgridspec(1, 2, wspace=0.16)
for j, (lab, yy, pref) in enumerate([('Ground truth', y_sub, ''),
                                     ('XGBoost predicted', pred_sub, 'pred: ')]):
    ax = fig.add_subplot(rowA[0, j])
    for c in COATINGS:
        m = yy == c
        ax.scatter(emb[m,0], emb[m,1], s=3, alpha=0.40, c=COAT_COLORS[c],
                   label=f'{pref}{COAT_LABELS[c]}', edgecolors='none', rasterized=True)
    ax.set_title(lab, fontsize=12.5, fontweight='bold', pad=6)
    ax.set_xlabel('UMAP-1', fontsize=11)
    if j == 0: ax.set_ylabel('UMAP-2', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True)
    ax.legend(loc='upper left', fontsize=10, markerscale=4, framealpha=0.92)
    ax.grid(True, alpha=0.07)

# Panel B / C
rowB = gs[1].subgridspec(1, 2, wspace=0.30, width_ratios=[1.0, 0.95])
axB = fig.add_subplot(rowB[0,0])
order = sorted(le.classes_, key=lambda c: -roc[c][2])
for c in order:
    f_, t_, a_ = roc[c]
    axB.plot(f_, t_, color=COAT_COLORS[c], lw=2.0, label=f'{COAT_LABELS[c]}  (AUC = {a_:.3f})')
axB.plot(fpr_mi, tpr_mi, color='#E377C2', lw=1.8, ls='--', label=f'Micro-avg  (AUC = {auc_mi:.3f})')
axB.plot(grid, tpr_ma, color='#17BECF', lw=1.8, ls='--', label=f'Macro-avg  (AUC = {auc_ma:.3f})')
axB.plot([0,1],[0,1], color='#999', lw=1.2, ls=':', label='Chance  (AUC = 0.5)')
axB.set_xlim(0,1); axB.set_ylim(0,1.005)
axB.set_xlabel('False Positive Rate', fontsize=11); axB.set_ylabel('True Positive Rate', fontsize=11)
axB.set_title('One-vs-rest ROC', fontsize=12.5, fontweight='bold', loc='left', pad=6)
axB.legend(loc='lower right', fontsize=9.5, frameon=True, framealpha=0.92)
axB.grid(True, alpha=0.12)

axC = fig.add_subplot(rowB[0,1])
yv = np.arange(len(imp))[::-1]
axC.barh(yv, imp.importance.values, height=0.68, color='#1F77B4',
         edgecolor='black', linewidth=0.5)
for y_, v_ in zip(yv, imp.importance.values):
    axC.text(v_ + imp.importance.max()*0.015, y_, f'{v_:.3f}', va='center', fontsize=9.5)
axC.set_yticks(yv); axC.set_yticklabels(imp.feature.values, fontsize=10.5)
axC.set_xlim(0, imp.importance.max()*1.16)
axC.set_xlabel('XGBoost gain (feature importance)', fontsize=11)
axC.set_title('XGBoost feature importance', fontsize=12.5, fontweight='bold', loc='left', pad=6)
axC.grid(True, axis='x', alpha=0.12)

# Panel D
rowD = gs[2].subgridspec(1, 1)
axD = fig.add_subplot(rowD[0,0])
x = np.arange(len(scen_res))
for xi, (name, col, r) in zip(x, scen_res):
    axD.bar(xi, r['macro_auc'], 0.55, color=col, edgecolor='black', linewidth=0.6)
    axD.text(xi, r['macro_auc']+0.015, f"{r['macro_auc']:.3f}", ha='center', fontsize=11)
axD.axhline(0.5, color='#777', lw=1.8, ls='--', label='chance (0.5)')
axD.set_xticks(x); axD.set_xticklabels([n for n,_,_ in scen_res], fontsize=10.5)
axD.set_ylabel('Macro AUC', fontsize=11); axD.set_ylim(0, 1.0)
axD.set_title('Classifier performance by feature set', fontsize=12.5, fontweight='bold',
              loc='left', pad=6)
axD.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10.5, frameon=False)
axD.grid(True, axis='y', alpha=0.12)

fig.canvas.draw()
def hdr(x, ax, txt, off):
    fig.text(x, ax.get_position().y1+off, txt, ha='left', fontsize=13.5,
             fontweight='bold', va='bottom')
hdr(0.070, fig.axes[0], 'A  UMAP of 9-axis side-scatter: ground truth vs XGBoost prediction', 0.030)
hdr(0.012, axB, 'B', 0.026)
hdr(0.520, axC, 'C', 0.026)
hdr(0.012, axD, 'D', 0.026)

for e in ['pdf','png']:
    fig.savefig(f'{OUT}/Fig4_ML_manuscript.{e}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('\nwrote Fig4_ML_manuscript (3 coatings)')
