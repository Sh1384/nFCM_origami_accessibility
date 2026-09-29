"""Supporting Information tables, everything except Deep Sets.

Most Supporting Information quantities are computed per coating from that
coating's own controls, so they are independent of which other coatings are in
the study. This script recomputes them all in one place, including the ones that
genuinely aggregate across coatings, such as the within-group and between-group
maximum mean discrepancy means, the per-channel drift means, and the
fluorescence ablation, which is refitted as a three-class problem.

Inputs   results/intermediate/*.csv; results/main/method1_method2_metrics.csv;
         cache/coating_features_3coat_fluor.parquet
Outputs  results/si/tableS*.csv, results/si/si51_ss_medians.csv
Usage    python scripts/06_si_statistics.py
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, os, json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from fcsparser import parse


# --- repository bootstrap -------------------------------------------------
# Resolve `nfcm` whether or not the package has been pip-installed, and take
# every path from nfcm.paths so this script holds no absolute path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from nfcm.gating import load_gates
from nfcm.paths import (
    CACHE_DIR, RESULTS_INTERMEDIATE, RESULTS_MAIN, RESULTS_SI,
)

# This script reads cached results, never the raw archive, so it needs no data
# root. It does need the feature matrix from scripts/02_build_features.py for
# the fluorescence ablation, so check for it up front and say so plainly rather
# than failing later inside pandas.
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_FEATURES = CACHE_DIR / 'coating_features_3coat_fluor.parquet'
if not _FEATURES.exists():
    raise SystemExit(
        f'missing feature matrix: {_FEATURES}\n'
        'Run scripts/02_build_features.py first (it needs the raw FCS archive).'
    )
R   = str(RESULTS_INTERMEDIATE)
OUT = str(RESULTS_SI); os.makedirs(OUT, exist_ok=True)
C3=['NoCoating','PLL','Silica']
L={'NoCoating':'Uncoated','PLL':'PLL-PEG','Silica':'Silica'}
def hdr(t): print('\n'+'='*76+f'\n{t}\n'+'='*76)

# ------------------------------------------------------- Table S3-1
hdr('TABLE S3-1  Acquisition statistics per coating')
flux=pd.read_csv(f'{R}/aux_event_flux.csv'); flux=flux[flux.Coating.isin(C3)]
dub =pd.read_csv(f'{R}/aux_doublet_fraction.csv'); dub=dub[dub.Coating.isin(C3)]
pur =pd.read_csv(f'{R}/aux_origami_purity.csv'); pur=pur[pur.Coating.isin(C3)]
rows=[]
for c in C3:
    f=flux[flux.Coating==c]; d=dub[dub.Coating==c]; p=pur[pur.Coating==c]
    rows.append(dict(Coating=L[c],
        acq_s=f.duration_s.mean(), raw=f.n_raw.mean(), rate=f.flux_raw_evtsec.mean(),
        retention_pct=100*f.gated_fraction.mean(), gated=f.n_gated.mean(),
        doublets_pct=d.doublet_frac_pct.mean(),
        atto_pos_pct=100*p.V525_pos_frac.mean()))
t31=pd.DataFrame(rows)
print(t31.round(3).to_string(index=False))
t31.to_csv(f'{OUT}/tableS3-1_acquisition.csv', index=False)

# ------------------------------------------------------- Table S3-2
hdr('TABLE S3-2  Side-scatter distribution shape per coating')
ho=pd.read_csv(f'{R}/aux_higher_order_stats.csv'); ho=ho[ho.Coating.isin(C3)]
t32=ho.groupby('Coating').agg(median=('median','mean'), log_sd=('log_std','mean'),
      skewness=('skewness','mean'), kurtosis=('kurtosis','mean'),
      iqr_log=('iqr_log','mean')).reindex(C3).rename(index=L)
print(t32.round(3).to_string())
t32.to_csv(f'{OUT}/tableS3-2_shape.csv')

# ------------------------------------------------------- GMM
hdr('GMM diagnostics (SI 3.2)')
g=pd.read_csv(f'{R}/aux_purity_v525_gmm.csv'); g=g[g.Coating.isin(C3)]
gg=g.groupby('Coating').agg(dBIC=('delta_BIC_favor_2c','mean'),
      low_w=('gmm2_low_weight','mean'), at_buffer=('low_mode_at_buffer_level','mean')).reindex(C3)
print(gg.round(4).to_string())
print(f'\n  delta-BIC range: {gg.dBIC.min():,.0f} to {gg.dBIC.max():,.0f}')
print(f'  low-component weight range: {100*gg.low_w.min():.0f}% to {100*gg.low_w.max():.0f}%')

# ------------------------------------------------------- Table S4-1
hdr('TABLE S4-1  Pairwise coating separability (3 pairs)')
pw=pd.read_csv(f'{R}/pairwise_results.csv')
print('  available columns:', list(pw.columns))
mask=pw.apply(lambda r: not any('orona' in str(v) for v in r.values), axis=1)
print(pw[mask].to_string(index=False))
pw[mask].to_csv(f'{OUT}/tableS4-1_pairwise.csv', index=False)

# ------------------------------------------------------- Table S4-2  MMD
hdr('TABLE S4-2  Within- and between-coating MMD, recomputed over 3 coatings')
mm=pd.read_csv(f'{R}/mmd_matrix_9axis.csv', index_col=0)
keys=[k for k in mm.index if k.rsplit('-',1)[0] in C3]
sub=mm.loc[keys,keys]
def coat(k): return k.rsplit('-',1)[0]
within={c:[] for c in C3}; between={}
for i,a in enumerate(keys):
    for b in keys[i+1:]:
        v=float(sub.loc[a,b]); ca,cb=coat(a),coat(b)
        if ca==cb: within[ca].append(v)
        else: between.setdefault(tuple(sorted((ca,cb))),[]).append(v)
wrows=[dict(Category='Within',Comparison=L[c],MMD2=np.mean(within[c]),Pairs=len(within[c])) for c in C3]
brows=[dict(Category='Between',Comparison=f'{L[a]} vs {L[b]}',MMD2=np.mean(v),Pairs=len(v))
       for (a,b),v in sorted(between.items(), key=lambda kv: np.mean(kv[1]))]
t42=pd.DataFrame(wrows+brows)
print(t42.round(5).to_string(index=False))
wm=np.mean([v for c in C3 for v in within[c]]); bm=np.mean([v for v in sum(between.values(),[])])
print(f'\n  mean WITHIN  = {wm:.4f}')
print(f'  mean BETWEEN = {bm:.4f}')
print(f'  ratio        = {bm/wm:.1f}x')
print(f'  largest between: {t42[t42.Category=="Between"].iloc[-1].Comparison} = {t42[t42.Category=="Between"].MMD2.max():.3f}')
print(f'  smallest between: {t42[t42.Category=="Between"].iloc[0].Comparison} = {t42[t42.Category=="Between"].MMD2.min():.3f}')
print(f'  uncoated within ({np.mean(within["NoCoating"]):.3f}) > uncoated-vs-PLL between? '
      f'{np.mean(within["NoCoating"]) > np.mean(between[tuple(sorted(("NoCoating","PLL")))])}')
t42.to_csv(f'{OUT}/tableS4-2_mmd.csv', index=False)

# ------------------------------------------------------- fluorescence ablation
hdr('TABLE S4-5  Fluorescence feature ablation, refitted as 3-class')
fp=str(CACHE_DIR / 'coating_features_3coat_fluor.parquet')
d=pd.read_parquet(fp)
tr=d[d.Replicate.isin(['A','B'])]; te=d[d.Replicate=='C']
def bal(x,s):
    n=int(x.Coating.value_counts().min())
    return pd.concat([g.sample(n,random_state=s) for _,g in x.groupby('Coating')])
tr,te=bal(tr,0),bal(te,1)
le=LabelEncoder().fit(C3); ytr=le.transform(tr.Coating); yte=le.transform(te.Coating)
SETS=[('SS only',['log_VSSC-1-H']),
      ('SS + V525',['log_VSSC-1-H','log_V525-H']),
      ('SS + V525 + B525',['log_VSSC-1-H','log_V525-H','log_B525-H'])]
d2=d.copy()
rows=[]
for name,fs in SETS:
    m=XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,objective='multi:softprob',
                    num_class=3,eval_metric='mlogloss',random_state=0,n_jobs=-1,verbosity=0)
    m.fit(tr[fs],ytr); pr=m.predict_proba(te[fs]); yp=pr.argmax(1)
    rows.append(dict(feature_set=name,n_features=len(fs),
        accuracy=accuracy_score(yte,yp),macro_f1=f1_score(yte,yp,average='macro'),
        macro_auc=roc_auc_score(yte,pr,multi_class='ovr',average='macro')))
    print(f"  {name:<20} acc={rows[-1]['accuracy']:.3f}  F1={rows[-1]['macro_f1']:.3f}  AUC={rows[-1]['macro_auc']:.3f}")
# ratio feature
tr=tr.assign(log_ratio=tr['log_V525-H']-tr['log_VSSC-1-H'])
te=te.assign(log_ratio=te['log_V525-H']-te['log_VSSC-1-H'])
fs=['log_VSSC-1-H','log_V525-H','log_ratio']
m=XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,objective='multi:softprob',
                num_class=3,eval_metric='mlogloss',random_state=0,n_jobs=-1,verbosity=0)
m.fit(tr[fs],ytr); pr=m.predict_proba(te[fs]); yp=pr.argmax(1)
rows.append(dict(feature_set='SS + V525 + log(V525/SS)',n_features=3,
    accuracy=accuracy_score(yte,yp),macro_f1=f1_score(yte,yp,average='macro'),
    macro_auc=roc_auc_score(yte,pr,multi_class='ovr',average='macro')))
print(f"  {'SS + V525 + log(V525/SS)':<20} acc={rows[-1]['accuracy']:.3f}  AUC={rows[-1]['macro_auc']:.3f}")
pd.DataFrame(rows).to_csv(f'{OUT}/tableS4-5_fluor_ablation.csv', index=False)

# ------------------------------------------------------- Table S6-1 drift
hdr('TABLE S6-1  Channel drift, recomputed over 3 coatings')
dr=pd.read_csv(f'{R}/aux_cross_session_drift.csv'); dr=dr[dr.coating.isin(C3)]
piv=dr.pivot(index='channel',columns='coating',values='shift_5_03_minus_5_25')[C3]
piv['mean_abs']=piv.abs().mean(1)
piv=piv.sort_values('mean_abs',ascending=False).rename(columns=L)
print(piv.round(3).to_string())
piv.to_csv(f'{OUT}/tableS6-1_drift.csv')
print('\n  top drift channels:')
for ch in piv.index[:4]:
    print(f'    {ch:<10} mean|d| = {piv.loc[ch,"mean_abs"]:.3f} log units '
          f'({100*(10**piv.loc[ch,"mean_abs"]-1):.0f}% linear)')
print(f'  all width + VSSC-2 channels <= {piv.loc[[c for c in piv.index if "W" in c or "VSSC-2" in c],"mean_abs"].max():.3f}')
print(f'  PLL-PEG max upward shift: {piv["PLL-PEG"].max():.3f} in {piv["PLL-PEG"].idxmax()}')
print(f'  silica BSSC-A: {piv.loc["BSSC-A","Silica"]:+.3f}, BSSC-H: {piv.loc["BSSC-H","Silica"]:+.3f}')

# ------------------------------------------------------- alt metrics (5.1)
hdr('SI 5.1  Alternative metrics, 3 coatings (9 conditions)')
ss=pd.read_csv(f'{R}/method1_ssshift.csv'); ss=ss[ss.Coating.isin(C3)]
ssn=pd.read_csv(f'{R}/method1_ssshift_NS_threshold.csv'); ssn=ssn[ssn.Coating.isin(C3)]
print('  within-coating replicate range of the SS median under a fixed condition:')
rng_=ss.groupby(['Coating','Extension'])['ss_med_bound'].agg(lambda x: x.max()-x.min())
print(f'    {rng_.min():.0f} to {rng_.max():.0f} a.u.')
alt=pd.read_csv(f'{R}/alt_metrics_emd.csv'); alt=alt[alt.coating.isin(C3)]
print('\n  methods present:', sorted(alt.method.unique()))
for meth in sorted(alt.method.unique()):
    a=alt[alt.method==meth]
    m=a.groupby('coating')['val'].mean().reindex(C3)
    print(f'    {meth}: ' + '  '.join(f'{L[c]} {m[c]:.1f}' for c in C3))
mc=pd.read_csv(f'{RESULTS_MAIN}/method1_method2_metrics.csv')
mc=mc[mc.coating.isin(C3)]
hi=mc[(mc.session=='strep')&(mc.coating.isin(['NoCoating','PLL']))&(mc.extension!='N')]
print(f'\n  probe-channel median (uncoated+PLL, streptavidin, biotin ext): see SI text 231-495 a.u.')

# ------------------------------------------------------- 5.4 extension length
hdr('SI 5.4  Biotin extension length, 3 coatings')
for sess,lbl in [('strep','STREPTAVIDIN'),('ab','ANTIBODY')]:
    print(f'\n{lbl}  (Method 1 log2 shift; is 12 nt <= mean of 3&6 nt?)')
    n_at_or_below=0
    for c in C3:
        s=mc[(mc.session==sess)&(mc.coating==c)]
        short=s[s.extension.isin(['3','6'])]['M1_biotin_MFI_shift'].mean()
        twelve=s[s.extension=='12']['M1_biotin_MFI_shift'].mean()
        ok = twelve <= short
        n_at_or_below += ok
        print(f'  {L[c]:<9} 3&6 nt mean {short:+.3f}   12 nt {twelve:+.3f}   12nt<=short: {ok}')
    print(f'  -> 12 nt at or below the shorter extensions in {n_at_or_below} of {len(C3)} coatings')
print(f'\nDONE. tables -> {OUT}')
