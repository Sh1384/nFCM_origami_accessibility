"""Deep Sets multi-particle coating classifier, THREE-COATING version.

Retrains the permutation-invariant Deep Sets network as a 3-class problem
(uncoated, PLL-PEG, silica) and reproduces every Deep Sets result the
Supporting Information reports:

  deepsets_scaling.csv            accuracy and per-class recall vs bag size
  deepsets_scaling_seeds.csv      seed sweep band
  deepsets_channel_ablation.csv   H / A / W / all-9 at four bag sizes
  deepsets_learning_curve.csv     25 / 50 / 100 % of training events
  deepsets_scaling_*_naive.csv    cross-session transfer, no recalibration
  deepsets_scaling_*_recal.csv    cross-session transfer, per-channel recal
  deepsets_confusion_N*.csv       bag-level confusion at four bag sizes

Chance level is 1/3 = 0.333 throughout, not 0.25.
"""
import os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from fcsparser import parse


# --- repository bootstrap -------------------------------------------------
# Resolve `nfcm` whether or not the package has been pip-installed, and take
# every path from nfcm.paths so this script holds no absolute path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from nfcm.gating import load_gates
from nfcm.paths import (
    CACHE_DIR, CONFIG_DIR, FIGURES_MAIN, FIGURES_SI,
    RESULTS_INTERMEDIATE, RESULTS_MAIN, RESULTS_SI, require_data_root,
)

#: Root of the raw FCS archive. See README.md for the download link and
#: nfcm.paths.data_root for how the location is resolved.
PROJ = str(require_data_root())
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RES  = str(RESULTS_SI)
os.makedirs(RES, exist_ok=True)
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'device: {DEVICE}')

COATINGS = ['NoCoating', 'PLL', 'Silica']
NC = len(COATINGS)
C2I = {c: i for i, c in enumerate(COATINGS)}
SS = ['VSSC-1-H','VSSC-1-A','VSSC-1-W','VSSC-2-H','VSSC-2-A','VSSC-2-W',
      'BSSC-H','BSSC-A','BSSC-W']
FC = [f'log_{c}' for c in SS]

# ---------------------------------------------------------------- data
df = pd.read_parquet(str(CACHE_DIR / 'coating_features_ss_full.parquet'))
df['Coating'] = df.Coating.astype(str); df['Replicate'] = df.Replicate.astype(str)
df = df[df.Coating.isin(COATINGS)]
tr, te = df[df.Replicate.isin(['A','B'])], df[df.Replicate == 'C']
mean = tr[FC].mean().values; std = tr[FC].std(ddof=0).values
trX = (tr[FC].values - mean) / std; teX = (te[FC].values - mean) / std
trY = np.array([C2I[c] for c in tr.Coating]); teY = np.array([C2I[c] for c in te.Coating])
tr_by = {c: np.where(trY == i)[0] for c, i in C2I.items()}
te_by = {c: np.where(teY == i)[0] for c, i in C2I.items()}
print('events per class:')
for c in COATINGS:
    print(f'  {c:<10} train={len(tr_by[c]):>7,}  test={len(te_by[c]):>7,}')

class DeepSets(nn.Module):
    def __init__(self, in_dim=9, embed=128, hid=64, ncls=NC):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim,hid), nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid,hid),    nn.LayerNorm(hid), nn.GELU(),
            nn.Linear(hid,embed))
        self.rho = nn.Sequential(
            nn.Linear(embed,hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hid,hid//2), nn.GELU(), nn.Linear(hid//2,ncls))
    def forward(self,x): return self.rho(self.phi(x).mean(1))

def train_model(X, by, feats_idx, seed=42, steps=4000, frac=1.0, verbose=True):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if frac < 1.0:
        by = {c: rng.choice(v, size=max(2,int(len(v)*frac)), replace=False) for c,v in by.items()}
    Xs = X[:, feats_idx]
    m = DeepSets(in_dim=len(feats_idx)).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    m.train(); t0=time.time()
    for step in range(1, steps+1):
        N = max(1, int(10**rng.uniform(np.log10(5), np.log10(500))))
        lab = rng.integers(0, NC, size=64)
        bags = np.stack([Xs[rng.choice(by[COATINGS[y]], size=N, replace=True)] for y in lab])
        xb = torch.tensor(bags, dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(lab, dtype=torch.long, device=DEVICE)
        opt.zero_grad(); loss = F.cross_entropy(m(xb), yb); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sch.step()
        if verbose and step % 1000 == 0:
            print(f'    step {step:>5}  loss {loss.item():.4f}  {time.time()-t0:.0f}s')
    return m

BAGS = [1,3,5,10,20,30,50,100,200,300,500,1000]
def evaluate(m, X, by, feats_idx, sizes=BAGS, nbags=2000, seed=7, want_cm=()):
    rng = np.random.default_rng(seed); Xs = X[:, feats_idx]
    m.eval(); rows=[]; cms={}
    with torch.no_grad():
        for N in sizes:
            cor=0; tot=0; pc={c:[0,0] for c in COATINGS}
            cm=np.zeros((NC,NC),dtype=int)
            for i in range(0, nbags, 256):
                b = min(256, nbags-i)
                lab = rng.integers(0, NC, size=b)
                bags = np.stack([Xs[rng.choice(by[COATINGS[y]], size=N, replace=True)] for y in lab])
                pred = m(torch.tensor(bags,dtype=torch.float32,device=DEVICE)).argmax(1).cpu().numpy()
                cor += int((pred==lab).sum()); tot += b
                for k in range(b):
                    pc[COATINGS[lab[k]]][1]+=1; cm[lab[k],pred[k]]+=1
                    if pred[k]==lab[k]: pc[COATINGS[lab[k]]][0]+=1
            r={'N':N,'accuracy':cor/tot,'n_bags':tot}
            r.update({f'acc_{c}': pc[c][0]/max(pc[c][1],1) for c in COATINGS})
            rows.append(r)
            if N in want_cm: cms[N]=cm
    return pd.DataFrame(rows), cms

ALL = list(range(9))
print('\n=== main model (all 9 axes) ===')
model = train_model(trX, tr_by, ALL, seed=42)
scal, cms = evaluate(model, teX, te_by, ALL, want_cm=(1,10,100,1000))
scal.to_csv(f'{RES}/deepsets_scaling.csv', index=False)
print(scal.round(3).to_string(index=False))
for N,cm in cms.items():
    pd.DataFrame(cm, index=COATINGS, columns=COATINGS).to_csv(f'{RES}/deepsets_confusion_N{N}.csv')

print('\n=== seed sweep ===')
srows=[]
for sd in [0,1,2]:
    mm = train_model(trX, tr_by, ALL, seed=sd, verbose=False)
    sc,_ = evaluate(mm, teX, te_by, ALL, nbags=1000, seed=100+sd)
    sc['seed']=sd; srows.append(sc); print(f'  seed {sd}: N=100 acc {sc[sc.N==100].accuracy.iloc[0]:.3f}')
pd.concat(srows).to_csv(f'{RES}/deepsets_scaling_seeds.csv', index=False)

print('\n=== channel ablation ===')
SETS={'Height (H)':[0,3,6],'Area (A)':[1,4,7],'Width (W)':[2,5,8],'All 9 axes':ALL}
arows=[]
for name,idx in SETS.items():
    mm = train_model(trX, tr_by, idx, seed=42, verbose=False)
    sc,_ = evaluate(mm, teX, te_by, idx, sizes=[1,10,100,1000], nbags=1000, seed=55)
    d={'feature_set':name,'n_features':len(idx)}
    d.update({f'N{int(r.N)}': r.accuracy for r in sc.itertuples()})
    arows.append(d); print(f'  {name:<12} ' + '  '.join(f'N={int(r.N)}:{r.accuracy:.3f}' for r in sc.itertuples()))
pd.DataFrame(arows).to_csv(f'{RES}/deepsets_channel_ablation.csv', index=False)

print('\n=== learning curve ===')
lrows=[]
for frac in [0.25,0.50,1.00]:
    mm = train_model(trX, tr_by, ALL, seed=42, frac=frac, verbose=False)
    sc,_ = evaluate(mm, teX, te_by, ALL, nbags=1000, seed=77)
    sc['train_frac']=frac; lrows.append(sc)
    print(f'  {frac:.0%}: N=30 {sc[sc.N==30].accuracy.iloc[0]:.3f}  N=100 {sc[sc.N==100].accuracy.iloc[0]:.3f}')
pd.concat(lrows).to_csv(f'{RES}/deepsets_learning_curve.csv', index=False)

print('\n=== cross-session (Shervin5_03) ===')
X03={}
for c in COATINGS:
    fp=f'{PROJ}/Shervin5_03/N_{c}_NoS.fcs'
    if not os.path.exists(fp): print(f'  missing {fp}'); continue
    d=parse(fp,reformat_meta=True)[1]
    f=d[d['VSSC-1-H']>=300]
    f=f[(f[SS]>0).all(axis=1)]
    X03[c]=np.log10(f[SS].values)
    print(f'  {c:<10} {len(f):>7,} events')
if len(X03)==NC:
    naive={c:(v-mean)/std for c,v in X03.items()}
    pooled=np.concatenate([X03[c] for c in COATINGS]); m03=pooled.mean(0); s03=pooled.std(0)
    recal={c:(v-m03)/s03 for c,v in X03.items()}
    for tag,Xd in [('naive',naive),('recal',recal)]:
        Xc=np.concatenate([Xd[c] for c in COATINGS])
        yc=np.concatenate([[C2I[c]]*len(Xd[c]) for c in COATINGS])
        by={c:np.where(yc==i)[0] for c,i in C2I.items()}
        sc,_=evaluate(model, Xc, by, ALL, nbags=2000, seed=777)
        sc.to_csv(f'{RES}/deepsets_scaling_shervin5_03_{tag}.csv', index=False)
        print(f'  {tag:<6} N=1 {sc[sc.N==1].accuracy.iloc[0]:.3f}  '
              f'N=10 {sc[sc.N==10].accuracy.iloc[0]:.3f}  '
              f'N=1000 {sc[sc.N==1000].accuracy.iloc[0]:.3f}')
torch.save({'state_dict':model.state_dict(),'feat_mean':mean,'feat_std':std,
            'feat_cols':FC,'coatings':COATINGS}, str(CACHE_DIR / 'deepsets_ss9_3coat.pt'))
print(f'\nall Deep Sets outputs -> {RES}')
