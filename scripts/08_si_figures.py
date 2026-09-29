"""The eleven Supporting Information figures.

Builds every SI figure. Individual figures can be rebuilt by passing their keys.

  S3-1   Acquisition quality control: event rates, durations, gated retention
  S3-2   Origami identity against coating-only debris on the ATTO 425 axis
  S3-3   All nine side-scatter axes compared across coatings
  S4-1   Per-event classifier detail: confusion, separability, correlations
  S4-2   Deep Sets aggregation: bag size, seed band, ablation, learning curve
  S5-1   Alternative metrics considered and rejected
  S5-2   Gate sensitivity (carried over, see below)
  S5-3a  Per-replicate probe-channel distributions, streptavidin
  S5-3b  Per-replicate probe-channel distributions, antibody
  S5-4   Extension length, and why 12 nt does not outperform 6 nt
  S6-1   Cross-session channel drift and classifier transfer

Figure S5-2 is version-controlled rather than rebuilt; its generator predates
the scripts kept here. This script confirms the file is present and warns if it
is not, rather than silently producing an incomplete set.

Figure S4-2 requires scripts/07_si_deepsets.py to have run first.

Inputs   raw FCS; results/; cache/
Outputs  figures/supplementary/FigS*.{pdf,png}
Usage    python scripts/08_si_figures.py [figure ...]
         with no arguments it builds all of them
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, json, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from fcsparser import parse

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,'axes.linewidth':0.9,
    'axes.spines.top':False,'axes.spines.right':False,
    'xtick.major.width':0.9,'ytick.major.width':0.9,'legend.frameon':False,
})


# --- repository bootstrap -------------------------------------------------
# Resolve `nfcm` whether or not the package has been pip-installed, and take
# every path from nfcm.paths so this script holds no absolute path.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from nfcm.gating import load_gates
from nfcm.paths import (
    CACHE_DIR, FIGURES_SI, RESULTS_INTERMEDIATE, RESULTS_MAIN,
    RESULTS_SI, data_root, require_data_root,
)

#: Root of the raw FCS archive. See README.md for the download link and
#: nfcm.paths.data_root for how the location is resolved.
PROJ = str(require_data_root())
CACHE_DIR.mkdir(parents=True, exist_ok=True)
R   = str(RESULTS_INTERMEDIATE)
R3  = str(RESULTS_SI)
ML3 = str(RESULTS_SI)
OUT = str(FIGURES_SI)
os.makedirs(OUT, exist_ok=True)

COAT=['NoCoating','PLL','Silica']
LBL ={'NoCoating':'Uncoated','PLL':'PLL-PEG','Silica':'Silica'}
COL ={'NoCoating':'#1F77B4','PLL':'#2CA02C','Silica':'#FF7F0E'}
EXT =['N','3','6','12']
EXTL={'N':'N (no biotin)','3':'3 nt','6':'6 nt','12':'12 nt'}
EXTC={'N':'#4D4D4D','3':'#377EB8','6':'#4DAF4A','12':'#E41A1C'}
REPS=['A','B','C']
SS9 =['VSSC-1-H','VSSC-1-A','VSSC-1-W','VSSC-2-H','VSSC-2-A','VSSC-2-W',
      'BSSC-H','BSSC-A','BSSC-W']
CHANCE=1/3

gates=load_gates()
DEBRIS={'NoCoating':f'{PROJ}/Shervin5_25/Buffer.fcs',
        'PLL':      f'{PROJ}/Shervin2_13_2026/PLL_PEG_NoOrigami.fcs',
        'Silica':   f'{PROJ}/Shervin2_13_2026/SilicaChemicalsOnly.fcs'}
SESS={'strep':dict(root='Shervin5_25',pt='S',npt='NoS',sk='SS_S',vk='V525_S_LO',
                   lab='Streptavidin'),
      'ab':   dict(root='Shervin4_18/Shervin4_18',pt='AB',npt='NoAB',sk='SS_A',
                   vk='V525_A_LO',lab='Anti-biotin IgG')}

RNG=np.random.default_rng(3)
def load(fp): return parse(fp,reformat_meta=True)[1]
def sub(d): return d[d.Coating.isin(COAT)] if 'Coating' in d else d[d.coating.isin(COAT)]

def suptitle(fig,t): fig.text(0.012,0.995,t,fontsize=15,fontweight='bold',va='top',ha='left')
def letter(fig,ax,s,dx=-0.055,dy=0.035):
    p=ax.get_position()
    fig.text(max(0.004,p.x0+dx),min(0.988,p.y1+dy),s,fontsize=16,fontweight='bold',
             va='bottom',ha='left')
def barpanel(ax,vals,ylab,title,log=False,pct=False):
    """vals: {coating: [replicate values]}"""
    x=np.arange(len(COAT))
    for xi,c in enumerate(COAT):
        v=np.asarray(vals[c],dtype=float); v=v[~np.isnan(v)]
        m=v.mean(); s=v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0
        ax.bar(xi,m,0.62,yerr=s,capsize=4,color=COL[c],edgecolor='black',
               linewidth=0.6,alpha=0.9)
        for q in v:
            ax.scatter(xi+RNG.uniform(-0.06,0.06),q,s=26,color='black',
                       edgecolor='white',linewidth=0.6,zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in COAT],fontsize=10.5,rotation=18)
    ax.set_ylabel(ylab,fontsize=11)
    ax.set_title(title,fontsize=12,fontweight='bold',pad=7)
    if log: ax.set_yscale('log')
    ax.grid(True,axis='y',alpha=0.15,linewidth=0.5)
def save(fig,name):
    for e in ('pdf','png'):
        fig.savefig(f'{OUT}/{name}.{e}',dpi=300,bbox_inches='tight')
    plt.close(fig); print(f'  wrote {name}')

# =====================================================================  S3-1
def figS3_1():
    flux=sub(pd.read_csv(f'{R}/aux_event_flux.csv'))
    dub =sub(pd.read_csv(f'{R}/aux_doublet_fraction.csv'))
    pur =sub(pd.read_csv(f'{R}/aux_origami_purity.csv'))
    fig,axs=plt.subplots(1,4,figsize=(17,4.3))
    plt.subplots_adjust(top=0.78,bottom=0.20,left=0.05,right=0.99,wspace=0.38)
    barpanel(axs[0],{c:flux[flux.Coating==c].flux_raw_evtsec.values for c in COAT},
             'Raw event rate (events s$^{-1}$)','Acquisition event rate')
    axs[0].axhline(7000,color='#D62728',ls='--',lw=1.4)
    axs[0].text(0.97,7000,' coincidence ceiling',color='#D62728',fontsize=9,
                va='bottom',ha='right',transform=axs[0].get_yaxis_transform())
    barpanel(axs[1],{c:100*flux[flux.Coating==c].gated_fraction.values for c in COAT},
             'Events inside monomer gate (%)','Monomer gate retention')
    barpanel(axs[2],{c:dub[dub.Coating==c].doublet_frac_pct.values for c in COAT},
             'Doublet fraction (%)','Doublet content')
    barpanel(axs[3],{c:100*pur[pur.Coating==c].V525_pos_frac.values for c in COAT},
             'ATTO 425-positive gated events (%)','Origami content of the gate',log=True)
    fig.canvas.draw()
    suptitle(fig,'Figure S3-1.  Acquisition quality control across coatings')
    for ax,s in zip(axs,'ABCD'): letter(fig,ax,s,dx=-0.040,dy=0.030)
    save(fig,'FigS3-1_acquisition_QC')

# =====================================================================  S3-2
def figS3_2():
    gmm=sub(pd.read_csv(f'{R}/aux_purity_v525_gmm.csv'))
    slp=sub(pd.read_csv(f'{R}/aux_purity_v525_ss_slope.csv'))
    ho =sub(pd.read_csv(f'{R}/aux_higher_order_stats.csv'))
    fig=plt.figure(figsize=(17,11.2))
    gs=fig.add_gridspec(3,1,height_ratios=[1.0,1.0,1.0],hspace=0.62,
                        top=0.915,bottom=0.055,left=0.055,right=0.99)
    # Row A: debris vs origami V525 overlay
    rA=gs[0].subgridspec(1,3,wspace=0.22); axA=[]
    xx=np.linspace(0.7,3.5,400)
    for i,c in enumerate(COAT):
        ax=fig.add_subplot(rA[0,i]); axA.append(ax)
        lo,hi=gates['SS_S'][c]; vlo=gates['V525_S_LO'][c]
        d=load(DEBRIS[c]); dv=d[(d['VSSC-1-H']>=lo)&(d['VSSC-1-H']<=hi)]['V525-H'].values
        dv=dv[dv>0]
        ov=[]
        for r in REPS:
            fp=f'{PROJ}/Shervin5_25/N_{c}_NoS_{r}.fcs'
            if not os.path.exists(fp): continue
            o=load(fp); v=o[(o['VSSC-1-H']>=lo)&(o['VSSC-1-H']<=hi)]['V525-H'].values
            ov.append(v[v>0])
        ov=np.concatenate(ov)
        if len(dv)>20:
            ax.plot(xx,gaussian_kde(np.log10(dv),bw_method=0.12)(xx),color='#8A8A8A',
                    lw=2.2,label='Coating debris')
        ax.plot(xx,gaussian_kde(np.log10(ov),bw_method=0.12)(xx),color=COL[c],
                lw=2.4,label='Coated origami')
        ax.axvline(np.log10(vlo),color='#D62728',ls='--',lw=1.6)
        ax.set_xlim(0.7,3.5); ax.set_yticks([])
        ax.set_xlabel('log$_{10}$(V525-H)',fontsize=10.5)
        if i==0: ax.set_ylabel('Density',fontsize=11); ax.legend(loc='upper left',fontsize=9.5)
        ax.set_title(LBL[c],fontsize=12,fontweight='bold',pad=6)
        ax.grid(True,axis='x',alpha=0.10)
    # Row B: GMM diagnostics
    rB=gs[1].subgridspec(1,4,wspace=0.42); axB=[]
    for i,(ylab,title,dat,log) in enumerate([
        ('$\\Delta$BIC favouring two components','V525-H bimodality',
         {c:gmm[gmm.Coating==c].delta_BIC_favor_2c.values for c in COAT},True),
        ('Low-mode weight (%)','Debris sub-population share',
         {c:100*gmm[gmm.Coating==c].gmm2_low_weight.values for c in COAT},False),
        ('V525-H vs side-scatter slope','Size-brightness coupling',
         {c:slp[slp.Coating==c].v525_vs_ss_slope.values for c in COAT},False),
        ('Pearson r','Coupling correlation',
         {c:slp[slp.Coating==c].r.values for c in COAT},False)]):
        ax=fig.add_subplot(rB[0,i]); axB.append(ax)
        barpanel(ax,dat,ylab,title,log=log)
    # Row C: shape statistics
    rC=gs[2].subgridspec(1,4,wspace=0.42); axC=[]
    for i,(ylab,title,col,log) in enumerate([
        ('log$_{10}$ SD of side scatter','Side-scatter spread','log_std',False),
        ('Skewness','Side-scatter skewness','skewness',False),
        ('Excess kurtosis','Side-scatter kurtosis','kurtosis',True),
        ('Interquartile range (log$_{10}$)','Side-scatter interquartile range','iqr_log',False)]):
        ax=fig.add_subplot(rC[0,i]); axC.append(ax)
        barpanel(ax,{c:ho[ho.Coating==c][col].values for c in COAT},ylab,title,log=log)
    fig.canvas.draw()
    suptitle(fig,'Figure S3-2.  Origami identity, debris overlap, and side-scatter distribution shape')
    letter(fig,axA[0],'A',dx=-0.045,dy=0.048)
    letter(fig,axB[0],'B',dx=-0.045,dy=0.030)
    letter(fig,axC[0],'C',dx=-0.045,dy=0.030)
    save(fig,'FigS3-2_identity_debris')

# =====================================================================  S3-3
def figS3_3():
    d=pd.read_parquet(str(CACHE_DIR / 'coating_features_ss_full.parquet'))
    d['Coating']=d.Coating.astype(str); d=d[d.Coating.isin(COAT)]
    fig,axs=plt.subplots(3,3,figsize=(15.5,11))
    plt.subplots_adjust(top=0.915,bottom=0.06,left=0.06,right=0.985,
                        hspace=0.42,wspace=0.20)
    for k,ch in enumerate(SS9):
        ax=axs[k//3,k%3]; f=f'log_{ch}'
        lo,hi=np.percentile(d[f],[0.5,99.5]); xx=np.linspace(lo,hi,300)
        for c in COAT:
            v=d[d.Coating==c][f].values
            if len(v)>5000: v=RNG.choice(v,5000,replace=False)
            ax.plot(xx,gaussian_kde(v,bw_method=0.20)(xx),color=COL[c],lw=2.2,
                    label=LBL[c] if k==0 else None)
        ax.set_xlabel(f'log$_{{10}}$({ch})',fontsize=10.5); ax.set_yticks([])
        if k%3==0: ax.set_ylabel('Density',fontsize=11)
        ax.set_title(ch,fontsize=12,fontweight='bold',pad=6)
        ax.grid(True,axis='x',alpha=0.10)
        if k==0: ax.legend(loc='upper right',fontsize=10)
    fig.canvas.draw()
    suptitle(fig,'Figure S3-3.  Coating signatures across all nine side-scatter pulse features')
    save(fig,'FigS3-3_ss_axes')

# =====================================================================  S6-1
def figS6_1():
    dr=pd.read_csv(f'{R}/aux_cross_session_drift.csv'); dr=dr[dr.coating.isin(COAT)]
    piv=dr.pivot(index='channel',columns='coating',values='shift_5_03_minus_5_25')[COAT]
    piv['mean_abs']=piv.abs().mean(1); piv=piv.sort_values('mean_abs',ascending=False)
    chans=list(piv.index)
    fig,axs=plt.subplots(1,3,figsize=(17,5.0))
    plt.subplots_adjust(top=0.80,bottom=0.30,left=0.055,right=0.99,wspace=0.30)
    # A: grouped bars per coating
    x=np.arange(len(chans)); w=0.26
    for i,c in enumerate(COAT):
        axs[0].bar(x+(i-1)*w,piv[c].values,w,color=COL[c],edgecolor='black',
                   linewidth=0.5,label=LBL[c])
    axs[0].axhline(0,color='#555',lw=0.9)
    axs[0].set_xticks(x); axs[0].set_xticklabels(chans,rotation=55,ha='right',fontsize=9.5)
    axs[0].set_ylabel('$\\Delta$ log$_{10}$ median',fontsize=11)
    axs[0].set_title('Channel drift between sessions',fontsize=12,fontweight='bold',pad=7)
    axs[0].legend(fontsize=10); axs[0].grid(True,axis='y',alpha=0.15)
    # B: per channel lines
    for c in COAT:
        axs[1].plot(x,piv[c].values,'o-',color=COL[c],lw=2.0,ms=6,label=LBL[c])
    axs[1].axhline(0,color='#555',lw=0.9)
    axs[1].set_xticks(x); axs[1].set_xticklabels(chans,rotation=55,ha='right',fontsize=9.5)
    axs[1].set_ylabel('$\\Delta$ log$_{10}$ median',fontsize=11)
    axs[1].set_title('Direction of drift per coating',fontsize=12,fontweight='bold',pad=7)
    axs[1].legend(fontsize=10); axs[1].grid(True,axis='y',alpha=0.15)
    # C: mean abs drift
    cols=['#D62728' if ch.startswith('BSSC') else '#777' for ch in chans]
    axs[2].bar(x,piv.mean_abs.values,0.66,color=cols,edgecolor='black',linewidth=0.5)
    axs[2].set_xticks(x); axs[2].set_xticklabels(chans,rotation=55,ha='right',fontsize=9.5)
    axs[2].set_ylabel('Mean |$\\Delta$ log$_{10}$ median|',fontsize=11)
    axs[2].set_title('Mean absolute drift per channel',fontsize=12,fontweight='bold',pad=7)
    axs[2].grid(True,axis='y',alpha=0.15)
    axs[2].legend(handles=[Line2D([],[],color='#D62728',lw=7,label='Blue side scatter'),
                           Line2D([],[],color='#777',lw=7,label='Violet side scatter')],
                  fontsize=10,loc='upper right')
    fig.canvas.draw()
    suptitle(fig,'Figure S6-1.  Between-session instrument drift')
    for ax,s in zip(axs,'ABC'): letter(fig,ax,s,dx=-0.042,dy=0.030)
    save(fig,'FigS6-1_cross_session_drift')

FIGS={'S3-1':figS3_1,'S3-2':figS3_2,'S3-3':figS3_3,'S6-1':figS6_1}

# =====================================================================  S4-1
def figS4_1():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    pw=pd.read_csv(f'{R3}/tableS4-1_pairwise.csv')
    cm=pd.read_csv(f'{ML3}/confusion_matrix.csv',index_col=0)
    t42=pd.read_csv(f'{R3}/tableS4-2_mmd.csv')
    fl =pd.read_csv(f'{R3}/tableS4-5_fluor_ablation.csv')
    mm=pd.read_csv(f'{R}/mmd_matrix_9axis.csv',index_col=0)
    keys=[k for k in mm.index if k.rsplit('-',1)[0] in COAT]
    M=mm.loc[keys,keys].values.astype(float); M=np.clip(M,0,None)

    # panel F: per-coating AUC, 1 feature vs 9
    d=pd.read_parquet(str(CACHE_DIR / 'coating_features_ss_full.parquet'))
    d['Coating']=d.Coating.astype(str); d['Replicate']=d.Replicate.astype(str)
    d=d[d.Coating.isin(COAT)]
    tr,te=d[d.Replicate.isin(['A','B'])],d[d.Replicate=='C']
    def bal(x,s):
        n=int(x.Coating.value_counts().min())
        return pd.concat([g.sample(n,random_state=s) for _,g in x.groupby('Coating')])
    tr,te=bal(tr,0),bal(te,1)
    le=LabelEncoder().fit(COAT); ytr=le.transform(tr.Coating); yte=le.transform(te.Coating)
    aucs={}
    for tag,fs in [('1 feature',['log_VSSC-1-H']),('9 features',[f'log_{c}' for c in SS9])]:
        m=XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,
                        objective='multi:softprob',num_class=3,eval_metric='mlogloss',
                        random_state=0,n_jobs=-1,verbosity=0).fit(tr[fs],ytr)
        pr=m.predict_proba(te[fs])
        aucs[tag]={c:roc_auc_score((yte==i).astype(int),pr[:,i]) for i,c in enumerate(le.classes_)}

    fig=plt.figure(figsize=(17,10.4))
    gs=fig.add_gridspec(2,3,hspace=0.62,wspace=0.34,top=0.915,bottom=0.075,
                        left=0.055,right=0.985)
    # A pairwise AUC
    axA=fig.add_subplot(gs[0,0])
    names=[p.replace('NoCoating','Uncoated').replace('  vs  ',' vs ') for p in pw.pair]
    order=np.argsort(-pw.auc.values)
    cols=['#2CA02C' if a>=0.8 else '#FF7F0E' if a>=0.6 else '#D62728' for a in pw.auc.values[order]]
    axA.barh(np.arange(len(order))[::-1],pw.auc.values[order],0.6,color=cols,
             edgecolor='black',linewidth=0.5)
    for y,a in zip(np.arange(len(order))[::-1],pw.auc.values[order]):
        axA.text(a+0.012,y,f'{a:.3f}',va='center',fontsize=10)
    axA.axvline(0.5,color='#777',ls='--',lw=1.3)
    axA.set_yticks(np.arange(len(order))[::-1])
    axA.set_yticklabels([names[i] for i in order],fontsize=10.5)
    axA.set_xlim(0,1.0); axA.set_xlabel('ROC AUC',fontsize=11)
    axA.set_title('Pairwise separability',fontsize=12,fontweight='bold',pad=7)
    axA.grid(True,axis='x',alpha=0.15)
    # B confusion
    axB=fig.add_subplot(gs[0,1])
    cn=cm.div(cm.sum(1),axis=0).values*100
    im=axB.imshow(cn,cmap='Blues',vmin=0,vmax=100)
    for i in range(3):
        for j in range(3):
            axB.text(j,i,f'{cn[i,j]:.0f}',ha='center',va='center',fontsize=12,
                     color='white' if cn[i,j]>55 else 'black')
    axB.set_xticks(range(3)); axB.set_xticklabels([LBL[c] for c in COAT],fontsize=10,rotation=20)
    axB.set_yticks(range(3)); axB.set_yticklabels([LBL[c] for c in COAT],fontsize=10)
    axB.set_xlabel('Predicted',fontsize=11); axB.set_ylabel('True',fontsize=11)
    axB.set_title('Per-event confusion (row %)',fontsize=12,fontweight='bold',pad=7)
    plt.colorbar(im,ax=axB,fraction=0.046,pad=0.04)
    # C MMD matrix
    axC=fig.add_subplot(gs[0,2])
    im=axC.imshow(M,cmap='magma')
    for k in (3,6): axC.axhline(k-0.5,color='white',lw=2); axC.axvline(k-0.5,color='white',lw=2)
    tk=[f'{LBL[k.rsplit("-",1)[0]]}-{k.rsplit("-",1)[1]}' for k in keys]
    axC.set_xticks(range(len(keys))); axC.set_xticklabels(tk,rotation=90,fontsize=8.5)
    axC.set_yticks(range(len(keys))); axC.set_yticklabels(tk,fontsize=8.5)
    axC.set_title('MMD$^2$ between coating-replicates',fontsize=12,fontweight='bold',pad=7)
    plt.colorbar(im,ax=axC,fraction=0.046,pad=0.04)
    # D within/between
    axD=fig.add_subplot(gs[1,0])
    w=t42[t42.Category=='Within']; b=t42[t42.Category=='Between']
    lab=list(w.Comparison)+list(b.Comparison)
    val=list(w.MMD2)+list(b.MMD2)
    cols=['#1F77B4']*len(w)+['#D62728']*len(b)
    axD.barh(np.arange(len(val))[::-1],val,0.62,color=cols,edgecolor='black',linewidth=0.5)
    for y,v in zip(np.arange(len(val))[::-1],val):
        axD.text(v+max(val)*0.02,y,f'{v:.4f}',va='center',fontsize=9.5)
    axD.set_yticks(np.arange(len(val))[::-1]); axD.set_yticklabels(lab,fontsize=10)
    axD.set_xlim(0,max(val)*1.24); axD.set_xlabel('MMD$^2$',fontsize=11)
    axD.set_title('Within- vs between-coating MMD',fontsize=12,fontweight='bold',pad=7)
    axD.grid(True,axis='x',alpha=0.15)
    axD.legend(handles=[Line2D([],[],color='#1F77B4',lw=7,label='Within coating'),
                        Line2D([],[],color='#D62728',lw=7,label='Between coatings')],
               fontsize=9.5,loc='lower right')
    # E fluorescence ablation
    axE=fig.add_subplot(gs[1,1])
    x=np.arange(len(fl))
    axE.bar(x,fl.macro_auc,0.6,color='#1F77B4',edgecolor='black',linewidth=0.5)
    for xi,v in zip(x,fl.macro_auc): axE.text(xi,v+0.008,f'{v:.3f}',ha='center',fontsize=10)
    axE.axhline(0.5,color='#777',ls='--',lw=1.3)
    axE.set_xticks(x)
    axE.set_xticklabels([s.replace(' + ','\n+ ') for s in fl.feature_set],fontsize=9)
    axE.set_ylim(0,0.95); axE.set_ylabel('Macro ROC AUC',fontsize=11)
    axE.set_title('Adding fluorescence to side scatter',fontsize=12,fontweight='bold',pad=7)
    axE.grid(True,axis='y',alpha=0.15)
    # F per-coating AUC 1 vs 9
    axF=fig.add_subplot(gs[1,2])
    x=np.arange(len(COAT)); w2=0.36
    for i,(tag,hatch) in enumerate([('1 feature',''),('9 features','//')]):
        axF.bar(x+(i-0.5)*w2,[aucs[tag][c] for c in COAT],w2,
                color=[COL[c] for c in COAT],edgecolor='black',linewidth=0.5,
                hatch=hatch,alpha=0.9)
        for xi,c in zip(x,COAT):
            axF.text(xi+(i-0.5)*w2,aucs[tag][c]+0.01,f'{aucs[tag][c]:.2f}',
                     ha='center',fontsize=9)
    axF.axhline(0.5,color='#777',ls='--',lw=1.3)
    axF.set_xticks(x); axF.set_xticklabels([LBL[c] for c in COAT],fontsize=10.5)
    axF.set_ylim(0,1.05); axF.set_ylabel('One-vs-rest ROC AUC',fontsize=11)
    axF.set_title('Single axis vs all nine axes',fontsize=12,fontweight='bold',pad=7)
    axF.grid(True,axis='y',alpha=0.15)
    axF.legend(handles=[Line2D([],[],color='#999',lw=7,label='VSSC-1-H only'),
                        Line2D([],[],color='#999',lw=7,ls='--',label='All 9 axes')],
               fontsize=9.5,loc='lower right')
    fig.canvas.draw()
    suptitle(fig,'Figure S4-1.  Per-event coating classification: separability limits and feature contributions')
    for ax,s in zip([axA,axB,axC,axD,axE,axF],'ABCDEF'):
        letter(fig,ax,s,dx=-0.045,dy=0.030)
    save(fig,'FigS4-1_perevent_classifier')

# =====================================================================  S4-2
def figS4_2():
    sc =pd.read_csv(f'{ML3}/deepsets_scaling.csv')
    sd =pd.read_csv(f'{ML3}/deepsets_scaling_seeds.csv')
    ab =pd.read_csv(f'{ML3}/deepsets_channel_ablation.csv')
    lc =pd.read_csv(f'{ML3}/deepsets_learning_curve.csv')
    nv =pd.read_csv(f'{ML3}/deepsets_scaling_shervin5_03_naive.csv')
    rc =pd.read_csv(f'{ML3}/deepsets_scaling_shervin5_03_recal.csv')
    fig,axs=plt.subplots(2,3,figsize=(17,10.0))
    plt.subplots_adjust(top=0.905,bottom=0.075,left=0.055,right=0.985,
                        hspace=0.42,wspace=0.28)
    def fin(ax,ylab,title,ylim=(0.2,0.95)):
        ax.axhline(CHANCE,color='#777',ls='--',lw=1.4)
        ax.set_xscale('log'); ax.set_xlabel('Particles per bag, N',fontsize=11)
        ax.set_ylabel(ylab,fontsize=11); ax.set_ylim(*ylim)
        ax.set_title(title,fontsize=12,fontweight='bold',pad=7)
        ax.grid(True,alpha=0.13)
    # A scaling + seed band
    ax=axs[0,0]
    g=sd.groupby('N')['accuracy'].agg(['min','max'])
    ax.fill_between(g.index,g['min'],g['max'],color='#2CA02C',alpha=0.22,zorder=1)
    ax.plot(sc.N,sc.accuracy,'o-',color='#2CA02C',lw=2.2,ms=7,zorder=3)
    pk=sc.loc[sc.accuracy.idxmax()]
    ax.annotate(f'peak {pk.accuracy:.3f}\nat N = {int(pk.N)}',xy=(pk.N,pk.accuracy),
                xytext=(pk.N*2.2,pk.accuracy-0.13),fontsize=10,
                arrowprops=dict(arrowstyle='->',lw=1.2))
    fin(ax,'Bag-level accuracy','Multi-particle aggregation scaling')
    ax.text(1.05,CHANCE+0.008,f'chance ({CHANCE:.3f})',fontsize=9.5,color='#777')
    # B per-class recall
    ax=axs[0,1]
    for c in COAT: ax.plot(sc.N,sc[f'acc_{c}'],'o-',color=COL[c],lw=2.0,ms=6,label=LBL[c])
    fin(ax,'Per-class recall','Per-coating recall vs bag size',ylim=(-0.03,1.06))
    ax.legend(fontsize=10,loc='center left')
    # C channel ablation
    ax=axs[0,2]
    Ns=[1,10,100,1000]
    for name,col in [('Height (H)','#1F77B4'),('Area (A)','#2CA02C'),
                     ('Width (W)','#D62728'),('All 9 axes','#333333')]:
        r=ab[ab.feature_set==name]
        if r.empty: continue
        ax.plot(Ns,[float(r[f'N{n}'].iloc[0]) for n in Ns],'o-',color=col,lw=2.0,ms=6,label=name)
    fin(ax,'Bag-level accuracy','Pulse-feature ablation')
    ax.legend(fontsize=10,loc='lower right')
    # D learning curve
    ax=axs[1,0]
    for frac,col in [(0.25,'#3B2E8C'),(0.50,'#1B9E9E'),(1.00,'#C7E02C')]:
        r=lc[lc.train_frac==frac].sort_values('N')
        ax.plot(r.N,r.accuracy,'o-',color=col,lw=2.0,ms=6,label=f'{frac:.0%} of training events')
    fin(ax,'Bag-level accuracy','Training-set size dependence')
    ax.legend(fontsize=10,loc='lower right')
    # E cross-session
    ax=axs[1,1]
    ax.plot(sc.N,sc.accuracy,'o-',color='#2CA02C',lw=2.2,ms=6,label='Within session')
    ax.plot(nv.N,nv.accuracy,'s-',color='#D62728',lw=2.0,ms=6,label='Held-out session')
    ax.plot(rc.N,rc.accuracy,'^--',color='#FF7F0E',lw=2.0,ms=6,label='Held-out, recalibrated')
    fin(ax,'Bag-level accuracy','Cross-session generalisation')
    ax.legend(fontsize=10,loc='upper left')
    # F cross-session per-class
    ax=axs[1,2]
    for c in COAT: ax.plot(nv.N,nv[f'acc_{c}'],'o-',color=COL[c],lw=2.0,ms=6,label=LBL[c])
    fin(ax,'Per-class recall','Held-out session per-coating recall',ylim=(-0.03,1.06))
    ax.legend(fontsize=10,loc='center left')
    fig.canvas.draw()
    suptitle(fig,'Figure S4-2.  Multi-particle aggregation with permutation-invariant Deep Sets')
    for ax,s in zip(axs.ravel(),'ABCDEF'): letter(fig,ax,s,dx=-0.043,dy=0.028)
    save(fig,'FigS4-2_deepsets')

FIGS.update({'S4-1':figS4_1,'S4-2':figS4_2})


# =====================================================================  S5-1
def _grouped(ax,get,ylab,title,null=0.0,log=False):
    """get(coat, ext) -> array of replicate values."""
    x=np.arange(len(COAT)); w=0.20
    for ei,e in enumerate(['N','3','6','12']):
        mm,ss=[],[]
        for c in COAT:
            v=np.asarray(get(c,e),dtype=float); v=v[~np.isnan(v)]
            mm.append(v.mean() if len(v) else np.nan)
            ss.append(v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0)
        ax.bar(x+(ei-1.5)*w,mm,w,yerr=ss,capsize=3,error_kw={'linewidth':0.8},
               color=EXTC[e],edgecolor='black',linewidth=0.5,label=EXTL[e])
        for xi,c in enumerate(COAT):
            v=np.asarray(get(c,e),dtype=float); v=v[~np.isnan(v)]
            for q in v:
                ax.scatter(xi+(ei-1.5)*w+RNG.uniform(-0.02,0.02),q,s=12,color='black',
                           edgecolor='white',linewidth=0.4,zorder=5,alpha=0.85)
    if null is not None: ax.axhline(null,color='#D62728',ls='--',lw=1.3,alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in COAT],fontsize=10.5)
    ax.set_ylabel(ylab,fontsize=11)
    ax.set_title(title,fontsize=12,fontweight='bold',pad=7)
    if log: ax.set_yscale('log')
    ax.grid(True,axis='y',alpha=0.15)

def figS5_1():
    ssm=pd.read_csv(f'{R3}/si51_ss_medians.csv')
    ssm=ssm[ssm.sess=='strep']
    mc =pd.read_csv(f'{RESULTS_MAIN}/method1_method2_metrics.csv'); mc=mc[mc.coating.isin(COAT)]
    alt=pd.read_csv(f'{R}/alt_metrics_emd.csv'); alt=alt[alt.coating.isin(COAT)]

    def ss_vs(ref_kind):
        def g(c,e):
            if ref_kind=='probe':
                base=ssm[(ssm.coat==c)&(ssm.ext=='N')&(ssm.kind=='probe')].ss_med.mean()
            else:
                base=ssm[(ssm.coat==c)&(ssm.kind=='noprobe')].ss_med.mean()
            v=ssm[(ssm.coat==c)&(ssm.ext==e)]
            v=v[v.kind=='probe'] if e=='N' else v
            return (v.ss_med.values-base)
        return g
    def probe_med(c,e):
        r=mc[(mc.session=='strep')&(mc.coating==c)&(mc.extension==e)]
        return (r.B525_median-r.N_noprobe_median).values
    def emd(meth):
        def g(c,e):
            return alt[(alt.method==meth)&(alt.coating==c)&(alt.ext==e)]['val'].values
        return g

    fig=plt.figure(figsize=(17,9.6))
    gs=fig.add_gridspec(2,6,hspace=0.46,wspace=1.25,top=0.905,bottom=0.075,
                        left=0.055,right=0.985)
    axes=[fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[0,2:4]),fig.add_subplot(gs[0,4:6]),
          fig.add_subplot(gs[1,1:3]),fig.add_subplot(gs[1,3:5])]
    _grouped(axes[0],ss_vs('probe'),'$\\Delta$ side-scatter median (a.u.)',
             'Rejected: side scatter vs no-biotin + probe')
    _grouped(axes[1],ss_vs('noprobe'),'$\\Delta$ side-scatter median (a.u.)',
             'Rejected: side scatter vs no-probe')
    _grouped(axes[2],probe_med,'$\\Delta$ B525-H median (a.u.)',
             'Accepted: probe channel vs no-probe')
    _grouped(axes[3],emd('D'),'Earth-mover distance (a.u.)',
             'Accepted: EMD vs no-probe',null=None)
    _grouped(axes[4],emd('E'),'Earth-mover distance (a.u.)',
             'Accepted: EMD vs no-biotin + probe',null=None)
    axes[0].legend(title='Biotin extension',fontsize=9.5,title_fontsize=10,
                   loc='upper left',ncol=2)
    fig.canvas.draw()
    suptitle(fig,'Figure S5-1.  Alternative accessibility metrics evaluated during method development')
    for ax,s_ in zip(axes,'ABCDE'): letter(fig,ax,s_,dx=-0.045,dy=0.028)
    save(fig,'FigS5-1_alternative_metrics')

# =====================================================================  S5-3a/b
def _per_replicate(skey,name,title):
    S=SESS[skey]
    fig,axs=plt.subplots(len(COAT),4,figsize=(15.5,3.05*len(COAT)),
                         sharex=True)
    plt.subplots_adjust(top=0.905,bottom=0.075,left=0.075,right=0.90,
                        hspace=0.34,wspace=0.14)
    xx=np.linspace(1,4,400)
    RCOL={'A':'#1F77B4','B':'#2CA02C','C':'#D62728'}
    def b525(ext,coat,tag,rep):
        fp=f'{PROJ}/{S["root"]}/{ext}_{coat}_{tag}_{rep}.fcs'
        if not os.path.exists(fp): return None
        d=load(fp); lo,hi=gates[S['sk']][coat]; vlo=gates[S['vk']][coat]
        a=d[(d['VSSC-1-H']>=lo)&(d['VSSC-1-H']<=hi)&(d['V525-H']>=vlo)]['B525-H'].values
        a=a[a>0]
        return a if len(a)>=40 else None
    for ri,c in enumerate(COAT):
        pool=[a for r in REPS if (a:=b525('N',c,S['npt'],r)) is not None]
        ref=np.concatenate(pool) if pool else None
        for ci,e in enumerate(EXT):
            ax=axs[ri,ci]
            if ref is not None:
                ax.fill_between(xx,gaussian_kde(np.log10(ref),bw_method=0.15)(xx),
                                color='#B8B8B8',alpha=0.40,edgecolor='#777',
                                linewidth=0.8,linestyle=(0,(4,2)),zorder=1)
            tag=S['npt'] if e=='N' else S['pt']
            for r in REPS:
                a=b525(e,c,tag,r)
                if a is None: continue
                ax.plot(xx,gaussian_kde(np.log10(a),bw_method=0.15)(xx),
                        color=RCOL[r],lw=1.9,zorder=3,
                        label=f'Replicate {r}' if (ri==0 and ci==0) else None)
            ax.set_xlim(1,4); ax.set_yticks([])
            ax.grid(True,axis='x',alpha=0.10)
            if ri==0: ax.set_title(EXTL[e],fontsize=12,fontweight='bold',pad=7)
            if ci==0: ax.set_ylabel(LBL[c],fontsize=12,fontweight='bold')
            if ri==len(COAT)-1: ax.set_xlabel('log$_{10}$(B525-H)',fontsize=10.5)
    h=[Line2D([],[],color=RCOL[r],lw=2.0,label=f'Replicate {r}') for r in REPS]
    h.append(Line2D([],[],color='#777',lw=2.0,ls='--',label='Pooled no-probe'))
    fig.legend(handles=h,loc='center left',bbox_to_anchor=(0.905,0.5),fontsize=10.5,
               title='',frameon=False)
    fig.canvas.draw()
    suptitle(fig,title)
    save(fig,name)

def figS5_3a():
    _per_replicate('strep','FigS5-3a_per_replicate_a',
        'Figure S5-3a.  Per-replicate probe-channel distributions for streptavidin')
def figS5_3b():
    _per_replicate('ab','FigS5-3b_per_replicate_b',
        'Figure S5-3b.  Per-replicate probe-channel distributions for anti-biotin immunoglobulin G')

# =====================================================================  S5-4
def figS5_4():
    mc=pd.read_csv(f'{RESULTS_MAIN}/method1_method2_metrics.csv'); mc=mc[mc.coating.isin(COAT)]
    fig,axs=plt.subplots(1,3,figsize=(17,4.8))
    plt.subplots_adjust(top=0.80,bottom=0.20,left=0.055,right=0.985,wspace=0.32)
    for k,(sess,lab) in enumerate([('strep','Streptavidin'),('ab','Anti-biotin IgG')]):
        ax=axs[k]; x=np.arange(len(COAT)); w=0.36
        for i,(grp,col,hatch) in enumerate([('3 and 6 nt','#4DAF4A',''),
                                            ('12 nt','#E41A1C','//')]):
            mm,ss=[],[]
            for c in COAT:
                s_=mc[(mc.session==sess)&(mc.coating==c)]
                v=s_[s_.extension.isin(['3','6'])] if i==0 else s_[s_.extension=='12']
                v=v['M1_biotin_MFI_shift'].dropna().values
                mm.append(v.mean()); ss.append(v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0)
            ax.bar(x+(i-0.5)*w,mm,w,yerr=ss,capsize=4,color=col,edgecolor='black',
                   linewidth=0.5,hatch=hatch,label=grp)
            for xi,c in enumerate(COAT):
                s_=mc[(mc.session==sess)&(mc.coating==c)]
                v=s_[s_.extension.isin(['3','6'])] if i==0 else s_[s_.extension=='12']
                for q in v['M1_biotin_MFI_shift'].dropna().values:
                    ax.scatter(xi+(i-0.5)*w+RNG.uniform(-0.03,0.03),q,s=14,color='black',
                               edgecolor='white',linewidth=0.4,zorder=5)
        ax.axhline(0,color='#D62728',ls='--',lw=1.3)
        ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in COAT],fontsize=10.5)
        ax.set_ylabel('log$_2$(biotin ext / N-ext)',fontsize=11)
        ax.set_title(f'Probe-channel median shift, {lab}',fontsize=12,fontweight='bold',pad=7)
        ax.legend(fontsize=10); ax.grid(True,axis='y',alpha=0.15)
    ax=axs[2]; x=np.arange(3)
    for xi,e in enumerate(['3','6','12']):
        v=mc[mc.extension==e]['M2_biotin_specific_fraction'].dropna().values
        m=v.mean(); s_=v.std(ddof=1)/np.sqrt(len(v))
        ax.bar(xi,m,0.6,yerr=s_,capsize=4,color=EXTC[e],edgecolor='black',linewidth=0.6)
        for q in v:
            ax.scatter(xi+RNG.uniform(-0.08,0.08),q,s=14,color='black',
                       edgecolor='white',linewidth=0.4,zorder=5)
    ax.axhline(1.0,color='#D62728',ls='--',lw=1.3)
    ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(['3 nt','6 nt','12 nt'],fontsize=11)
    ax.set_ylabel('% monomers > N-ext P99',fontsize=11)
    ax.set_title('Biotin-specific fraction, both probes pooled',fontsize=12,
                 fontweight='bold',pad=7)
    ax.grid(True,axis='y',alpha=0.15)
    fig.canvas.draw()
    suptitle(fig,'Figure S5-4.  Biotin extension length does not follow the expected monotonic trend')
    for ax,s_ in zip(axs,'ABC'): letter(fig,ax,s_,dx=-0.042,dy=0.030)
    save(fig,'FigS5-4_extension_length')

FIGS.update({'S5-1':figS5_1,'S5-3a':figS5_3a,'S5-3b':figS5_3b,'S5-4':figS5_4})


# =====================================================================  main
ORDER=['S3-1','S3-2','S3-3','S4-1','S4-2','S5-1','S5-3a','S5-3b','S5-4','S6-1']
if __name__=='__main__':
    want=sys.argv[1:] or [k for k in ORDER if k in FIGS]
    for k in want:
        if k in FIGS:
            print(f'building Figure {k} ...'); FIGS[k]()
        else:
            print(f'  (no builder for {k})')
    if not sys.argv[1:]:
        # Figure S5-2 shows only uncoated and silica, so removing the fourth
        # surface condition left it unchanged and it was carried over rather
        # than rebuilt. Its generator predates the scripts kept in this
        # repository, so the published file is version-controlled directly and
        # this step only confirms it is present.
        missing = [e for e in ('pdf', 'png')
                   if not os.path.exists(f'{OUT}/FigS5-2_gate_sensitivity.{e}')]
        if missing:
            print(f'  WARNING: FigS5-2_gate_sensitivity.{{{",".join(missing)}}} '
                  f'not found in {OUT}; restore it from version control')
        else:
            print('  FigS5-2 carried over unchanged (uncoated and silica only)')
