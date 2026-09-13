#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'; DER=ROOT/'derived'
FIG.mkdir(exist_ok=True); DER.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Liberation Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})

ca='CA_Level5A'; cb='CB_Level6A_strict'

def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))

def savefig(fig,name):
    fig.tight_layout()
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)

def fig1_concept():
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    ax.set_aspect('equal'); ax.set_xlim(-0.5,7.8); ax.set_ylim(-0.6,3.6); ax.axis('off')
    O=np.array([0.5,0.4]); z1=np.array([2.4,2.6]); z2=np.array([3.4,1.45])
    for z,label in [(z1,r'$z_i=A_i e^{i\phi_i}$'),(z2,r'$z_j=A_j e^{i\phi_j}$')]:
        ax.add_patch(FancyArrowPatch(O,z,arrowstyle='-|>',mutation_scale=12,lw=1.6))
        ax.text(*(z+[0.08,0.08]),label)
    ax.plot([z1[0],z2[0]],[z1[1],z2[1]],lw=1.4)
    ax.text(2.55,2.0,r'$|z_i-z_j|$',rotation=-42)
    ax.text(0.75,1.75,r'radial amplitude $A$',rotation=49)
    ax.text(1.2,0.55,r'angular relation $\Delta\phi$')
    # Right-hand equations
    x=4.2
    ax.text(x,3.0,r'$D_{FULL}=D_{AMP}+D_{AP}$',fontsize=12,weight='bold')
    ax.text(x,2.35,r'$D_{AMP}=\sum(A_i-A_j)^2$')
    ax.text(x,1.75,r'$D_{AP}=\sum 2A_iA_j[1-\cos(\Delta\phi)]$')
    ax.text(x,1.05,r'$D_{FULL}=ENERGY+INTERFERENCE$')
    ax.text(x,0.45,r'$INTERFERENCE=-\sum 2A_iA_j\cos(\Delta\phi)$')
    ax.text(0.2,-0.35,'Mathematics is exact; the empirical discovery concerns reproducibility and phase-randomization tests.',fontsize=8.5)
    savefig(fig,'fig1_radial_angular_concept')

def fig2_primary():
    p1=load('results/paper01/ARTICLE_CONTROL_SUMMARY.json')
    labels=['FULL','AP','AMP']; x=np.arange(3); w=.34
    fig,ax=plt.subplots(figsize=(6.7,4.2))
    for off,coh,name,hatch in [(-w/2,ca,'Cohort 1: frozen confirmation',''),(w/2,cb,'Cohort 2: strict-template','//')]:
        vals=[p1[coh]['observed_spearman'][k]['mean'] for k in labels]
        cis=[p1[coh]['observed_spearman'][k]['ci95'] for k in labels]
        err=np.array([[v-c[0] for v,c in zip(vals,cis)],[c[1]-v for v,c in zip(vals,cis)]])
        ax.bar(x+off,vals,w,label=name,hatch=hatch,edgecolor='black',linewidth=.8)
        ax.errorbar(x+off,vals,yerr=err,fmt='none',capsize=3,lw=1,color='black')
    ax.axhline(.5,ls='--',lw=1,color='0.4',label='Frozen FULL floor')
    ax.set_xticks(x,labels); ax.set_ylabel('Mean cross-run RDM reliability (Spearman)'); ax.set_ylim(0,1.02)
    ax.set_title('Replicated component ordering: FULL ≈ AP > AMP')
    ax.legend(frameon=False,loc='lower left')
    savefig(fig,'fig2_primary_two_cohorts')

def fig3_basis_controls():
    p1=load('results/paper01/ARTICLE_CONTROL_SUMMARY.json')
    cats=['GFT','OOB-PCA','rank-matched\nsensor']; x=np.arange(3); w=.34
    fig,ax=plt.subplots(figsize=(6.7,4.0))
    for off,coh,name,hatch in [(-w/2,ca,'Cohort 1',''),(w/2,cb,'Cohort 2','//')]:
        c=p1[coh]['C2_basis']
        vals=[c['GFT_identity_AP_minus_AMP'],c['OOB_PCA']['AP_minus_AMP_mean'],c['rankmatched_sensor_coordinates']['AP_minus_AMP_mean']]
        ax.bar(x+off,vals,w,label=name,hatch=hatch,edgecolor='black',linewidth=.8)
        # random rotations as band at right visually
    ax.axhline(0,color='black',lw=.8)
    ax.set_xticks(x,cats); ax.set_ylabel('AP − AMP reliability'); ax.set_title('AP > AMP survives alternative coordinate systems')
    ax.legend(frameon=False)
    savefig(fig,'fig3_basis_robustness')

def fig4_components():
    p2=load('results/paper02/PAPER02_SUMMARY.json')
    labels=['ENERGY','PRODUCT','INTERFERENCE','ANGLE']; x=np.arange(len(labels)); w=.34
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    for off,coh,name,hatch in [(-w/2,ca,'Cohort 1',''),(w/2,cb,'Cohort 2','//')]:
        vals=[p2[coh]['spearman'][k]['mean'] for k in labels]
        cis=[p2[coh]['spearman'][k]['ci95'] for k in labels]
        err=np.array([[v-c[0] for v,c in zip(vals,cis)],[c[1]-v for v,c in zip(vals,cis)]])
        ax.bar(x+off,vals,w,label=name,hatch=hatch,edgecolor='black',linewidth=.8)
        ax.errorbar(x+off,vals,yerr=err,fmt='none',capsize=3,lw=1,color='black')
    ax.axhline(0,color='black',lw=.8)
    ax.set_xticks(x,labels); ax.set_ylim(0,1); ax.set_ylabel('Mean cross-run RDM reliability (Spearman)')
    ax.set_title('Both radial scaffold and phase-dependent geometry are reproducible')
    ax.legend(frameon=False)
    savefig(fig,'fig4_exact_components')

def fig5_phase_null():
    p3=load('results/paper03/PAPER03_SUMMARY.json')
    groups=[(ca,'INTERFERENCE'),(cb,'INTERFERENCE'),(ca,'ANGLE'),(cb,'ANGLE')]
    labels=['C1\nINTERFERENCE','C2\nINTERFERENCE','C1\nANGLE','C2\nANGLE']
    x=np.arange(4); w=.34
    obs=[p3[c]['components'][k]['observed_mean'] for c,k in groups]
    nul=[p3[c]['components'][k]['phase_null_population_mean'] for c,k in groups]
    q=[p3[c]['components'][k]['phase_null_population_q025_q975'] for c,k in groups]
    fig,ax=plt.subplots(figsize=(7.2,4.3))
    ax.bar(x-w/2,obs,w,label='Observed',edgecolor='black',linewidth=.8)
    ax.bar(x+w/2,nul,w,label='Amplitude-preserving phase null',hatch='//',edgecolor='black',linewidth=.8)
    nullerr=np.array([[v-qq[0] for v,qq in zip(nul,q)],[qq[1]-v for v,qq in zip(nul,q)]])
    ax.errorbar(x+w/2,nul,yerr=nullerr,fmt='none',capsize=3,lw=1,color='black')
    ax.axhline(0,color='black',lw=.8)
    ax.set_xticks(x,labels); ax.set_ylabel('Mean cross-run RDM reliability'); ax.set_ylim(-.12,.86)
    ax.set_title('Category-specific phase organization exceeds amplitude-preserving nulls')
    ax.legend(frameon=False)
    savefig(fig,'fig5_phase_null')

def fig6_convergence():
    d=load('results/level6d/LEVEL6D_RESULTS.json')
    n=np.array([r['n'] for r in d['population_curve']],float)
    y=np.array([r['mean_R_FULL'] for r in d['population_curve']],float)
    R=d['saturating_fit']['R_infinity']; tau=d['saturating_fit']['tau_trials']
    xx=np.linspace(0,70,400); yy=R*(1-np.exp(-xx/tau))
    fig,ax=plt.subplots(figsize=(6.5,4.0))
    ax.plot(xx,yy,lw=1.6,label=fr'$R_\infty={R:.3f}$, $\tau_n={tau:.2f}$ trials')
    ax.scatter(n,y,s=35,zorder=3)
    ax.set_xlabel('Trials contributing to each ensemble estimate'); ax.set_ylabel('Mean FULL reliability'); ax.set_ylim(-.03,.9)
    ax.set_title('Ensemble convergence of the confirmed geometry'); ax.legend(frameon=False)
    savefig(fig,'fig6_ensemble_convergence')

def write_summary():
    p1=load('results/paper01/ARTICLE_CONTROL_SUMMARY.json'); p2=load('results/paper02/PAPER02_SUMMARY.json'); p3=load('results/paper03/PAPER03_SUMMARY.json')
    rows=[]
    for coh in (ca,cb):
        row={'cohort':coh,'N':p2[coh]['N']}
        for k in ('FULL','AP','AMP','ENERGY','PRODUCT','INTERFERENCE','ANGLE'):
            row[k]=p2[coh]['spearman'][k]['mean']
        row['AP_minus_AMP']=p1[coh]['observed_spearman']['AP_minus_AMP']['mean']
        row['INTERFERENCE_minus_null']=p3[coh]['components']['INTERFERENCE']['observed_minus_null_mean']
        row['ANGLE_minus_null']=p3[coh]['components']['ANGLE']['observed_minus_null_mean']
        rows.append(row)
    pd.DataFrame(rows).to_csv(DER/'MANUSCRIPT_NUMERICAL_SUMMARY.csv',index=False)
    (DER/'MANUSCRIPT_NUMERICAL_SUMMARY.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

def main():
    fig1_concept(); fig2_primary(); fig3_basis_controls(); fig4_components(); fig5_phase_null(); fig6_convergence(); write_summary()
    print('Regenerated figures and numerical summary.')

if __name__=='__main__': main()
