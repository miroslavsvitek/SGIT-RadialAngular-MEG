import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out=Path('/mnt/data/SGIT_RadialAngular_ImagingNeuroscience_v6/figures')
out.mkdir(parents=True, exist_ok=True)

# Fig 1: conceptual radial-angular geometry and terminology map
fig, ax = plt.subplots(figsize=(11.0,6.4))
ax.set_xlim(0,11.0); ax.set_ylim(0,6.4); ax.axis('off')
O=np.array([0.7,0.8]); zi=np.array([2.9,5.0]); zj=np.array([4.1,2.4])
ax.annotate('', xy=zi, xytext=O, arrowprops=dict(arrowstyle='-|>', lw=2.3))
ax.annotate('', xy=zj, xytext=O, arrowprops=dict(arrowstyle='-|>', lw=2.3))
ax.plot([zi[0],zj[0]],[zi[1],zj[1]],lw=2.2)
ax.text(1.75,3.15,r'$A_i=|z_i|$',rotation=58,fontsize=12)
ax.text(2.45,1.55,r'$A_j=|z_j|$',rotation=21,fontsize=12)
ax.text(2.85,5.18,r'$z_i=A_i e^{i\phi_i}$',fontsize=12)
ax.text(4.08,2.20,r'$z_j=A_j e^{i\phi_j}$',fontsize=12)
ax.text(1.55,0.95,r'$\Delta\phi=\phi_i-\phi_j$',fontsize=12)
ax.text(3.30,3.55,r'$|z_i-z_j|$',fontsize=12,rotation=-66)
ax.text(0.35,5.92,'A  Complex-coordinate geometry',fontsize=13,fontweight='bold')
# algebra panel
x=5.15
ax.text(x,5.92,'B  Two exact decompositions and one diagnostic',fontsize=13,fontweight='bold')
ax.text(x,5.30,r'$D_{\mathrm{FULL}}=D_{\mathrm{AMP}}+D_{\mathrm{AP}}$',fontsize=13)
ax.text(x,4.82,r'$D_{\mathrm{FULL}}=D_{\mathrm{ENERGY}}+D_{\mathrm{INTERFERENCE}}$',fontsize=13)
ax.text(x,4.28,r'$D_{\mathrm{AMP}}=D_{\mathrm{ENERGY}}-D_{\mathrm{PRODUCT}}$',fontsize=12.5)
ax.text(x,3.82,r'$D_{\mathrm{AP}}=D_{\mathrm{PRODUCT}}+D_{\mathrm{INTERFERENCE}}$',fontsize=12.5)
ax.text(x,3.14,r'$D_{\mathrm{ENERGY}}=\sum_m(A_{i,m}^2+A_{j,m}^2)$',fontsize=11.5)
ax.text(x,2.70,r'$D_{\mathrm{PRODUCT}}=\sum_m2A_{i,m}A_{j,m}$',fontsize=11.5)
ax.text(x,2.25,r'$D_{\mathrm{INTERFERENCE}}=-\sum_m2A_{i,m}A_{j,m}\cos(\Delta\phi_m)$',fontsize=11.3)
ax.text(x,1.70,r'$D_{\mathrm{ANGLE}}=\sum_m2[1-\cos(\Delta\phi_m)]$',fontsize=11.5)
ax.text(x,1.05,'PRODUCT: co-activation magnitude; independent of phase',fontsize=10.5)
ax.text(x,0.65,'INTERFERENCE: signed phase cross-term; aligned phases reduce FULL distance',fontsize=10.5)
ax.text(x,0.25,'ANGLE: amplitude-free phase diagnostic (not an exact FULL component)',fontsize=10.5)
fig.tight_layout()
fig.savefig(out/'fig1_radial_angular_concept.pdf', bbox_inches='tight')
plt.close(fig)

# helper colors (matplotlib defaults; hatches distinguish cohorts)
# Fig 2
labels=['FULL complex\ndistance','AP amplitude-weighted\nangular component','AMP amplitude\ndifference component']
conf=np.array([.78514,.74844,.60622]); gen=np.array([.67975,.65371,.48470])
conf_lo=np.array([.716,.680,.500]); conf_hi=np.array([.839,.804,.701])
gen_lo=np.array([.603,.575,.375]); gen_hi=np.array([.749,.725,.589])
xv=np.arange(3); w=.34
fig,ax=plt.subplots(figsize=(9.6,5.5))
ax.bar(xv-w/2,conf,w,label='Confirmation cohort (N=30)',yerr=np.vstack([conf-conf_lo,conf_hi-conf]),capsize=4)
ax.bar(xv+w/2,gen,w,label='Within-resource generalization cohort (N=30)',yerr=np.vstack([gen-gen_lo,gen_hi-gen]),capsize=4,hatch='//')
ax.axhline(.5,ls='--',lw=1,label='Pre-specified FULL reliability floor')
ax.set_xticks(xv,labels); ax.set_ylim(0,1.02); ax.set_ylabel('Mean participant cross-run RDM reliability, R')
ax.set_title('Full complex and exact component reliabilities')
ax.legend(fontsize=9,loc='lower left'); fig.tight_layout(); fig.savefig(out/'fig2_primary_two_cohorts.pdf'); plt.close(fig)

# Fig 3
labels=['GFT\n(original coordinates)','OOB-PCA\n(PCA fit on unused run)','Rank-matched sensor\nreconstruction']
conf=np.array([.14222,.1020,.1434]); gen=np.array([.16902,.1658,.1163])
fig,ax=plt.subplots(figsize=(9.6,5.3)); xv=np.arange(3); w=.34
ax.bar(xv-w/2,conf,w,label='Confirmation cohort (N=30)')
ax.bar(xv+w/2,gen,w,label='Generalization cohort (N=30)',hatch='//')
ax.axhline(0,lw=.8)
ax.set_xticks(xv,labels); ax.set_ylabel('Mean reliability difference: R(AP) - R(AMP)'); ax.set_title('AP advantage is robust to alternative coordinate systems')
ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out/'fig3_basis_robustness.pdf'); plt.close(fig)

# Fig 4
labels=['ENERGY\nradial sum','PRODUCT\namplitude product','INTERFERENCE\nweighted phase term','ANGLE\namplitude-free phase term']
conf=np.array([.790,.770,.743,.684]); gen=np.array([.719,.710,.588,.495])
conf_lo=np.array([.720,.697,.663,.593]); conf_hi=np.array([.844,.829,.813,.762])
gen_lo=np.array([.656,.646,.488,.397]); gen_hi=np.array([.774,.767,.682,.588])
fig,ax=plt.subplots(figsize=(9.8,5.4)); xv=np.arange(4); w=.34
ax.bar(xv-w/2,conf,w,label='Confirmation cohort',yerr=np.vstack([conf-conf_lo,conf_hi-conf]),capsize=4)
ax.bar(xv+w/2,gen,w,label='Generalization cohort',yerr=np.vstack([gen-gen_lo,gen_hi-gen]),capsize=4,hatch='//')
ax.set_xticks(xv,labels); ax.set_ylim(0,1.02); ax.set_ylabel('Mean participant cross-run RDM reliability, R'); ax.set_title('Radial and angular terms each contain reproducible structure')
ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out/'fig4_exact_components.pdf'); plt.close(fig)

# Fig 5
labels=['Confirmation\nINTERFERENCE','Generalization\nINTERFERENCE','Confirmation\nANGLE','Generalization\nANGLE']
obs=np.array([.7430,.5882,.6838,.4955]); null=np.array([.5310,.4832,-.0012,-.0010])
fig,ax=plt.subplots(figsize=(9.8,5.4)); xv=np.arange(4); w=.34
ax.bar(xv-w/2,obs,w,label='Observed category-phase organization')
ax.bar(xv+w/2,null,w,label='Amplitude-preserving phase-label null',hatch='//')
ax.axhline(0,lw=.8)
ax.set_xticks(xv,labels); ax.set_ylabel('Mean participant cross-run RDM reliability, R'); ax.set_title('Category-specific phase organization exceeds matched phase-label nulls (999 null realizations)')
ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out/'fig5_phase_null.pdf'); plt.close(fig)

# Fig 6
n=np.array([1,2,4,8,16,32,64],dtype=float); r=np.array([.008,.021,.075,.207,.404,.624,.757])
Rinf=.847; tau=26.36; xx=np.linspace(1,70,300); yy=Rinf*(1-np.exp(-xx/tau))
fig,ax=plt.subplots(figsize=(9.4,5.3)); ax.plot(n,r,'o-',label='Observed mean FULL reliability'); ax.plot(xx,yy,'--',label=r'Descriptive fit: $R_\infty=0.847$, convergence scale $\tau_n=26.36$ trials')
ax.set_xscale('log',base=2); ax.set_xticks(n,[str(int(v)) for v in n]); ax.set_ylim(-.02,.92); ax.set_xlabel('Repeated trials per category/run'); ax.set_ylabel('Mean FULL cross-run RDM reliability, R'); ax.set_title('Ensemble reliability increases with repeated-trial averaging (descriptive, not prescriptive)'); ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(out/'fig6_ensemble_convergence.pdf'); plt.close(fig)
