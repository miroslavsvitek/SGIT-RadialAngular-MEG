from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

from common import canonical_subject, save_json_atomic, utc_now
import level5a_frozen_dir as base


def graph_within_subject_compatible(root: Path, s: str):
    refs=[]
    for r in base.RUNS:
        _,_,gf=base.paths_for(root,s,r)
        if gf is None:
            return False,f'missing_graph_run{r}'
        refs.append(gf)
    z0=np.load(refs[0],allow_pickle=True)
    for rr,g in zip(base.RUNS[1:],refs[1:]):
        z=np.load(g,allow_pickle=True)
        for k in ('channel_names','coords','eigenvalues','eigenvectors'):
            if k not in z0.files or k not in z.files:
                return False,f'run{rr}_missing_{k}'
            a,b=z0[k],z[k]
            if a.shape!=b.shape:
                return False,f'run{rr}_{k}_shape'
            if a.dtype.kind in 'USO':
                if not np.array_equal(a,b): return False,f'run{rr}_{k}_differs'
            elif not np.allclose(a,b,rtol=0,atol=1e-7):
                return False,f'run{rr}_{k}_differs'
    return True,'OK'


def preflight_siteadapted(root: Path, subjects: list[str]):
    rows=[];eligible=[]
    for s0 in subjects:
        s=canonical_subject(s0);reasons=[]
        if s in base.DISCOVERY: reasons.append('discovery_subject_excluded')
        for r in base.RUNS:
            ff,mf,gf=base.paths_for(root,s,r)
            if not ff: reasons.append(f'missing_fine_run{r}')
            if not mf: reasons.append(f'missing_metadata_run{r}')
            if not gf: reasons.append(f'missing_graph_run{r}')
            if ff:
                with np.load(ff,allow_pickle=False) as z:
                    if 'gft_trajectories' not in z.files or 'times_s' not in z.files: reasons.append(f'invalid_fine_keys_run{r}')
                    else:
                        X=z['gft_trajectories']
                        if X.ndim!=3 or X.shape[1]!=base.K: reasons.append(f'bad_K_or_shape_run{r}:{X.shape}')
                        if not np.isfinite(X).all(): reasons.append(f'nonfinite_trajectory_run{r}')
            if mf:
                meta=pd.read_csv(mf)
                if 'sgit_category' not in meta.columns: reasons.append(f'missing_category_run{r}')
                else:
                    vc=meta.sgit_category.astype(str).value_counts()
                    for c in base.CATS:
                        n=int(vc.get(c,0))
                        if n<base.MIN_TRIALS: reasons.append(f'run{r}_{c}_trials={n}<32')
        gok,gwhy=graph_within_subject_compatible(root,s)
        if not gok: reasons.append('within_subject_graph:'+gwhy)
        reasons=sorted(set(reasons));ok=not reasons
        if ok:eligible.append(s)
        rows.append({'subject':s,'eligible':ok,'reasons':reasons})
    return eligible,rows


def run_final_siteadapted(root: Path, subjects: list[str], outdir: Path):
    subjects=[canonical_subject(s) for s in subjects]
    eligible,rows=preflight_siteadapted(root,subjects)
    outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'subject':r['subject'],'eligible':r['eligible'],'reasons':' ; '.join(r['reasons'])} for r in rows]).to_csv(outdir/'LEVEL6A_L5_siteadapted_preflight.csv',index=False)
    if eligible!=subjects:
        raise RuntimeError('Site-adapted Level-5A list contains subject failing frozen technical QC.')
    if len(subjects)<base.N_MIN:
        status={'level':'6A-L5-siteadapted','status':'INSUFFICIENT_NEW_DATA','N_eligible':len(subjects),'N_min':base.N_MIN}
        save_json_atomic(outdir/'LEVEL6A_L5_siteadapted_status.json',status);return status
    if len(subjects)>base.N_TARGET:
        raise RuntimeError(f'cohort has {len(subjects)} > frozen target {base.N_TARGET}')

    parts=base.partitions_2v2();subject_results={};canonical={};native={};recon_max=0.0
    for idx,s in enumerate(subjects,1):
        print(f'Level-6A site-adapted L5 {idx}/{len(subjects)}: {s}',flush=True)
        store=base.build_subject(root,s);vals={k:[] for k in ['FULL','AP','AMP']}
        for a,b in parts:
            da=base.decomposed_rdm(base.mean_y(store,a));db=base.decomposed_rdm(base.mean_y(store,b))
            recon_max=max(recon_max,float(np.max(np.abs(da['FULL']-(da['AMP']+da['AP'])))),float(np.max(np.abs(db['FULL']-(db['AMP']+db['AP'])))))
            for k in vals: vals[k].append(base.rho(da[k],db[k]))
        subject_results[s]={k:float(np.mean(v)) for k,v in vals.items()}
        ca=base.decomposed_rdm(base.mean_y(store,[1,2]));cb=base.decomposed_rdm(base.mean_y(store,[4,5]))
        canonical[s]={k:base.rho(ca[k],cb[k]) for k in vals};native[s]=base.decomposed_rdm(base.mean_y(store,base.RUNS))
    full=np.array([subject_results[s]['FULL'] for s in subjects]);apv=np.array([subject_results[s]['AP'] for s in subjects]);amp=np.array([subject_results[s]['AMP'] for s in subjects])
    pfull,modefull,Bfull=base.signflip(full);delta=apv-amp;pdel,moded,Bdel=base.signflip(delta)
    primary_pass=bool(pfull<base.ALPHA and full.mean()>=base.RHO_FLOOR)
    ap_led=bool(primary_pass and delta.mean()>=base.DELTA_AP_AMP_FLOOR and pdel<base.ALPHA and np.mean(full-apv)<=base.FULL_AP_CEILING)
    sem={}
    for k in ['FULL','AP','AMP']:
        obs,p,b=base.semantic_mc([native[s][k] for s in subjects]);sem[k]={'mean_pairwise_subject_rho':obs,'mc_p':p,'B':b}
    result={
      'level':'6A-L5-siteadapted','status':'COMPLETED_PREDECLARED_SITE_ADAPTED_REPLICATION','completed_utc':utc_now(),'N_new_evaluable':len(subjects),'subjects':subjects,
      'adaptation':'same K=16, bands, window, rank-2 residualization, distances, 15 partitions and decision thresholds as Level-5A; graph/eigenbasis must be stable within subject but need not equal the CA discovery template',
      'primary':{'mean_FULL_all2v2_rho':float(full.mean()),'median':float(np.median(full)),'positive_fraction':float(np.mean(full>0)),'signflip_p_one_sided':pfull,'signflip_mode':modefull,'signflip_B':Bfull,'effect_floor':base.RHO_FLOOR,'PASS':primary_pass},
      'hierarchical_AP_led':{'mean_AP_rho':float(apv.mean()),'mean_AMP_rho':float(amp.mean()),'mean_AP_minus_AMP':float(delta.mean()),'delta_p_one_sided':pdel,'delta_signflip_mode':moded,'delta_signflip_B':Bdel,'mean_FULL_minus_AP':float(np.mean(full-apv)),'PASS':ap_led,'interpretable':primary_pass},
      'exact_reconstruction_max_abs_error':recon_max,'subject_all2v2':subject_results,'canonical_12vs45':canonical,'cross_subject_semantic_secondary':sem,
      'guardrail':'This is a predeclared acquisition-adapted replication, not an exact graph-template replication and not a consciousness-specific test.'}
    save_json_atomic(outdir/'LEVEL6A_L5_SITEADAPTED_RESULTS.json',result)
    pd.DataFrame([{'subject':s,**subject_results[s],**{f'canonical_{k}':canonical[s][k] for k in canonical[s]}} for s in subjects]).to_csv(outdir/'LEVEL6A_L5_SITEADAPTED_subject_results.csv',index=False)
    return result
