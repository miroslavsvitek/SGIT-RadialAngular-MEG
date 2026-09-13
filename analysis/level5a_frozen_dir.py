from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import rankdata

from common import load_json, save_json_atomic, canonical_subject, utc_now

DISCOVERY={'CA124','CA140','CB013','CB072'}
CATS=['face','object','letter','false']; RUNS=[1,2,3,4,5]
FS=250.0; K=16; RANK=2; MIN_TRIALS=32; N_MIN=20; N_TARGET=30
ALPHA=0.025; RHO_FLOOR=0.50; DELTA_AP_AMP_FLOOR=0.10; FULL_AP_CEILING=0.10; SEED=20260820
TRIU=np.triu_indices(4,1)


def rho(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ra=rankdata(a,method='average'); rb=rankdata(b,method='average')
    ra-=ra.mean(); rb-=rb.mean(); den=np.linalg.norm(ra)*np.linalg.norm(rb)
    return float(np.dot(ra,rb)/den) if den else np.nan


def signflip(vals,seed=SEED):
    vals=np.asarray(vals,float); vals=vals[np.isfinite(vals)]; obs=float(vals.mean()); n=len(vals)
    if n<=20:
        ge=0; tot=0
        for sg in itertools.product([-1,1],repeat=n):
            tot+=1
            if np.mean(vals*np.asarray(sg))>=obs-1e-15: ge+=1
        return ge/tot,'exact',tot
    rng=np.random.default_rng(seed); B=100000; ge=0
    for start in range(0,B,5000):
        m=min(5000,B-start); signs=rng.choice(np.array([-1.,1.]),size=(m,n)); null=(signs*vals).mean(axis=1)
        ge+=int(np.sum(null>=obs-1e-15))
    return ge/B,'monte_carlo',B


def partitions_2v2():
    out=[]; seen=set()
    for a in itertools.combinations(RUNS,2):
        rem=[x for x in RUNS if x not in a]
        for b in itertools.combinations(rem,2):
            key=tuple(sorted([tuple(a),tuple(b)]))
            if key not in seen: seen.add(key); out.append((a,b))
    assert len(out)==15
    return out


def paths_for(root:Path,s:str,r:int):
    fine=list((root/'meeg_fine').glob(f'sub-{s}_*run-{r:02d}_sgit_level4e_finetf.npz'))
    meta=list((root/'metadata').glob(f'sub-{s}_*run-{r:02d}_trials.csv'))
    graph=list((root/'graphs').glob(f'sub-{s}_*run-{r:02d}_meeg_graph.npz'))
    return (fine[0] if len(fine)==1 else None, meta[0] if len(meta)==1 else None, graph[0] if len(graph)==1 else None)


def graph_compatible(path:Path,template:Path):
    z=np.load(path,allow_pickle=True); t=np.load(template,allow_pickle=True)
    keys=['channel_names','coords','eigenvalues','eigenvectors']
    for k in keys:
        if k not in z.files or k not in t.files:return False,f'missing graph key {k}'
        a=z[k]; b=t[k]
        if a.shape!=b.shape:return False,f'{k} shape {a.shape}!={b.shape}'
        if a.dtype.kind in 'USO':
            if not np.array_equal(a,b):return False,f'{k} differs'
        else:
            if not np.allclose(a,b,rtol=0,atol=1e-7):return False,f'{k} differs'
    return True,'ok'


def preflight(root:Path,subjects:list[str],template:Path):
    rows=[]; eligible=[]
    for s in subjects:
        reasons=[]
        if s in DISCOVERY: reasons.append('discovery_subject_excluded')
        for r in RUNS:
            ff,mf,gf=paths_for(root,s,r)
            if not ff: reasons.append(f'missing_fine_run{r}')
            if not mf: reasons.append(f'missing_metadata_run{r}')
            if not gf: reasons.append(f'missing_graph_run{r}')
            if ff:
                with np.load(ff,allow_pickle=False) as z:
                    if 'gft_trajectories' not in z.files or 'times_s' not in z.files: reasons.append(f'invalid_fine_keys_run{r}')
                    else:
                        X=z['gft_trajectories'];
                        if X.ndim!=3 or X.shape[1]!=K: reasons.append(f'bad_K_or_shape_run{r}:{X.shape}')
                        if not np.isfinite(X).all(): reasons.append(f'nonfinite_trajectory_run{r}')
            if mf:
                meta=pd.read_csv(mf)
                if 'sgit_category' not in meta.columns: reasons.append(f'missing_category_run{r}')
                else:
                    vc=meta.sgit_category.astype(str).value_counts()
                    for c in CATS:
                        n=int(vc.get(c,0))
                        if n<MIN_TRIALS: reasons.append(f'run{r}_{c}_trials={n}<32')
            if gf and template.exists():
                ok,msg=graph_compatible(gf,template)
                if not ok: reasons.append(f'graph_run{r}:{msg}')
        reasons=sorted(set(reasons)); ok=not reasons
        if ok:eligible.append(s)
        rows.append({'subject':s,'eligible':ok,'reasons':reasons})
    return eligible,rows


def build_subject(root:Path,s:str):
    sos_t=butter(4,[4,8],btype='bandpass',fs=FS,output='sos')
    sos_a=butter(4,[8,13],btype='bandpass',fs=FS,output='sos')
    out={}
    for r in RUNS:
        ff,mf,_=paths_for(root,s,r)
        with np.load(ff,allow_pickle=False) as npz:
            raw=npz['gft_trajectories'].astype(np.float64); times=npz['times_s'].astype(float)
        tm=(times>=.05)&(times<.25)
        zt=hilbert(sosfiltfilt(sos_t,raw,axis=-1),axis=-1)[:,:,tm]
        za=hilbert(sosfiltfilt(sos_a,raw,axis=-1),axis=-1)[:,:,tm]
        meta=pd.read_csv(mf); cc=meta.sgit_category.astype(str).to_numpy()
        Et=[]; Ea=[]; Pa=[]; Xcats=[]
        for c in CATS:
            ix=np.where(cc==c)[0]
            et=zt[ix].mean(0); ea=za[ix].mean(0)
            Et.append(et); Ea.append(ea); Pa.append(np.mean(np.abs(za[ix])**2,axis=(0,2)))
            Xcats.append(np.concatenate([et.T,ea.T],axis=1))
        G=np.mean(np.asarray(Xcats),axis=0); _,_,Vh=np.linalg.svd(G,full_matrices=False)
        V=Vh.conj().T[:,:RANK]; M=np.eye(2*K,dtype=np.complex128)-V@V.conj().T
        Ya=[]
        for ci in range(4):
            X=np.concatenate([Et[ci].T,Ea[ci].T],axis=1)@M
            ra=X[:,K:].T; Ya.append(ra/np.sqrt(np.maximum(Pa[ci],1e-30))[:,None])
        out[r]=np.stack(Ya)
    return out


def decomposed_rdm(y):
    A=np.abs(y); phi=np.angle(y); amp=[]; ap=[]; full=[]
    for i,j in zip(*TRIU):
        amp.append(float(np.sum((A[i]-A[j])**2)))
        ap.append(float(np.sum(2*A[i]*A[j]*(1-np.cos(phi[i]-phi[j])))))
        full.append(float(np.sum(np.abs(y[i]-y[j])**2)))
    return {'AMP':np.asarray(amp),'AP':np.asarray(ap),'FULL':np.asarray(full)}


def mean_y(store,rs):return np.mean([store[r] for r in rs],axis=0)


def semantic_mc(native_rdms,B=10000,seed=SEED):
    n=len(native_rdms)
    if n<2:return np.nan,np.nan,0
    def stat(rdms):return float(np.mean([rho(rdms[i],rdms[j]) for i,j in itertools.combinations(range(n),2)]))
    obs=stat(native_rdms); perms=list(itertools.permutations(range(4))); mats=[]
    for v in native_rdms:
        D=np.zeros((4,4));D[TRIU]=v;D[(TRIU[1],TRIU[0])]=v;mats.append(D)
    rng=np.random.default_rng(seed);ge=0
    for _ in range(B):
        rr=[native_rdms[0]]
        for si in range(1,n):
            p=perms[int(rng.integers(24))];rr.append(mats[si][np.ix_(p,p)][TRIU])
        if stat(rr)>=obs-1e-15:ge+=1
    return obs,(ge+1)/(B+1),B


def run_final(root:Path,subjects:list[str],template:Path,outdir:Path):
    subjects=[canonical_subject(s) for s in subjects]
    eligible,rows=preflight(root,subjects,template)
    outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'subject':r['subject'],'eligible':r['eligible'],'reasons':' ; '.join(r['reasons'])} for r in rows]).to_csv(outdir/'LEVEL5A_preflight.csv',index=False)
    save_json_atomic(outdir/'LEVEL5A_preflight.json',{'subjects_requested':subjects,'eligible':eligible,'rows':rows,'N_min':N_MIN,'N_target':N_TARGET})
    if eligible!=subjects:
        raise RuntimeError('Level-5A analysis list obsahuje subjekt, který neprošel ZMRAZENÝM QC. Opravte cohort lock, nikoli výsledky.')
    if len(subjects)<N_MIN:
        status={'level':'5A','status':'INSUFFICIENT_NEW_DATA','N_eligible':len(subjects),'N_min':N_MIN,'confirmatory_outcome_computed':False}
        save_json_atomic(outdir/'LEVEL5A_status.json',status);return status
    if len(subjects)>N_TARGET:
        raise RuntimeError(f'Level-5A cohort má {len(subjects)} > frozen target {N_TARGET}. Použijte zmrazených prvních 30.')
    parts=partitions_2v2(); subject_results={};canonical={};native={};recon_max=0.0
    for idx,s in enumerate(subjects,1):
        print(f'Level-5A {idx}/{len(subjects)}: {s}',flush=True);store=build_subject(root,s)
        vals={k:[] for k in ['FULL','AP','AMP']}
        for a,b in parts:
            da=decomposed_rdm(mean_y(store,a));db=decomposed_rdm(mean_y(store,b))
            recon_max=max(recon_max,float(np.max(np.abs(da['FULL']-(da['AMP']+da['AP'])))),float(np.max(np.abs(db['FULL']-(db['AMP']+db['AP'])))))
            for k in vals:vals[k].append(rho(da[k],db[k]))
        subject_results[s]={k:float(np.mean(v)) for k,v in vals.items()}
        ca=decomposed_rdm(mean_y(store,[1,2]));cb=decomposed_rdm(mean_y(store,[4,5]));canonical[s]={k:rho(ca[k],cb[k]) for k in vals}
        native[s]=decomposed_rdm(mean_y(store,RUNS))
    full=np.array([subject_results[s]['FULL'] for s in subjects]);apv=np.array([subject_results[s]['AP'] for s in subjects]);amp=np.array([subject_results[s]['AMP'] for s in subjects])
    pfull,modefull,Bfull=signflip(full);delta=apv-amp;pdel,moded,Bdel=signflip(delta)
    primary_pass=bool(pfull<ALPHA and full.mean()>=RHO_FLOOR)
    ap_led=bool(primary_pass and delta.mean()>=DELTA_AP_AMP_FLOOR and pdel<ALPHA and np.mean(full-apv)<=FULL_AP_CEILING)
    sem={}
    for k in ['FULL','AP','AMP']:
        obs,p,b=semantic_mc([native[s][k] for s in subjects]);sem[k]={'mean_pairwise_subject_rho':obs,'mc_p':p,'B':b}
    result={
      'level':'5A','status':'COMPLETED_CONFIRMATORY_TEST','completed_utc':utc_now(),'N_new_evaluable':len(subjects),'subjects':subjects,
      'frozen':{'K':16,'fs':250.0,'theta_hz':[4,8],'alpha_hz':[8,13],'window_ms':[50,250],'rank':2,'min_trials_category_run':32,'alpha':ALPHA},
      'primary':{'mean_FULL_all2v2_rho':float(full.mean()),'median':float(np.median(full)),'positive_fraction':float(np.mean(full>0)),'signflip_p_one_sided':pfull,'signflip_mode':modefull,'signflip_B':Bfull,'effect_floor':RHO_FLOOR,'PASS':primary_pass},
      'hierarchical_AP_led':{'mean_AP_rho':float(apv.mean()),'mean_AMP_rho':float(amp.mean()),'mean_AP_minus_AMP':float(delta.mean()),'delta_p_one_sided':pdel,'delta_signflip_mode':moded,'delta_signflip_B':Bdel,'mean_FULL_minus_AP':float(np.mean(full-apv)),'PASS':ap_led,'interpretable':primary_pass},
      'exact_reconstruction_max_abs_error':recon_max,'subject_all2v2':subject_results,'canonical_12vs45':canonical,'cross_subject_semantic_secondary':sem,
      'guardrail':'Coarse stimulus-locked early-alpha ensemble geometry only; not consciousness specificity, causality, PAC, or fine-identity code.'}
    save_json_atomic(outdir/'SGIT_Level5A_results.json',result)
    pd.DataFrame([{'subject':s,**subject_results[s],**{f'canonical_{k}':canonical[s][k] for k in canonical[s]}} for s in subjects]).to_csv(outdir/'SGIT_Level5A_subject_results.csv',index=False)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort-root',required=True);ap.add_argument('--cohort-file',required=True);ap.add_argument('--template-graph',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();plan=load_json(a.cohort_file);subjects=plan['level5a_subjects']
    r=run_final(Path(a.cohort_root),subjects,Path(a.template_graph),Path(a.output));print(json.dumps(r,indent=2,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
