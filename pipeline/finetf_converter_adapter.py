from __future__ import annotations

import importlib.util, json, math, re, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

from common import canonical_subject, sha256_file, save_json_atomic

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor' / 'sgit_cogitate_converter_base_v1_2.py'
TARGET_SFREQ = 250.0
TMIN, TMAX = -1.50, 1.80
N_TIMES = 825
BASELINE = (-0.20, 0.0)
K_MODES = 16


def _load_base():
    spec = importlib.util.spec_from_file_location('sgit_cogitate_converter_base_v1_2_fine', VENDOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Nelze načíst converter {VENDOR}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stimulus_mask_and_metadata(df: pd.DataFrame, meta: pd.DataFrame):
    mask = np.zeros(len(df), dtype=bool)
    meta = meta.copy()
    for c in ('sgit_category','sgit_identity','sgit_orientation','sgit_duration','sgit_relevance'):
        meta[c] = ''
    tt = df['trial_type'].astype(str) if 'trial_type' in df.columns else pd.Series(['']*len(df))
    for q, val in enumerate(tt):
        mt = re.match(r'^(face|object|letter|false)(\d+)$', str(val).strip())
        if not mt:
            continue
        ori = dur = rel = ''
        for off in range(1, 8):
            if q + off >= len(df):
                break
            nxt = str(tt.iloc[q+off]).strip().lower()
            if nxt in ('left','right','center') and not ori:
                ori = nxt
            if re.fullmatch(r'(500|1000|1500)ms', nxt) and not dur:
                dur = nxt
            if nxt.startswith('task ') and not rel:
                rel = nxt
        mask[q] = True
        meta.loc[q,'sgit_category'] = mt.group(1)
        meta.loc[q,'sgit_identity'] = mt.group(0)
        meta.loc[q,'sgit_orientation'] = ori
        meta.loc[q,'sgit_duration'] = dur
        meta.loc[q,'sgit_relevance'] = rel
    return mask, meta


def _resample_exact(data: np.ndarray, native_sfreq: float) -> np.ndarray:
    if data.ndim != 2 or data.shape[1] < 16:
        raise RuntimeError('Neočekávaný tvar epochy.')
    if abs(native_sfreq - TARGET_SFREQ) < 1e-9 and data.shape[1] == N_TIMES:
        return data.astype(np.float64, copy=False)
    return signal.resample(data, N_TIMES, axis=1).astype(np.float64, copy=False)


def _stem(ent: dict[str,str]) -> str:
    return f"sub-{ent.get('subject') or 'NA'}_ses-{ent.get('session') or 'NA'}_task-{ent.get('task') or 'NA'}_run-{ent.get('run') or 'NA'}"


def _process_run(base, events_path: Path, bids_root: Path, out_root: Path, logger):
    ent = base.parse_bids_entities(events_path)
    df = base.read_events(events_path)
    meta = base.sanitize_metadata(df, ent, 'meeg', events_path)
    stim, meta = _stimulus_mask_and_metadata(df, meta)
    if int(stim.sum()) == 0:
        raise RuntimeError(f'{events_path.name}: žádné visual stimulus trialy')

    raw, bp = base.load_raw_for_events(events_path, bids_root, 'meeg', logger)
    try:
        picks, pick_kind = base.choose_ephys_picks(raw, 'meeg')
        if len(picks) < 3:
            raise RuntimeError('Příliš málo MEEG kanálů.')
        sfreq = float(raw.info['sfreq'])
        A, graph_source, coords = base.graph_from_positions_or_functional(raw, picks, 6, logger)
        L, evals, U = base.gft_basis(A, K_MODES)
        if U.shape[1] != K_MODES:
            raise RuntimeError(f'GFT K={U.shape[1]} místo 16')
        run_scale = np.asarray(base.estimate_run_channel_scale(raw, picks), dtype=np.float64)
        bad = ~np.isfinite(run_scale) | (run_scale <= 0)
        if np.any(bad):
            good = run_scale[np.isfinite(run_scale) & (run_scale > 0)]
            run_scale[bad] = float(np.median(good)) if good.size else 1.0
        ch_names = [raw.ch_names[int(i)] for i in picks]
        times = TMIN + np.arange(N_TIMES, dtype=np.float64) / TARGET_SFREQ
        bmask = (times >= BASELINE[0]) & (times < BASELINE[1])
        duration_s = raw.n_times / sfreq

        X_list, kept = [], []
        stim_idx = np.flatnonzero(stim)
        for jj, ridx in enumerate(stim_idx, 1):
            if jj == 1 or jj % 50 == 0 or jj == len(stim_idx):
                logger.log(f'      FineTF trial {jj}/{len(stim_idx)}')
            row = df.iloc[int(ridx)]
            onset = float(row['onset']) if 'onset' in row and pd.notna(row['onset']) else math.nan
            if not np.isfinite(onset):
                continue
            start_s, stop_s = onset + TMIN, onset + TMAX
            if start_s < 0 or stop_s > duration_s:
                continue
            s0, s1 = int(round(start_s*sfreq)), int(round(stop_s*sfreq))
            data = raw.get_data(picks=picks, start=s0, stop=s1).astype(np.float64, copy=False)
            if data.shape[1] < 16:
                continue
            data = _resample_exact(data, sfreq)
            baseline = np.nanmean(data[:, bmask], axis=1, keepdims=True)
            data = np.nan_to_num(data-baseline, nan=0.0, posinf=0.0, neginf=0.0)
            data = data / run_scale[:,None]
            modes = U.T @ data
            if modes.shape != (K_MODES, N_TIMES) or not np.all(np.isfinite(modes)):
                continue
            X_list.append(modes.astype(np.float32))
            kept.append(int(ridx))
        if not X_list:
            raise RuntimeError('Žádný trial s kompletní FineTF epochou.')

        X = np.stack(X_list)
        mout = meta.iloc[kept].reset_index(drop=True)
        mout['trial_uid'] = base.make_trial_uids(mout)
        mout['derivative_quality'] = 'fine_real_gft_trajectory'
        mout['channel_selection'] = pick_kind
        stem = _stem(ent)
        (out_root/'meeg_fine').mkdir(parents=True, exist_ok=True)
        (out_root/'graphs').mkdir(parents=True, exist_ok=True)
        (out_root/'metadata').mkdir(parents=True, exist_ok=True)
        npz_path = out_root/'meeg_fine'/f'{stem}_sgit_level4e_finetf.npz'
        np.savez_compressed(
            npz_path,
            trial_uid=np.asarray(mout['trial_uid'].astype(str),dtype='U'),
            gft_trajectories=X,
            times_s=times.astype(np.float32), sfreq_hz=np.float32(TARGET_SFREQ),
            epoch_s=np.asarray([TMIN,TMAX],dtype=np.float32), baseline_s=np.asarray(BASELINE,dtype=np.float32),
            k_modes=np.int32(K_MODES), channel_names=np.asarray(ch_names,dtype='U'),
            channel_coords=np.asarray(coords,dtype=np.float32), graph_source=np.asarray(graph_source),
            channel_selection=np.asarray(pick_kind), run_scale=np.asarray(run_scale,dtype=np.float32),
            format_version=np.asarray('SGIT_LEVEL4E_FINETF_1.1')
        )
        graph_path = out_root/'graphs'/f'{stem}_meeg_graph.npz'
        np.savez_compressed(
            graph_path, adjacency=np.asarray(A,dtype=np.float32), laplacian=np.asarray(L,dtype=np.float32),
            eigenvalues=np.asarray(evals,dtype=np.float32), eigenvectors=np.asarray(U,dtype=np.float32),
            coords=np.asarray(coords,dtype=np.float32), channel_names=np.asarray(ch_names,dtype='U'),
            graph_source=np.asarray(graph_source), channel_selection=np.asarray(pick_kind)
        )
        mout.to_csv(out_root/'metadata'/f'{stem}_trials.csv', index=False)
        return {'subject':canonical_subject(ent.get('subject','')), 'run':int(ent.get('run') or 0),
                'n_trials':int(X.shape[0]), 'shape':list(map(int,X.shape)), 'n_channels':len(picks),
                'native_sfreq_hz':sfreq, 'graph_source':str(graph_source), 'fine_file':npz_path.name,
                'graph_file':graph_path.name}
    finally:
        try: raw.close()
        except Exception: pass


def validate_finetf(root: Path, subject: str, required_runs=(1,2,3,4,5), min_trials_per_category=1) -> dict:
    s = canonical_subject(subject)
    info={'subject':s,'format':'SGIT_LEVEL4E_FINETF_1.1','runs':{},'n_trials':0,'level5_qc':True,'level5_reasons':[],'files':[]}
    for run in required_runs:
        fs=list((root/'meeg_fine').glob(f'sub-{s}_*run-{run:02d}_sgit_level4e_finetf.npz'))
        ms=list((root/'metadata').glob(f'sub-{s}_*run-{run:02d}_trials.csv'))
        gs=list((root/'graphs').glob(f'sub-{s}_*run-{run:02d}_meeg_graph.npz'))
        if len(fs)!=1 or len(ms)!=1 or len(gs)!=1:
            raise RuntimeError(f'{s} run {run}: chybí FineTF/metadata/graph ({len(fs)}/{len(ms)}/{len(gs)})')
        with np.load(fs[0],allow_pickle=False) as z:
            X=np.asarray(z['gft_trajectories']); times=np.asarray(z['times_s'],float)
            if X.ndim!=3 or X.shape[1:]!=(16,825):
                raise RuntimeError(f'{s} run {run}: FineTF shape {X.shape} místo [trial,16,825]')
            if not np.all(np.isfinite(X)):
                raise RuntimeError(f'{s} run {run}: FineTF obsahuje NaN/Inf')
            if abs(float(z['sfreq_hz'])-250)>1e-6 or not np.allclose(times[:3],[-1.5,-1.496,-1.492],atol=1e-5):
                raise RuntimeError(f'{s} run {run}: neočekávané časování FineTF')
        meta=pd.read_csv(ms[0])
        if len(meta)!=X.shape[0]:
            raise RuntimeError(f'{s} run {run}: metadata {len(meta)} != FineTF trials {X.shape[0]}')
        if 'sgit_category' not in meta.columns:
            raise RuntimeError(f'{s} run {run}: chybí sgit_category')
        vc=meta.sgit_category.astype(str).value_counts()
        counts={c:int(vc.get(c,0)) for c in ('face','object','letter','false')}
        for c,n in counts.items():
            if n < int(min_trials_per_category):
                info['level5_qc']=False; info['level5_reasons'].append(f'run{run}_{c}_trials={n}<{min_trials_per_category}')
        info['runs'][str(run)]={'n_trials':int(X.shape[0]),'category_counts':counts,'fine_file':fs[0].name,'metadata_file':ms[0].name,'graph_file':gs[0].name}
        info['n_trials'] += int(X.shape[0])
        for p in (fs[0],ms[0],gs[0]):
            info['files'].append({'path':str(p.relative_to(root)),'size':p.stat().st_size,'sha256':sha256_file(p)})
    return info


def merge_subject_finetf(src: Path, cohort_root: Path, subject: str):
    cohort_root.mkdir(parents=True,exist_ok=True)
    s=canonical_subject(subject)
    for subdir in ('meeg_fine','graphs','metadata'):
        (cohort_root/subdir).mkdir(parents=True,exist_ok=True)
        for p in (src/subdir).glob(f'sub-{s}_*'):
            if p.is_file(): shutil.copy2(p,cohort_root/subdir/p.name)


def convert_subject(bids_root: Path, output_dir: Path, cohort_root: Path, subject: str,
                    required_runs=(1,2,3,4,5), level5_min_trials=32) -> dict:
    base=_load_base()
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    logger=base.Logger(output_dir/'FINETF_CONVERTER_LOG.txt')
    try:
        events=[p for p,m in base.scan_events(bids_root,logger) if m=='meeg' and canonical_subject(base.parse_bids_entities(p).get('subject',''))==canonical_subject(subject)]
        byrun={}
        for p in events:
            ent=base.parse_bids_entities(p)
            if str(ent.get('task','')).lower()!='dur': continue
            try:r=int(ent.get('run') or 0)
            except Exception:r=0
            if r in required_runs: byrun.setdefault(r,[]).append(p)
        for r in required_runs:
            if len(byrun.get(r,[]))!=1:
                raise RuntimeError(f'{subject}: pro task-dur run {r} očekávám 1 events.tsv, nalezeno {len(byrun.get(r,[]))}')
        records=[]
        for r in required_runs:
            logger.log(f'FineTF {subject}: run {r}/5')
            records.append(_process_run(base,byrun[r][0],bids_root,output_dir,logger))
    finally:
        try: logger._f.close()
        except Exception: pass
    info=validate_finetf(output_dir,subject,required_runs,level5_min_trials)
    merge_subject_finetf(output_dir,cohort_root,subject)
    (cohort_root/'subjects').mkdir(parents=True,exist_ok=True)
    save_json_atomic(cohort_root/'subjects'/f'{canonical_subject(subject)}_finetf_manifest.json',info)
    return info
