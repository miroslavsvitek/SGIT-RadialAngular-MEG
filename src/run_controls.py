from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

from common import (
    CATS, RUNS, assoc, bootstrap_mean_ci, decomposed_rdm, discover_finetf_root,
    fit_pca_rotation, graph_compatible, load_json, partitions_2v2, paths_for,
    preflight_subject, random_orthogonal, rotate_modes, save_json, sha256_file,
    signflip_one_sided, stable_subject_seed,
)

KEYS = ["FULL", "AP", "AMP"]


@dataclass
class RunData:
    categories: np.ndarray
    raw_post: np.ndarray
    zt_post: np.ndarray
    za_post: np.ndarray
    zt_pre: np.ndarray
    za_pre: np.ndarray


def log_factory(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(msg: str):
        s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(s, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(s + "\n")
    return log


def load_subject_windows(root: Path, subject: str, cfg: dict, log) -> dict[int, RunData]:
    fs = float(cfg["frozen_signal_pipeline"]["fs_hz"])
    theta = cfg["frozen_signal_pipeline"]["theta_hz"]
    alpha = cfg["frozen_signal_pipeline"]["alpha_hz"]
    post = cfg["frozen_signal_pipeline"]["post_window_s"]
    pre = cfg["frozen_signal_pipeline"]["far_prestim_window_s"]
    sos_t = butter(4, theta, btype="bandpass", fs=fs, output="sos")
    sos_a = butter(4, alpha, btype="bandpass", fs=fs, output="sos")
    out = {}
    for r in RUNS:
        ff, mf, _ = paths_for(root, subject, r)
        meta = pd.read_csv(mf)
        with np.load(ff, allow_pickle=False) as z:
            raw = z["gft_trajectories"].astype(np.float64)
            times = z["times_s"].astype(float)
        if raw.ndim != 3 or raw.shape[1] != int(cfg["frozen_signal_pipeline"]["K"]):
            raise RuntimeError(f"{subject} run {r}: bad FineTF shape {raw.shape}")
        if raw.shape[0] != len(meta):
            raise RuntimeError(f"{subject} run {r}: FineTF trial count does not match metadata")
        if not np.isfinite(raw).all() or not np.isfinite(times).all():
            raise RuntimeError(f"{subject} run {r}: non-finite FineTF data")
        mpost = (times >= float(post[0])) & (times < float(post[1]))
        mpre = (times >= float(pre[0])) & (times < float(pre[1]))
        if int(mpost.sum()) < 10 or int(mpre.sum()) < 10:
            raise RuntimeError(f"{subject} run {r}: missing requested windows; post={mpost.sum()}, pre={mpre.sum()}")
        raw_post = raw[:, :, mpost].copy()

        ft = sosfiltfilt(sos_t, raw, axis=-1)
        zt = hilbert(ft, axis=-1)
        zt_post = zt[:, :, mpost].copy()
        zt_pre = zt[:, :, mpre].copy()
        del ft, zt

        fa = sosfiltfilt(sos_a, raw, axis=-1)
        za = hilbert(fa, axis=-1)
        za_post = za[:, :, mpost].copy()
        za_pre = za[:, :, mpre].copy()
        del fa, za, raw

        cats = meta["sgit_category"].astype(str).to_numpy()
        if len(cats) != raw_post.shape[0]:
            raise RuntimeError(f"{subject} run {r}: metadata rows {len(cats)} != trials {raw_post.shape[0]}")
        out[r] = RunData(cats, raw_post, zt_post, za_post, zt_pre, za_pre)
    return out


def _category_statistics(zt: np.ndarray, za: np.ndarray, cats: np.ndarray, sensor_U: np.ndarray | None = None):
    Et, Ea, Pa, Xcats = [], [], [], []
    if sensor_U is None:
        p = zt.shape[1]
        for c in CATS:
            ix = np.where(cats == c)[0]
            et = zt[ix].mean(axis=0)
            ea = za[ix].mean(axis=0)
            pa = np.mean(np.abs(za[ix]) ** 2, axis=(0, 2))
            Et.append(et); Ea.append(ea); Pa.append(pa)
            Xcats.append(np.concatenate([et.T, ea.T], axis=1))
        return p, Et, Ea, Pa, Xcats

    # Rank-matched reconstructed sensor coordinates from the first K graph modes.
    U = np.asarray(sensor_U, float)
    p = U.shape[0]
    for c in CATS:
        ix = np.where(cats == c)[0]
        et_g = zt[ix].mean(axis=0)
        ea_g = za[ix].mean(axis=0)
        et = U @ et_g
        ea = U @ ea_g
        zz = za[ix]
        C = np.einsum("nkt,nlt->kl", zz, np.conj(zz), optimize=True) / float(zz.shape[0] * zz.shape[2])
        pa = np.einsum("sk,kl,sl->s", U, C, U, optimize=True).real
        pa = np.maximum(pa, 0.0)
        Et.append(et); Ea.append(ea); Pa.append(pa)
        Xcats.append(np.concatenate([et.T, ea.T], axis=1))
    return p, Et, Ea, Pa, Xcats


def build_field(rd: RunData, rank: int, window: str = "post", Q: np.ndarray | None = None, sensor_U: np.ndarray | None = None) -> np.ndarray:
    if window == "post":
        zt, za = rd.zt_post, rd.za_post
    elif window == "pre":
        zt, za = rd.zt_pre, rd.za_pre
    else:
        raise ValueError(window)
    if Q is not None:
        zt = rotate_modes(zt, Q)
        za = rotate_modes(za, Q)
    p, Et, Ea, Pa, Xcats = _category_statistics(zt, za, rd.categories, sensor_U=sensor_U)
    G = np.mean(np.asarray(Xcats), axis=0)
    _, _, Vh = np.linalg.svd(G, full_matrices=False)
    rr = min(rank, Vh.shape[0])
    V = Vh.conj().T[:, :rr]
    Ya = []
    for ci in range(4):
        X = np.concatenate([Et[ci].T, Ea[ci].T], axis=1)
        Xr = X - (X @ V) @ V.conj().T
        ra = Xr[:, p:].T
        ya = ra / np.sqrt(np.maximum(Pa[ci], 1e-30))[:, None]
        Ya.append(ya)
    return np.stack(Ya)


def reliability(run_fields: dict[int, np.ndarray], metric: str = "spearman"):
    vals = {k: [] for k in KEYS}
    half_pairs = []
    for a, b, unused in partitions_2v2():
        ya = np.mean([run_fields[r] for r in a], axis=0)
        yb = np.mean([run_fields[r] for r in b], axis=0)
        da = decomposed_rdm(ya); db = decomposed_rdm(yb)
        for k in KEYS:
            vals[k].append(assoc(da[k], db[k], metric))
        half_pairs.append((ya, yb, a, b, unused))
    return np.asarray([np.nanmean(vals[k]) for k in KEYS], float), half_pairs


def reliability_oob_pca(data: dict[int, RunData], rank: int, metric: str = "spearman"):
    vals = {k: [] for k in KEYS}
    for a, b, unused in partitions_2v2():
        Q = fit_pca_rotation(data[unused].raw_post)
        fa = [build_field(data[r], rank, "post", Q=Q) for r in a]
        fb = [build_field(data[r], rank, "post", Q=Q) for r in b]
        ya = np.mean(fa, axis=0); yb = np.mean(fb, axis=0)
        da = decomposed_rdm(ya); db = decomposed_rdm(yb)
        for k in KEYS:
            vals[k].append(assoc(da[k], db[k], metric))
    return np.asarray([np.nanmean(vals[k]) for k in KEYS], float)


def shuffle_category_phases(y: np.ndarray, rng: np.random.Generator):
    A = np.abs(y)
    P = np.angle(y)
    shp = P.shape
    Pf = P.reshape(4, -1)
    Af = A.reshape(4, -1)
    perm = np.argsort(rng.random(size=Pf.shape), axis=0)
    P2 = np.take_along_axis(Pf, perm, axis=0)
    z = (Af * np.exp(1j * P2)).reshape(shp)
    err = float(np.max(np.abs(np.abs(z) - A)))
    return z, err


def phase_null(half_pairs, observed_amp: float, B: int, seed: int):
    rng = np.random.default_rng(seed)
    null_full = np.empty(B, dtype=float)
    null_ap = np.empty(B, dtype=float)
    max_amp_err = 0.0
    for bi in range(B):
        vf, va = [], []
        for ya, yb, *_ in half_pairs:
            sa, ea = shuffle_category_phases(ya, rng)
            sb, eb = shuffle_category_phases(yb, rng)
            max_amp_err = max(max_amp_err, ea, eb)
            da = decomposed_rdm(sa); db = decomposed_rdm(sb)
            vf.append(assoc(da["FULL"], db["FULL"], "spearman"))
            va.append(assoc(da["AP"], db["AP"], "spearman"))
        null_full[bi] = np.nanmean(vf)
        null_ap[bi] = np.nanmean(va)
    null_delta = null_ap - float(observed_amp)
    return null_full, null_ap, null_delta, max_amp_err


def subject_cache_path(cache_root: Path, cohort_name: str, subject: str) -> Path:
    return cache_root / f"{cohort_name}__{subject}.npz"


def save_subject_cache(path: Path, protocol_hash: str, subject: str, obs_s: np.ndarray, obs_p: np.ndarray, obs_k: np.ndarray,
                       pre_s: np.ndarray, pca_s: np.ndarray, sensor_s: np.ndarray,
                       phase_full: np.ndarray, phase_ap: np.ndarray, phase_delta: np.ndarray,
                       rand_s: np.ndarray, amp_audit: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        protocol_hash=np.asarray(protocol_hash), subject=np.asarray(subject),
        obs_s=obs_s, obs_p=obs_p, obs_k=obs_k, pre_s=pre_s, pca_s=pca_s, sensor_s=sensor_s,
        phase_full=phase_full, phase_ap=phase_ap, phase_delta=phase_delta,
        rand_s=rand_s, amp_audit=np.asarray(amp_audit, float),
    )


def load_subject_cache(path: Path, protocol_hash: str):
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        h = str(z["protocol_hash"].item())
        if h != protocol_hash:
            return None
        return {k: z[k] for k in z.files}
    except Exception:
        return None


def run_subject(root: Path, subject: str, cfg: dict, sensor_U: np.ndarray, random_Qs: list[np.ndarray], protocol_hash: str,
                cache_path: Path, log):
    cached = load_subject_cache(cache_path, protocol_hash)
    if cached is not None:
        log(f"{subject}: cache OK, skipping recomputation")
        return cached

    rank = int(cfg["frozen_signal_pipeline"]["common_subspace_rank"])
    Bphase = int(cfg["controls"]["C1_phase_null"]["B"])
    seed_phase = int(cfg["controls"]["C1_phase_null"]["seed"]) + stable_subject_seed(subject)

    data = load_subject_windows(root, subject, cfg, log)

    # Identity post and prestimulus.
    identity_post = {r: build_field(data[r], rank, "post") for r in RUNS}
    identity_pre = {r: build_field(data[r], rank, "pre") for r in RUNS}
    obs_s, half_pairs = reliability(identity_post, "spearman")
    obs_p, _ = reliability(identity_post, "pearson")
    obs_k, _ = reliability(identity_post, "kendall")
    pre_s, _ = reliability(identity_pre, "spearman")

    # Amplitude-preserving phase null.
    phase_full, phase_ap, phase_delta, amp_audit = phase_null(half_pairs, obs_s[2], Bphase, seed_phase)

    # OOB PCA.
    pca_s = reliability_oob_pca(data, rank, "spearman")

    # Rank-matched sensor-coordinate baseline.
    sensor_fields = {r: build_field(data[r], rank, "post", sensor_U=sensor_U) for r in RUNS}
    sensor_s, _ = reliability(sensor_fields, "spearman")

    # Fixed random orthogonal rotations.
    rand_s = np.empty((len(random_Qs), 3), dtype=float)
    for qi, Q in enumerate(random_Qs):
        rf = {r: build_field(data[r], rank, "post", Q=Q) for r in RUNS}
        rand_s[qi], _ = reliability(rf, "spearman")
        if (qi + 1) % 10 == 0 or qi + 1 == len(random_Qs):
            log(f"{subject}: random basis {qi+1}/{len(random_Qs)}")

    save_subject_cache(cache_path, protocol_hash, subject, obs_s, obs_p, obs_k, pre_s, pca_s, sensor_s,
                       phase_full, phase_ap, phase_delta, rand_s, amp_audit)
    return load_subject_cache(cache_path, protocol_hash)


def _ci_record(vals, cfg, seed):
    m, lo, hi = bootstrap_mean_ci(vals, int(cfg["bootstrap_B"]), seed)
    return {"mean": m, "ci95": [lo, hi]}


def aggregate_cohort(cohort_name: str, subjects: list[str], caches: list[dict], cfg: dict, outdir: Path):
    obs_s = np.vstack([c["obs_s"] for c in caches])
    obs_p = np.vstack([c["obs_p"] for c in caches])
    obs_k = np.vstack([c["obs_k"] for c in caches])
    pre_s = np.vstack([c["pre_s"] for c in caches])
    pca_s = np.vstack([c["pca_s"] for c in caches])
    sensor_s = np.vstack([c["sensor_s"] for c in caches])
    phase_full = np.vstack([c["phase_full"] for c in caches])
    phase_ap = np.vstack([c["phase_ap"] for c in caches])
    phase_delta = np.vstack([c["phase_delta"] for c in caches])
    rand_s = np.stack([c["rand_s"] for c in caches], axis=0)  # N x B x 3
    amp_audit = max(float(c["amp_audit"].item()) for c in caches)

    seed0 = int(cfg["signflip_seed"]) + (10000 if cohort_name.startswith("CB") else 0)
    Bsf = int(cfg["signflip_B"])
    delta_obs = obs_s[:, 1] - obs_s[:, 2]
    delta_pca = pca_s[:, 1] - pca_s[:, 2]
    delta_sensor = sensor_s[:, 1] - sensor_s[:, 2]
    post_pre_full = obs_s[:, 0] - pre_s[:, 0]
    post_pre_ap = obs_s[:, 1] - pre_s[:, 1]

    summary = {
        "cohort": cohort_name,
        "N": len(subjects),
        "subjects": subjects,
        "keys": KEYS,
        "observed_spearman": {},
        "C1_phase_null": {},
        "C2_basis": {},
        "C3_prestim": {},
        "C4_metric": {},
        "audits": {"max_amplitude_error_phase_null": amp_audit},
    }

    for j, k in enumerate(KEYS):
        summary["observed_spearman"][k] = _ci_record(obs_s[:, j], cfg, seed0 + j)
    summary["observed_spearman"]["AP_minus_AMP"] = _ci_record(delta_obs, cfg, seed0 + 10)
    _, pdel, mode, B = signflip_one_sided(delta_obs, Bsf, seed0 + 11)
    summary["observed_spearman"]["AP_minus_AMP"].update({"signflip_p_one_sided": pdel, "mode": mode, "B": B})

    # C1 population null and paired comparison.
    pop_null_full = phase_full.mean(axis=0)
    pop_null_ap = phase_ap.mean(axis=0)
    pop_null_delta = phase_delta.mean(axis=0)
    obs_full_mean = float(obs_s[:, 0].mean()); obs_ap_mean = float(obs_s[:, 1].mean()); obs_delta_mean = float(delta_obs.mean())
    ep_full = (1 + int(np.sum(pop_null_full >= obs_full_mean - 1e-15))) / (len(pop_null_full) + 1)
    ep_ap = (1 + int(np.sum(pop_null_ap >= obs_ap_mean - 1e-15))) / (len(pop_null_ap) + 1)
    ep_delta = (1 + int(np.sum(pop_null_delta >= obs_delta_mean - 1e-15))) / (len(pop_null_delta) + 1)
    subj_null_full = phase_full.mean(axis=1); subj_null_ap = phase_ap.mean(axis=1)
    _, p_pair_full, _, _ = signflip_one_sided(obs_s[:,0] - subj_null_full, Bsf, seed0 + 20)
    _, p_pair_ap, _, _ = signflip_one_sided(obs_s[:,1] - subj_null_ap, Bsf, seed0 + 21)
    summary["C1_phase_null"] = {
        "B": int(phase_full.shape[1]),
        "observed_FULL": obs_full_mean,
        "null_FULL_mean": float(pop_null_full.mean()),
        "empirical_p_FULL": ep_full,
        "paired_signflip_p_FULL_gt_nullmean": p_pair_full,
        "observed_AP": obs_ap_mean,
        "null_AP_mean": float(pop_null_ap.mean()),
        "empirical_p_AP": ep_ap,
        "paired_signflip_p_AP_gt_nullmean": p_pair_ap,
        "observed_AP_minus_AMP": obs_delta_mean,
        "null_AP_minus_AMP_mean": float(pop_null_delta.mean()),
        "empirical_p_delta": ep_delta,
        "max_amplitude_error": amp_audit,
    }

    # C2 basis controls.
    _, ppca, _, _ = signflip_one_sided(delta_pca, Bsf, seed0 + 30)
    _, psensor, _, _ = signflip_one_sided(delta_sensor, Bsf, seed0 + 31)
    rand_pop = rand_s.mean(axis=0)  # B x 3
    rand_delta = rand_pop[:, 1] - rand_pop[:, 2]
    summary["C2_basis"] = {
        "GFT_identity_AP_minus_AMP": obs_delta_mean,
        "OOB_PCA": {
            "FULL_mean": float(pca_s[:,0].mean()), "AP_mean": float(pca_s[:,1].mean()), "AMP_mean": float(pca_s[:,2].mean()),
            "AP_minus_AMP_mean": float(delta_pca.mean()), "signflip_p_one_sided": ppca,
        },
        "rankmatched_sensor_coordinates": {
            "FULL_mean": float(sensor_s[:,0].mean()), "AP_mean": float(sensor_s[:,1].mean()), "AMP_mean": float(sensor_s[:,2].mean()),
            "AP_minus_AMP_mean": float(delta_sensor.mean()), "signflip_p_one_sided": psensor,
            "scope": "102 sensor coordinates reconstructed only from the frozen first 16 graph modes",
        },
        "random_orthogonal": {
            "B": int(rand_pop.shape[0]),
            "FULL_mean_median": float(np.median(rand_pop[:,0])),
            "FULL_mean_q05_q95": [float(np.quantile(rand_pop[:,0],0.05)), float(np.quantile(rand_pop[:,0],0.95))],
            "AP_minus_AMP_median": float(np.median(rand_delta)),
            "AP_minus_AMP_q05_q95": [float(np.quantile(rand_delta,0.05)), float(np.quantile(rand_delta,0.95))],
            "fraction_delta_positive": float(np.mean(rand_delta > 0)),
            "fraction_delta_ge_identity": float(np.mean(rand_delta >= obs_delta_mean - 1e-15)),
        }
    }

    # C3 post vs far prestim.
    _, ppre_full, _, _ = signflip_one_sided(post_pre_full, Bsf, seed0 + 40)
    _, ppre_ap, _, _ = signflip_one_sided(post_pre_ap, Bsf, seed0 + 41)
    summary["C3_prestim"] = {
        "pre_FULL_mean": float(pre_s[:,0].mean()), "post_FULL_mean": obs_full_mean,
        "post_minus_pre_FULL_mean": float(post_pre_full.mean()), "paired_signflip_p_one_sided": ppre_full,
        "pre_AP_mean": float(pre_s[:,1].mean()), "post_AP_mean": obs_ap_mean,
        "post_minus_pre_AP_mean": float(post_pre_ap.mean()), "paired_signflip_p_one_sided_AP": ppre_ap,
    }

    # C4 metrics.
    for name, arr, off in [("spearman",obs_s,50),("pearson",obs_p,60),("kendall",obs_k,70)]:
        d = arr[:,1] - arr[:,2]
        _, p, _, _ = signflip_one_sided(d, Bsf, seed0 + off)
        summary["C4_metric"][name] = {
            "FULL_mean": float(arr[:,0].mean()), "AP_mean": float(arr[:,1].mean()), "AMP_mean": float(arr[:,2].mean()),
            "AP_minus_AMP_mean": float(d.mean()), "signflip_p_one_sided": p,
        }

    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "SUMMARY.json", summary)
    pd.DataFrame({
        "subject": subjects,
        "FULL": obs_s[:,0], "AP": obs_s[:,1], "AMP": obs_s[:,2], "AP_minus_AMP": delta_obs,
        "PCA_FULL": pca_s[:,0], "PCA_AP": pca_s[:,1], "PCA_AMP": pca_s[:,2], "PCA_AP_minus_AMP": delta_pca,
        "SENSOR_FULL": sensor_s[:,0], "SENSOR_AP": sensor_s[:,1], "SENSOR_AMP": sensor_s[:,2], "SENSOR_AP_minus_AMP": delta_sensor,
        "PRE_FULL": pre_s[:,0], "PRE_AP": pre_s[:,1], "PRE_AMP": pre_s[:,2],
        "phase_null_FULL_mean": subj_null_full, "phase_null_AP_mean": subj_null_ap,
        "pearson_AP_minus_AMP": obs_p[:,1]-obs_p[:,2], "kendall_AP_minus_AMP": obs_k[:,1]-obs_k[:,2],
    }).to_csv(outdir / "SUBJECT_SUMMARY.csv", index=False)
    pd.DataFrame({
        "replicate": np.arange(rand_pop.shape[0]),
        "FULL": rand_pop[:,0], "AP": rand_pop[:,1], "AMP": rand_pop[:,2], "AP_minus_AMP": rand_delta,
    }).to_csv(outdir / "RANDOM_BASIS_POPULATION.csv", index=False)
    pd.DataFrame({
        "replicate": np.arange(pop_null_full.shape[0]),
        "FULL": pop_null_full, "AP": pop_null_ap, "AP_minus_AMP": pop_null_delta,
    }).to_csv(outdir / "PHASE_NULL_POPULATION.csv", index=False)
    return summary


def make_figures(results_root: Path, summaries: dict[str, dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir = results_root / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    names = list(summaries.keys())

    # Phase-null figure
    x = np.arange(len(names))
    obs = [summaries[n]["C1_phase_null"]["observed_AP"] for n in names]
    nul = [summaries[n]["C1_phase_null"]["null_AP_mean"] for n in names]
    w = 0.35
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.bar(x-w/2, obs, w, label="Observed AP")
    ax.bar(x+w/2, nul, w, label="Amplitude-preserving phase null")
    ax.set_xticks(x, names); ax.set_ylabel("Mean cross-run RDM reliability"); ax.set_title("C1: phase organization control")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir/"FIG_C1_PHASE_NULL.png", dpi=180); plt.close(fig)

    # Basis delta figure
    labels=[]; vals=[]
    for n in names:
        s=summaries[n]
        labels += [f"{n}\nGFT", f"{n}\nOOB-PCA", f"{n}\nSensor"]
        vals += [s["C2_basis"]["GFT_identity_AP_minus_AMP"], s["C2_basis"]["OOB_PCA"]["AP_minus_AMP_mean"], s["C2_basis"]["rankmatched_sensor_coordinates"]["AP_minus_AMP_mean"]]
    fig, ax=plt.subplots(figsize=(9,4.8)); ax.bar(np.arange(len(vals)), vals); ax.axhline(0,linewidth=1); ax.set_xticks(np.arange(len(vals)),labels,rotation=20,ha="right"); ax.set_ylabel("Mean AP - AMP reliability"); ax.set_title("C2: coordinate/basis robustness"); fig.tight_layout(); fig.savefig(figdir/"FIG_C2_BASIS_ROBUSTNESS.png",dpi=180); plt.close(fig)

    # Post vs pre
    obs=[summaries[n]["C3_prestim"]["post_FULL_mean"] for n in names]; pre=[summaries[n]["C3_prestim"]["pre_FULL_mean"] for n in names]
    fig, ax=plt.subplots(figsize=(7,4.5)); ax.bar(x-w/2,obs,w,label="50-250 ms"); ax.bar(x+w/2,pre,w,label="-450 to -250 ms"); ax.set_xticks(x,names); ax.set_ylabel("Mean FULL reliability"); ax.set_title("C3: poststimulus vs far-prestimulus"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"FIG_C3_POST_VS_PRE.png",dpi=180); plt.close(fig)


def article_draft(results_root: Path, summaries: dict[str, dict]):
    lines = ["# Manuscript integration draft — generated from frozen controls", "",
             "These analyses are post-confirmatory robustness/falsification controls on cohorts whose primary outcomes were already known; they are not a new prospective confirmation.", ""]
    for name, s in summaries.items():
        c1=s["C1_phase_null"]; c2=s["C2_basis"]; c3=s["C3_prestim"]; c4=s["C4_metric"]
        lines += [f"## {name}",
                  f"Observed FULL={s['observed_spearman']['FULL']['mean']:.4f}, AP={s['observed_spearman']['AP']['mean']:.4f}, AMP={s['observed_spearman']['AMP']['mean']:.4f}, AP-AMP={s['observed_spearman']['AP_minus_AMP']['mean']:.4f}.",
                  f"Amplitude-preserving phase null: AP null mean={c1['null_AP_mean']:.4f}, empirical p={c1['empirical_p_AP']:.4g}; FULL null mean={c1['null_FULL_mean']:.4f}, empirical p={c1['empirical_p_FULL']:.4g}.",
                  f"OOB-PCA AP-AMP={c2['OOB_PCA']['AP_minus_AMP_mean']:.4f} (sign-flip p={c2['OOB_PCA']['signflip_p_one_sided']:.4g}); rank-matched sensor-coordinate AP-AMP={c2['rankmatched_sensor_coordinates']['AP_minus_AMP_mean']:.4f} (p={c2['rankmatched_sensor_coordinates']['signflip_p_one_sided']:.4g}).",
                  f"Across {c2['random_orthogonal']['B']} random orthogonal rotations, median AP-AMP={c2['random_orthogonal']['AP_minus_AMP_median']:.4f}, fraction positive={c2['random_orthogonal']['fraction_delta_positive']:.3f}.",
                  f"Far-prestimulus FULL={c3['pre_FULL_mean']:.4f} versus poststimulus FULL={c3['post_FULL_mean']:.4f}; paired post-pre p={c3['paired_signflip_p_one_sided']:.4g}.",
                  f"Metric robustness: Pearson AP-AMP={c4['pearson']['AP_minus_AMP_mean']:.4f} (p={c4['pearson']['signflip_p_one_sided']:.4g}); Kendall AP-AMP={c4['kendall']['AP_minus_AMP_mean']:.4f} (p={c4['kendall']['signflip_p_one_sided']:.4g}).", ""]
    (results_root / "ARTICLE_RESULTS_DRAFT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--drives", nargs="*", default=["D:", "E:"])
    args = ap.parse_args()
    root = Path(args.package_root).resolve()
    cfg_path = root / "protocol" / "CONTROL_CONFIG.json"
    cfg = load_json(cfg_path)
    log = log_factory(root / "logs" / "RUN.log")
    protocol_hash = sha256_file(cfg_path)
    template = root / "protocol" / "SGIT_Level5A_template_graph.npz"
    template_hash = sha256_file(template)
    save_json(root / "state" / "FROZEN_PLAN_HASH.json", {
        "control_config_sha256": protocol_hash,
        "template_graph_sha256": template_hash,
        "status": "HASHED_BEFORE_CONTROL_OUTCOMES",
    })
    log(f"Protocol hash: {protocol_hash}")
    log(f"Template graph hash: {template_hash}")

    ca_root = discover_finetf_root(args.drives, "CA", "SGIT_CA_FINETF_ROOT", log)
    cb_root = discover_finetf_root(args.drives, "CB", "SGIT_CB_FINETF_ROOT", log)
    if ca_root is None or cb_root is None:
        missing=[]
        if ca_root is None: missing.append("CA cohort_finetf")
        if cb_root is None: missing.append("CB cohort_finetf_cb")
        msg = "Could not auto-detect: " + ", ".join(missing) + ". Set SGIT_CA_FINETF_ROOT / SGIT_CB_FINETF_ROOT and rerun."
        log(msg); raise SystemExit(3)
    save_json(root / "state" / "DATA_PATHS.json", {"CA": str(ca_root), "CB": str(cb_root)})
    log(f"CA root: {ca_root}")
    log(f"CB root: {cb_root}")

    # Full frozen preflight before any new control outcome is calculated.
    pre_rows=[]; bad=[]
    for cohort_name, croot in [("CA_Level5A",ca_root),("CB_Level6A_strict",cb_root)]:
        for s in cfg["cohorts"][cohort_name]:
            reasons=preflight_subject(croot,s,template,k_modes=int(cfg["frozen_signal_pipeline"]["K"]))
            pre_rows.append({"cohort":cohort_name,"subject":s,"eligible":not reasons,"reasons":" ; ".join(reasons)})
            if reasons: bad.append((cohort_name,s,reasons))
    pd.DataFrame(pre_rows).to_csv(root/"state"/"PREFLIGHT.csv",index=False)
    save_json(root/"state"/"PREFLIGHT.json", pre_rows)
    if bad:
        log(f"Preflight FAILED for {len(bad)} frozen subjects; no control outcome computed.")
        for c,s,r in bad[:20]: log(f"  {c} {s}: {'; '.join(r)}")
        raise SystemExit(4)
    log(f"Preflight PASS: {len(cfg['cohorts']['CA_Level5A'])}+{len(cfg['cohorts']['CB_Level6A_strict'])} frozen subjects available and graph-compatible.")

    tz=np.load(template,allow_pickle=True)
    U=np.asarray(tz["eigenvectors"],float)[:,:int(cfg["frozen_signal_pipeline"]["K"])]
    if not np.allclose(U.T@U,np.eye(U.shape[1]),atol=1e-6):
        raise RuntimeError("Frozen graph eigenvectors are not orthonormal within tolerance.")

    Brot=int(cfg["controls"]["C2_random_basis"]["B"]); seedrot=int(cfg["controls"]["C2_random_basis"]["seed"])
    rrng=np.random.default_rng(seedrot)
    random_Qs=[random_orthogonal(U.shape[1],rrng) for _ in range(Brot)]
    np.savez_compressed(root/"state"/"RANDOM_BASES.npz", **{f"Q{i:02d}":q for i,q in enumerate(random_Qs)})

    cache_root=root/"state"/"subject_cache"
    summaries={}
    for cohort_name,croot in [("CA_Level5A",ca_root),("CB_Level6A_strict",cb_root)]:
        subjects=cfg["cohorts"][cohort_name]
        caches=[]
        log(f"=== {cohort_name}: {len(subjects)} subjects ===")
        for i,s in enumerate(subjects,1):
            log(f"{cohort_name} {i}/{len(subjects)}: {s}")
            cp=subject_cache_path(cache_root,cohort_name,s)
            caches.append(run_subject(croot,s,cfg,U,random_Qs,protocol_hash,cp,log))
        summaries[cohort_name]=aggregate_cohort(cohort_name,subjects,caches,cfg,root/"results"/cohort_name)
        log(f"{cohort_name}: aggregation complete")

    save_json(root/"results"/"ARTICLE_CONTROL_SUMMARY.json", summaries)
    rows=[]
    for name,s in summaries.items():
        rows += [
            {"cohort":name,"analysis":"identity","FULL":s["observed_spearman"]["FULL"]["mean"],"AP":s["observed_spearman"]["AP"]["mean"],"AMP":s["observed_spearman"]["AMP"]["mean"],"AP_minus_AMP":s["observed_spearman"]["AP_minus_AMP"]["mean"]},
            {"cohort":name,"analysis":"OOB_PCA","FULL":s["C2_basis"]["OOB_PCA"]["FULL_mean"],"AP":s["C2_basis"]["OOB_PCA"]["AP_mean"],"AMP":s["C2_basis"]["OOB_PCA"]["AMP_mean"],"AP_minus_AMP":s["C2_basis"]["OOB_PCA"]["AP_minus_AMP_mean"]},
            {"cohort":name,"analysis":"rankmatched_sensor","FULL":s["C2_basis"]["rankmatched_sensor_coordinates"]["FULL_mean"],"AP":s["C2_basis"]["rankmatched_sensor_coordinates"]["AP_mean"],"AMP":s["C2_basis"]["rankmatched_sensor_coordinates"]["AMP_mean"],"AP_minus_AMP":s["C2_basis"]["rankmatched_sensor_coordinates"]["AP_minus_AMP_mean"]},
            {"cohort":name,"analysis":"far_prestim","FULL":s["C3_prestim"]["pre_FULL_mean"],"AP":s["C3_prestim"]["pre_AP_mean"],"AMP":np.nan,"AP_minus_AMP":np.nan},
        ]
    pd.DataFrame(rows).to_csv(root/"results"/"ARTICLE_CONTROL_TABLE.csv",index=False)
    make_figures(root/"results",summaries)
    article_draft(root/"results",summaries)
    (root/"state"/"RUN_COMPLETE.flag").write_text(time.strftime("%Y-%m-%d %H:%M:%S")+"\n",encoding="utf-8")
    log("ALL CONTROLS COMPLETE")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(10)
