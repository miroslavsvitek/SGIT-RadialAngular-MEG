from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from common import (
    CATS, RUNS, assoc, bootstrap_mean_ci, discover_finetf_root, load_json,
    partitions_2v2, paths_for, preflight_subject, save_json, sha256_file,
    signflip_one_sided,
)

COMPONENTS = ["FULL", "AP", "AMP", "ENERGY", "PRODUCT", "INTERFERENCE", "ANGLE"]
XPRED = ["INTERFERENCE_TO_FULL", "INTERFERENCE_TO_AP", "PRODUCT_TO_AP", "ENERGY_TO_FULL"]


@dataclass
class RunData:
    categories: np.ndarray
    zt_post: np.ndarray
    za_post: np.ndarray


def log_factory(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def log(msg: str):
        s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(s, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(s + "\n")
    return log


def valid_root(path: Path | None, prefix: str) -> bool:
    if path is None:
        return False
    try:
        fine = path / "meeg_fine"
        meta = path / "metadata"
        graphs = path / "graphs"
        if not (fine.is_dir() and meta.is_dir() and graphs.is_dir()):
            return False
        return any(fine.glob(f"sub-{prefix}*_run-01_sgit_level4e_finetf.npz"))
    except OSError:
        return False


def resolve_data_roots(root: Path, drives: list[str], log):
    # 1) explicit environment overrides
    env_ca = os.environ.get("SGIT_CA_FINETF_ROOT", "").strip().strip('"')
    env_cb = os.environ.get("SGIT_CB_FINETF_ROOT", "").strip().strip('"')
    ca = Path(env_ca) if env_ca else None
    cb = Path(env_cb) if env_cb else None
    if valid_root(ca, "CA"):
        log(f"CA: explicit environment path {ca}")
    else:
        ca = None
    if valid_root(cb, "CB"):
        log(f"CB: explicit environment path {cb}")
    else:
        cb = None

    # 2) reuse the exact validated paths from PAPER01 if its state is still present
    prior = root.parent / "SGIT_PAPER_CONTROLS_v1_1" / "state" / "DATA_PATHS.json"
    if prior.exists() and (ca is None or cb is None):
        try:
            p = json.loads(prior.read_text(encoding="utf-8"))
            pca = Path(p.get("CA", "")); pcb = Path(p.get("CB", ""))
            if ca is None and valid_root(pca, "CA"):
                ca = pca; log(f"CA: reusing validated PAPER01 path {ca}")
            if cb is None and valid_root(pcb, "CB"):
                cb = pcb; log(f"CB: reusing validated PAPER01 path {cb}")
        except Exception as e:
            log(f"PAPER01 path reuse skipped: {type(e).__name__}")

    # 3) exact paths observed in the successful PAPER01 run; try D and E only, no broad scan yet
    preferred_rel = {
        "CA": [
            r"SGIT\SGIT_COGITATE_Population_Runner_v2_1\cohort_finetf",
            r"SGIT\SGIT_COGITATE_Population_Runner_v2_2\cohort_finetf",
        ],
        "CB": [
            r"SGIT\SGIT_LEVEL6_AllInOne_v3_0\SGIT_LEVEL6_AllInOne_v3_1\level6_data\cohort_finetf_cb",
            r"SGIT\SGIT_LEVEL6_AllInOne_v3_1\level6_data\cohort_finetf_cb",
        ],
    }
    for d in drives:
        base = Path(d + "\\") if os.name == "nt" else Path(d)
        if ca is None:
            for rel in preferred_rel["CA"]:
                p = base / Path(rel)
                if valid_root(p, "CA"):
                    ca = p; log(f"CA: preferred path {ca}"); break
        if cb is None:
            for rel in preferred_rel["CB"]:
                p = base / Path(rel)
                if valid_root(p, "CB"):
                    cb = p; log(f"CB: preferred path {cb}"); break

    # 4) bounded fallback discovery identical to PAPER01
    if ca is None:
        ca = discover_finetf_root(drives, "CA", "SGIT_CA_FINETF_ROOT", log)
    if cb is None:
        cb = discover_finetf_root(drives, "CB", "SGIT_CB_FINETF_ROOT", log)
    return ca, cb


def load_subject(root: Path, subject: str, cfg: dict) -> dict[int, RunData]:
    fs = float(cfg["frozen_signal_pipeline"]["fs_hz"])
    theta = cfg["frozen_signal_pipeline"]["theta_hz"]
    alpha = cfg["frozen_signal_pipeline"]["alpha_hz"]
    post = cfg["frozen_signal_pipeline"]["post_window_s"]
    K = int(cfg["frozen_signal_pipeline"]["K"])
    sos_t = butter(4, theta, btype="bandpass", fs=fs, output="sos")
    sos_a = butter(4, alpha, btype="bandpass", fs=fs, output="sos")
    out = {}
    for r in RUNS:
        ff, mf, _ = paths_for(root, subject, r)
        if ff is None or mf is None:
            raise RuntimeError(f"{subject} run {r}: required files disappeared after preflight")
        meta = pd.read_csv(mf)
        with np.load(ff, allow_pickle=False) as z:
            raw = z["gft_trajectories"].astype(np.float64)
            times = z["times_s"].astype(float)
        if raw.ndim != 3 or raw.shape[1] != K or raw.shape[0] != len(meta):
            raise RuntimeError(f"{subject} run {r}: invalid FineTF shape {raw.shape} / metadata {len(meta)}")
        m = (times >= float(post[0])) & (times < float(post[1]))
        if int(m.sum()) < 10:
            raise RuntimeError(f"{subject} run {r}: post window absent")
        zt = hilbert(sosfiltfilt(sos_t, raw, axis=-1), axis=-1)[:, :, m].copy()
        za = hilbert(sosfiltfilt(sos_a, raw, axis=-1), axis=-1)[:, :, m].copy()
        cats = meta["sgit_category"].astype(str).to_numpy()
        out[r] = RunData(cats, zt, za)
    return out


def build_field(rd: RunData, rank: int) -> np.ndarray:
    Et, Ea, Pa, Xcats = [], [], [], []
    for c in CATS:
        ix = np.where(rd.categories == c)[0]
        et = rd.zt_post[ix].mean(axis=0)
        ea = rd.za_post[ix].mean(axis=0)
        pa = np.mean(np.abs(rd.za_post[ix]) ** 2, axis=(0, 2))
        Et.append(et); Ea.append(ea); Pa.append(pa)
        Xcats.append(np.concatenate([et.T, ea.T], axis=1))
    G = np.mean(np.asarray(Xcats), axis=0)
    _, _, Vh = np.linalg.svd(G, full_matrices=False)
    V = Vh.conj().T[:, :rank]
    Ya = []
    K = rd.za_post.shape[1]
    for ci in range(4):
        X = np.concatenate([Et[ci].T, Ea[ci].T], axis=1)
        Xr = X - (X @ V) @ V.conj().T
        ra = Xr[:, K:].T
        Ya.append(ra / np.sqrt(np.maximum(Pa[ci], 1e-30))[:, None])
    return np.stack(Ya)


def component_rdm(y: np.ndarray):
    A = np.abs(y)
    phi = np.angle(y)
    vals = {k: [] for k in COMPONENTS}
    max_abs = 0.0
    max_rel = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            ai, aj = A[i], A[j]
            dphi = phi[i] - phi[j]
            E = float(np.sum(ai * ai + aj * aj))
            P = float(np.sum(2.0 * ai * aj))
            C = float(np.sum(2.0 * ai * aj * np.cos(dphi)))
            I = -C
            ANG = float(np.sum(2.0 * (1.0 - np.cos(dphi))))
            AMP = float(np.sum((ai - aj) ** 2))
            AP = float(np.sum(2.0 * ai * aj * (1.0 - np.cos(dphi))))
            FULL = float(np.sum(np.abs(y[i] - y[j]) ** 2))
            identities = [AMP - (E - P), AP - (P + I), FULL - (E + I)]
            sc = max(1.0, abs(E), abs(P), abs(C), abs(FULL), abs(AP), abs(AMP))
            max_abs = max(max_abs, max(abs(x) for x in identities))
            max_rel = max(max_rel, max(abs(x) for x in identities) / sc)
            row = {
                "FULL": FULL, "AP": AP, "AMP": AMP, "ENERGY": E,
                "PRODUCT": P, "INTERFERENCE": I, "ANGLE": ANG,
            }
            for k in COMPONENTS:
                vals[k].append(row[k])
    return {k: np.asarray(vals[k], float) for k in COMPONENTS}, max_abs, max_rel


def sym_cross(a1, b1, a2, b2, metric: str) -> float:
    x = assoc(a1, b2, metric)
    y = assoc(a2, b1, metric)
    return float(np.nanmean([x, y]))


def subject_scores(run_fields: dict[int, np.ndarray], metric: str = "spearman"):
    by_key = {k: [] for k in COMPONENTS}
    xp = {k: [] for k in XPRED}
    max_abs = 0.0; max_rel = 0.0
    for a, b, unused in partitions_2v2():
        ya = np.mean([run_fields[r] for r in a], axis=0)
        yb = np.mean([run_fields[r] for r in b], axis=0)
        da, aa, ra = component_rdm(ya)
        db, ab, rb = component_rdm(yb)
        max_abs = max(max_abs, aa, ab); max_rel = max(max_rel, ra, rb)
        for k in COMPONENTS:
            by_key[k].append(assoc(da[k], db[k], metric))
        xp["INTERFERENCE_TO_FULL"].append(sym_cross(da["INTERFERENCE"], da["FULL"], db["INTERFERENCE"], db["FULL"], metric))
        xp["INTERFERENCE_TO_AP"].append(sym_cross(da["INTERFERENCE"], da["AP"], db["INTERFERENCE"], db["AP"], metric))
        xp["PRODUCT_TO_AP"].append(sym_cross(da["PRODUCT"], da["AP"], db["PRODUCT"], db["AP"], metric))
        xp["ENERGY_TO_FULL"].append(sym_cross(da["ENERGY"], da["FULL"], db["ENERGY"], db["FULL"], metric))
    scores = np.asarray([np.nanmean(by_key[k]) for k in COMPONENTS], float)
    xps = np.asarray([np.nanmean(xp[k]) for k in XPRED], float)
    return scores, xps, max_abs, max_rel


def cache_path(cache_root: Path, cohort: str, subject: str) -> Path:
    return cache_root / f"{cohort}__{subject}.npz"


def load_cache(path: Path, protocol_hash: str):
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["protocol_hash"].item()) != protocol_hash:
            return None
        return {k: z[k] for k in z.files}
    except Exception:
        return None


def run_subject(data_root: Path, subject: str, cohort: str, cfg: dict, protocol_hash: str, cp: Path, log):
    c = load_cache(cp, protocol_hash)
    if c is not None:
        log(f"{subject}: cache OK")
        return c
    rank = int(cfg["frozen_signal_pipeline"]["common_subspace_rank"])
    data = load_subject(data_root, subject, cfg)
    fields = {r: build_field(data[r], rank) for r in RUNS}
    sp, xp_sp, a1, r1 = subject_scores(fields, "spearman")
    pe, xp_pe, a2, r2 = subject_scores(fields, "pearson")
    ke, xp_ke, a3, r3 = subject_scores(fields, "kendall")
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cp, protocol_hash=np.asarray(protocol_hash), subject=np.asarray(subject), cohort=np.asarray(cohort),
        spearman=sp, pearson=pe, kendall=ke,
        xpred_spearman=xp_sp, xpred_pearson=xp_pe, xpred_kendall=xp_ke,
        audit_abs=np.asarray(max(a1,a2,a3)), audit_rel=np.asarray(max(r1,r2,r3)),
    )
    return load_cache(cp, protocol_hash)


def ci_record(x, cfg, seed):
    m, lo, hi = bootstrap_mean_ci(x, int(cfg["bootstrap_B"]), seed)
    return {"mean": m, "ci95": [lo, hi]}


def stat_record(x, cfg, seed):
    rec = ci_record(x, cfg, seed)
    _, p, mode, B = signflip_one_sided(x, int(cfg["signflip_B"]), seed + 1)
    rec.update({"signflip_p_one_sided": p, "signflip_mode": mode, "signflip_B": B})
    return rec


def aggregate(cohort: str, subjects: list[str], caches: list[dict], cfg: dict, outdir: Path):
    S = np.vstack([c["spearman"] for c in caches])
    P = np.vstack([c["pearson"] for c in caches])
    K = np.vstack([c["kendall"] for c in caches])
    XS = np.vstack([c["xpred_spearman"] for c in caches])
    XP = np.vstack([c["xpred_pearson"] for c in caches])
    XK = np.vstack([c["xpred_kendall"] for c in caches])
    seed = int(cfg["signflip_seed"]) + (10000 if cohort.startswith("CB") else 0)
    summary = {
        "cohort": cohort, "N": len(subjects), "subjects": subjects,
        "component_order": COMPONENTS, "cross_predictive_order": XPRED,
        "spearman": {}, "cross_predictive_spearman": {}, "secondary_metrics": {},
        "contrasts": {},
        "audits": {
            "max_exact_identity_abs_error": max(float(c["audit_abs"].item()) for c in caches),
            "max_exact_identity_relative_error": max(float(c["audit_rel"].item()) for c in caches),
        },
    }
    for j, name in enumerate(COMPONENTS):
        summary["spearman"][name] = stat_record(S[:,j], cfg, seed + 10*j)
    for j, name in enumerate(XPRED):
        summary["cross_predictive_spearman"][name] = stat_record(XS[:,j], cfg, seed + 200 + 10*j)

    idx = {k:i for i,k in enumerate(COMPONENTS)}
    d_ap_prod = S[:,idx["AP"]] - S[:,idx["PRODUCT"]]
    d_prod_ap = -d_ap_prod
    d_int_prod = S[:,idx["INTERFERENCE"]] - S[:,idx["PRODUCT"]]
    summary["contrasts"]["AP_minus_PRODUCT"] = stat_record(d_ap_prod, cfg, seed + 400)
    summary["contrasts"]["PRODUCT_minus_AP"] = stat_record(d_prod_ap, cfg, seed + 410)
    summary["contrasts"]["INTERFERENCE_minus_PRODUCT"] = stat_record(d_int_prod, cfg, seed + 420)

    for metric_name, arr, xarr, off in [("pearson",P,XP,600),("kendall",K,XK,800)]:
        summary["secondary_metrics"][metric_name] = {
            "INTERFERENCE": stat_record(arr[:,idx["INTERFERENCE"]], cfg, seed + off),
            "PRODUCT": stat_record(arr[:,idx["PRODUCT"]], cfg, seed + off + 10),
            "ANGLE": stat_record(arr[:,idx["ANGLE"]], cfg, seed + off + 20),
            "INTERFERENCE_TO_FULL": stat_record(xarr[:,XPRED.index("INTERFERENCE_TO_FULL")], cfg, seed + off + 30),
        }

    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "SUMMARY.json", summary)
    df = pd.DataFrame({"subject": subjects})
    for j,k in enumerate(COMPONENTS): df[k] = S[:,j]
    for j,k in enumerate(XPRED): df[k] = XS[:,j]
    df["AP_minus_PRODUCT"] = d_ap_prod
    df["PRODUCT_minus_AP"] = d_prod_ap
    df.to_csv(outdir / "SUBJECT_COMPONENT_RELIABILITY.csv", index=False)
    return summary


def classify(summaries: dict[str, dict], alpha=0.025):
    def pass_stat(s, section, key):
        r = s[section][key]
        return r["mean"] > 0 and r["signflip_p_one_sided"] < alpha
    phase_by = {}
    prod_by = {}
    angle_by = {}
    for name,s in summaries.items():
        phase_by[name] = pass_stat(s,"spearman","INTERFERENCE") and pass_stat(s,"cross_predictive_spearman","INTERFERENCE_TO_FULL")
        prod_by[name] = pass_stat(s,"spearman","PRODUCT")
        angle_by[name] = pass_stat(s,"spearman","ANGLE")
    phase_both = all(phase_by.values())
    prod_both = all(prod_by.values())
    if phase_both and prod_both:
        decision = "MIXED_PHASE_AND_PRODUCT"
    elif phase_both and not prod_both:
        decision = "PHASE_SPECIFIC_WITHOUT_PRODUCT_DOMINANCE"
    elif prod_both and not phase_both:
        decision = "PRODUCT_SCAFFOLD_DOMINANT"
    else:
        decision = "INDETERMINATE"
    return {
        "decision": decision,
        "alpha_one_sided": alpha,
        "phase_specific_pass_by_cohort": phase_by,
        "product_pass_by_cohort": prod_by,
        "phase_only_angle_pass_by_cohort": angle_by,
        "guardrail": "This decision adjudicates new frozen component endpoints on previously used cohorts; it is not a new independent acquisition."
    }


def make_figures(results_root: Path, summaries: dict[str, dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir = results_root / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    names = list(summaries)
    comps = ["FULL","AP","AMP","ENERGY","PRODUCT","INTERFERENCE","ANGLE"]
    x = np.arange(len(comps)); w = 0.35
    fig, ax = plt.subplots(figsize=(10,5.2))
    for q,name in enumerate(names):
        vals=[summaries[name]["spearman"][c]["mean"] for c in comps]
        ax.bar(x + (q-0.5)*w, vals, w, label=name)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, comps, rotation=25, ha="right")
    ax.set_ylabel("Mean cross-run Spearman reliability")
    ax.set_title("PAPER02: exact component reliabilities")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir/"FIG_PAPER02_COMPONENTS.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5,4.8))
    xp=["INTERFERENCE_TO_FULL","INTERFERENCE_TO_AP","PRODUCT_TO_AP","ENERGY_TO_FULL"]
    x=np.arange(len(xp))
    for q,name in enumerate(names):
        vals=[summaries[name]["cross_predictive_spearman"][k]["mean"] for k in xp]
        ax.bar(x+(q-0.5)*w,vals,w,label=name)
    ax.axhline(0,linewidth=1); ax.set_xticks(x,xp,rotation=25,ha="right")
    ax.set_ylabel("Symmetric cross-half Spearman congruence")
    ax.set_title("PAPER02: cross-component prediction")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir/"FIG_PAPER02_CROSS_COMPONENT.png",dpi=180); plt.close(fig)


def write_article_draft(results_root: Path, summaries: dict[str, dict], decision: dict):
    lines=[
        "# PAPER02 article-ready result draft",
        "",
        "PAPER02 was frozen after the PAPER01 amplitude-preserving phase-shuffle result and before the component outcomes below were inspected. It reuses the same two participant cohorts and therefore adjudicates mechanism rather than providing a new independent acquisition.",
        "",
        f"Frozen decision: **{decision['decision']}**.",
        "",
    ]
    for name,s in summaries.items():
        R=s["spearman"]; X=s["cross_predictive_spearman"]; C=s["contrasts"]
        lines += [
            f"## {name}",
            f"ENERGY R={R['ENERGY']['mean']:.4f}, PRODUCT R={R['PRODUCT']['mean']:.4f}, INTERFERENCE R={R['INTERFERENCE']['mean']:.4f} (p={R['INTERFERENCE']['signflip_p_one_sided']:.4g}), ANGLE R={R['ANGLE']['mean']:.4f}.",
            f"INTERFERENCE-to-FULL cross-half congruence={X['INTERFERENCE_TO_FULL']['mean']:.4f} (p={X['INTERFERENCE_TO_FULL']['signflip_p_one_sided']:.4g}).",
            f"AP-product reliability contrast={C['AP_minus_PRODUCT']['mean']:.4f}; product-AP={C['PRODUCT_minus_AP']['mean']:.4f} (one-sided p for product>AP={C['PRODUCT_minus_AP']['signflip_p_one_sided']:.4g}).",
            f"Exact identity audit max relative error={s['audits']['max_exact_identity_relative_error']:.3e}.",
            "",
        ]
    (results_root/"ARTICLE_RESULTS_DRAFT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--drives", nargs="*", default=["D:","E:"])
    args=ap.parse_args()
    root=Path(args.package_root).resolve()
    cfg_path=root/"protocol"/"PAPER02_CONFIG.json"
    cfg=load_json(cfg_path)
    log=log_factory(root/"logs"/"RUN.log")
    protocol_hash=sha256_file(cfg_path)
    template=root/"protocol"/"SGIT_Level5A_template_graph.npz"
    template_hash=sha256_file(template)
    save_json(root/"state"/"FROZEN_PLAN_HASH.json",{
        "paper02_config_sha256":protocol_hash,
        "template_graph_sha256":template_hash,
        "status":"HASHED_BEFORE_PAPER02_COMPONENT_OUTCOMES"
    })
    log(f"PAPER02 protocol hash: {protocol_hash}")
    log(f"Template hash: {template_hash}")

    ca_root,cb_root=resolve_data_roots(root,args.drives,log)
    if ca_root is None or cb_root is None:
        log("Could not resolve both frozen FineTF roots. Set SGIT_CA_FINETF_ROOT and SGIT_CB_FINETF_ROOT and rerun.")
        raise SystemExit(3)
    save_json(root/"state"/"DATA_PATHS.json",{"CA":str(ca_root),"CB":str(cb_root)})

    pre=[]; bad=[]
    for cohort,croot in [("CA_Level5A",ca_root),("CB_Level6A_strict",cb_root)]:
        for s in cfg["cohorts"][cohort]:
            reasons=preflight_subject(croot,s,template,k_modes=int(cfg["frozen_signal_pipeline"]["K"]))
            pre.append({"cohort":cohort,"subject":s,"eligible":not reasons,"reasons":" ; ".join(reasons)})
            if reasons: bad.append((cohort,s,reasons))
    pd.DataFrame(pre).to_csv(root/"state"/"PREFLIGHT.csv",index=False)
    save_json(root/"state"/"PREFLIGHT.json",pre)
    if bad:
        log(f"Preflight FAILED for {len(bad)} frozen subjects; no PAPER02 outcomes computed.")
        for c,s,r in bad[:20]: log(f"  {c} {s}: {'; '.join(r)}")
        raise SystemExit(4)
    log(f"Preflight PASS: {len(pre)} frozen subjects available and graph-compatible.")

    summaries={}; cache_root=root/"state"/"subject_cache"
    for cohort,croot in [("CA_Level5A",ca_root),("CB_Level6A_strict",cb_root)]:
        subjects=cfg["cohorts"][cohort]; caches=[]
        log(f"=== {cohort}: {len(subjects)} subjects ===")
        for i,s in enumerate(subjects,1):
            log(f"{cohort} {i}/{len(subjects)}: {s}")
            caches.append(run_subject(croot,s,cohort,cfg,protocol_hash,cache_path(cache_root,cohort,s),log))
        summaries[cohort]=aggregate(cohort,subjects,caches,cfg,root/"results"/cohort)
        log(f"{cohort}: aggregation complete")

    decision=classify(summaries,alpha=0.025)
    save_json(root/"results"/"PAPER02_SUMMARY.json",summaries)
    save_json(root/"results"/"PAPER02_DECISION.json",decision)
    rows=[]
    for cohort,s in summaries.items():
        for comp in COMPONENTS:
            r=s["spearman"][comp]
            rows.append({"cohort":cohort,"endpoint":comp,"mean":r["mean"],"ci_low":r["ci95"][0],"ci_high":r["ci95"][1],"p_one_sided":r["signflip_p_one_sided"]})
        for k in XPRED:
            r=s["cross_predictive_spearman"][k]
            rows.append({"cohort":cohort,"endpoint":k,"mean":r["mean"],"ci_low":r["ci95"][0],"ci_high":r["ci95"][1],"p_one_sided":r["signflip_p_one_sided"]})
    pd.DataFrame(rows).to_csv(root/"results"/"PAPER02_ARTICLE_TABLE.csv",index=False)
    make_figures(root/"results",summaries)
    write_article_draft(root/"results",summaries,decision)
    (root/"state"/"RUN_COMPLETE.flag").write_text(time.strftime("%Y-%m-%d %H:%M:%S")+"\n",encoding="utf-8")
    log(f"PAPER02 COMPLETE. Frozen decision: {decision['decision']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(10)
