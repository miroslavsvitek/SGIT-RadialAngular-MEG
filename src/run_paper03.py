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
from scipy.stats import rankdata

from common import (
    CATS, RUNS, bootstrap_mean_ci, discover_finetf_root, load_json,
    partitions_2v2, paths_for, preflight_subject, save_json, sha256_file,
    signflip_one_sided, stable_subject_seed,
)

COMPONENTS = ["INTERFERENCE", "ANGLE"]


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
        fine = path / "meeg_fine"; meta = path / "metadata"; graphs = path / "graphs"
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
    if valid_root(ca, "CA"): log(f"CA: explicit environment path {ca}")
    else: ca = None
    if valid_root(cb, "CB"): log(f"CB: explicit environment path {cb}")
    else: cb = None

    # 2) exact validated paths from the successful PAPER02 run if sibling state exists
    prior = root.parent / "SGIT_PAPER02_INTERFERENCE_v1_0" / "state" / "DATA_PATHS.json"
    if prior.exists() and (ca is None or cb is None):
        try:
            p = json.loads(prior.read_text(encoding="utf-8"))
            pca = Path(p.get("CA", "")); pcb = Path(p.get("CB", ""))
            if ca is None and valid_root(pca, "CA"):
                ca = pca; log(f"CA: reusing validated PAPER02 path {ca}")
            if cb is None and valid_root(pcb, "CB"):
                cb = pcb; log(f"CB: reusing validated PAPER02 path {cb}")
        except Exception as e:
            log(f"PAPER02 path reuse skipped: {type(e).__name__}")

    # 3) paths frozen in provenance, tried on both attached external drives
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

    # 4) bounded fallback discovery
    if ca is None: ca = discover_finetf_root(drives, "CA", "SGIT_CA_FINETF_ROOT", log)
    if cb is None: cb = discover_finetf_root(drives, "CB", "SGIT_CB_FINETF_ROOT", log)
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
    K = rd.za_post.shape[1]
    Ya = []
    for ci in range(4):
        X = np.concatenate([Et[ci].T, Ea[ci].T], axis=1)
        Xr = X - (X @ V) @ V.conj().T
        ra = Xr[:, K:].T
        Ya.append(ra / np.sqrt(np.maximum(Pa[ci], 1e-30))[:, None])
    return np.stack(Ya)


def component_rdm(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.abs(y).reshape(4, -1)
    P = np.angle(y).reshape(4, -1)
    inter, angle = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            cd = np.cos(P[i] - P[j])
            inter.append(float(-2.0 * np.sum(A[i] * A[j] * cd)))
            angle.append(float(2.0 * np.sum(1.0 - cd)))
    return np.asarray(inter, float), np.asarray(angle, float)


def row_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx = rankdata(np.asarray(x, float), axis=1, method="average")
    ry = rankdata(np.asarray(y, float), axis=1, method="average")
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    den = np.linalg.norm(rx, axis=1) * np.linalg.norm(ry, axis=1)
    out = np.full(len(rx), np.nan, dtype=float)
    ok = den > 0
    out[ok] = np.sum(rx[ok] * ry[ok], axis=1) / den[ok]
    return out


def scalar_spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(row_spearman(np.asarray(a, float)[None, :], np.asarray(b, float)[None, :])[0])


def shuffled_rdms(y: np.ndarray, rng: np.random.Generator, B: int, batch_size: int):
    """Return B x 6 INTERFERENCE and ANGLE RDMs under category-phase-label permutation.

    At each coordinate, the four phase values are permuted across categories while amplitudes
    stay attached to their original categories. This is the frozen PAPER01/PAPER03 null.
    """
    A = np.abs(y).reshape(4, -1)
    P = np.angle(y).reshape(4, -1)
    M = P.shape[1]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    out_i = np.empty((B, 6), dtype=float)
    out_a = np.empty((B, 6), dtype=float)
    max_amp_error = 0.0
    phase_multiset_error = 0.0
    pos = 0
    first_batch = True
    while pos < B:
        m = min(batch_size, B - pos)
        # axis 1 is category, axis 2 is coordinate: one random category permutation per coordinate/replicate
        perm = np.argsort(rng.random((m, 4, M)), axis=1)
        pbase = np.broadcast_to(P[None, :, :], (m, 4, M))
        ps = np.take_along_axis(pbase, perm, axis=1)
        for q, (i, j) in enumerate(pairs):
            cd = np.cos(ps[:, i, :] - ps[:, j, :])
            out_i[pos:pos+m, q] = -2.0 * np.sum((A[i] * A[j])[None, :] * cd, axis=1)
            out_a[pos:pos+m, q] = 2.0 * np.sum(1.0 - cd, axis=1)
        if first_batch:
            # numerical audits on one batch are sufficient because all later shuffles use the same operation.
            z = A[None, :, :] * np.exp(1j * ps)
            max_amp_error = float(np.max(np.abs(np.abs(z) - A[None, :, :])))
            s0 = np.sort(P, axis=0)[None, :, :]
            s1 = np.sort(ps, axis=1)
            phase_multiset_error = float(np.max(np.abs(s1 - s0)))
            first_batch = False
        pos += m
    return out_i, out_a, max_amp_error, phase_multiset_error


def subject_scores_and_null(run_fields: dict[int, np.ndarray], B: int, seed: int, batch_size: int):
    rng = np.random.default_rng(seed)
    obs_i, obs_a = [], []
    null_i = np.zeros(B, dtype=float)
    null_a = np.zeros(B, dtype=float)
    max_amp_error = 0.0
    phase_multiset_error = 0.0
    n_parts = 0
    for ha, hb, unused in partitions_2v2():
        ya = np.mean([run_fields[r] for r in ha], axis=0)
        yb = np.mean([run_fields[r] for r in hb], axis=0)
        ia, aa = component_rdm(ya)
        ib, ab = component_rdm(yb)
        obs_i.append(scalar_spearman(ia, ib))
        obs_a.append(scalar_spearman(aa, ab))

        nia, naa, e1, e2 = shuffled_rdms(ya, rng, B, batch_size)
        nib, nab, e3, e4 = shuffled_rdms(yb, rng, B, batch_size)
        null_i += row_spearman(nia, nib)
        null_a += row_spearman(naa, nab)
        max_amp_error = max(max_amp_error, e1, e3)
        phase_multiset_error = max(phase_multiset_error, e2, e4)
        n_parts += 1
    return (
        np.asarray([np.nanmean(obs_i), np.nanmean(obs_a)], float),
        np.asarray([null_i / n_parts, null_a / n_parts], float),
        max_amp_error,
        phase_multiset_error,
    )


def cache_path(cache_root: Path, cohort: str, subject: str) -> Path:
    return cache_root / f"{cohort}__{subject}.npz"


def load_cache(path: Path, protocol_hash: str):
    if not path.exists(): return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["protocol_hash"].item()) != protocol_hash: return None
        return {k: z[k] for k in z.files}
    except Exception:
        return None


def run_subject(data_root: Path, subject: str, cohort: str, cfg: dict, protocol_hash: str, cp: Path, log):
    c = load_cache(cp, protocol_hash)
    if c is not None:
        log(f"{subject}: cache OK")
        return c
    rank = int(cfg["frozen_signal_pipeline"]["common_subspace_rank"])
    B = int(cfg["phase_null"]["B"])
    batch_size = int(cfg["phase_null"].get("batch_size", 32))
    seed = int(cfg["phase_null"]["seed"]) + stable_subject_seed(subject)
    data = load_subject(data_root, subject, cfg)
    fields = {r: build_field(data[r], rank) for r in RUNS}
    obs, null, amp_err, phase_err = subject_scores_and_null(fields, B, seed, batch_size)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cp,
        protocol_hash=np.asarray(protocol_hash), subject=np.asarray(subject), cohort=np.asarray(cohort),
        observed=obs, null=null, amplitude_audit=np.asarray(amp_err), phase_multiset_audit=np.asarray(phase_err),
    )
    return load_cache(cp, protocol_hash)


def load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"cohort", "subject", "INTERFERENCE", "ANGLE"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"Reference table missing columns: {sorted(needed - set(df.columns))}")
    return df


def check_reference(cohort: str, subjects: list[str], observed: np.ndarray, ref: pd.DataFrame, tol: float):
    r = ref[ref["cohort"].astype(str) == cohort].copy().set_index("subject")
    diffs = []
    rows = []
    for n, s in enumerate(subjects):
        if s not in r.index:
            raise RuntimeError(f"PAPER02 reference missing subject {cohort}/{s}")
        for j, k in enumerate(COMPONENTS):
            rv = float(r.loc[s, k]); ov = float(observed[n, j]); d = abs(ov - rv)
            diffs.append(d)
            rows.append({"cohort": cohort, "subject": s, "component": k, "paper03_recomputed": ov, "paper02_reference": rv, "abs_diff": d})
    mx = max(diffs) if diffs else 0.0
    return mx <= tol, mx, rows


def component_summary(obs: np.ndarray, null: np.ndarray, cfg: dict, seed: int):
    observed_mean = float(np.mean(obs))
    pop_null = np.mean(null, axis=0)
    null_mean = float(np.mean(pop_null))
    null_q = [float(np.quantile(pop_null, 0.025)), float(np.quantile(pop_null, 0.975))]
    empirical_p = (1 + int(np.sum(pop_null >= observed_mean - 1e-15))) / (len(pop_null) + 1)
    subject_null_mean = np.mean(null, axis=1)
    paired = obs - subject_null_mean
    paired_mean, lo, hi = bootstrap_mean_ci(paired, int(cfg["bootstrap_B"]), seed)
    _, paired_p, mode, Bsf = signflip_one_sided(paired, int(cfg["signflip_B"]), seed + 1)
    return {
        "observed_mean": observed_mean,
        "phase_null_population_mean": null_mean,
        "phase_null_population_q025_q975": null_q,
        "observed_minus_null_mean": observed_mean - null_mean,
        "empirical_p_observed_gt_phase_null": empirical_p,
        "paired_subject_effect_mean": paired_mean,
        "paired_subject_effect_ci95": [lo, hi],
        "paired_subject_signflip_p_one_sided": paired_p,
        "paired_subject_signflip_mode": mode,
        "paired_subject_signflip_B": Bsf,
        "fraction_subject_effect_positive": float(np.mean(paired > 0)),
        "N": int(len(obs)),
        "null_B": int(null.shape[1]),
    }, subject_null_mean, paired


def aggregate(cohort: str, subjects: list[str], caches: list[dict], cfg: dict, outdir: Path, ref: pd.DataFrame):
    observed = np.vstack([c["observed"] for c in caches])
    null = np.stack([c["null"] for c in caches], axis=0)  # N x 2 x B
    tol = float(cfg["paper02_reference_tolerance"])
    ref_ok, ref_max, ref_rows = check_reference(cohort, subjects, observed, ref, tol)
    if not ref_ok:
        raise RuntimeError(f"{cohort}: recomputed PAPER03 observed values do not match PAPER02 reference; max abs diff={ref_max:.3e} > {tol:.3e}")
    seed0 = int(cfg["signflip_seed"]) + (10000 if cohort.startswith("CB") else 0)
    summary = {
        "cohort": cohort,
        "N": len(subjects),
        "subjects": subjects,
        "components": {},
        "reference_consistency": {"PASS": ref_ok, "max_abs_diff": ref_max, "tolerance": tol},
        "audits": {
            "max_amplitude_error": max(float(c["amplitude_audit"].item()) for c in caches),
            "max_phase_multiset_error": max(float(c["phase_multiset_audit"].item()) for c in caches),
        },
    }
    subject_rows = []
    for j, k in enumerate(COMPONENTS):
        rec, subj_null, paired = component_summary(observed[:, j], null[:, j, :], cfg, seed0 + 100*j)
        summary["components"][k] = rec
        for n, s in enumerate(subjects):
            subject_rows.append({
                "cohort": cohort, "subject": s, "component": k,
                "observed": float(observed[n, j]), "subject_phase_null_mean": float(subj_null[n]),
                "observed_minus_null": float(paired[n]),
            })
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "SUMMARY.json", summary)
    pd.DataFrame(subject_rows).to_csv(outdir / "SUBJECT_PHASE_NULL.csv", index=False)
    pd.DataFrame(ref_rows).to_csv(outdir / "PAPER02_REFERENCE_AUDIT.csv", index=False)
    # compact population null arrays for independent auditing/replotting
    np.savez_compressed(outdir / "POPULATION_NULLS.npz",
                        INTERFERENCE=np.mean(null[:, 0, :], axis=0),
                        ANGLE=np.mean(null[:, 1, :], axis=0))
    return summary


def endpoint_pass(rec: dict, alpha: float) -> bool:
    return bool(
        rec["observed_minus_null_mean"] > 0
        and rec["paired_subject_effect_mean"] > 0
        and rec["empirical_p_observed_gt_phase_null"] < alpha
        and rec["paired_subject_signflip_p_one_sided"] < alpha
    )


def classify(summaries: dict[str, dict], alpha: float):
    passes = {k: {} for k in COMPONENTS}
    for comp in COMPONENTS:
        for cohort, s in summaries.items():
            passes[comp][cohort] = endpoint_pass(s["components"][comp], alpha)
    i_both = all(passes["INTERFERENCE"].values())
    a_both = all(passes["ANGLE"].values())
    if i_both and a_both:
        decision = "PHASE_ORGANIZATION_CONFIRMED"
    elif i_both and not a_both:
        decision = "WEIGHTED_PHASE_ORGANIZATION_ONLY"
    elif a_both and not i_both:
        decision = "ANGULAR_PHASE_ORGANIZATION_ONLY"
    else:
        decision = "PHASE_ORGANIZATION_NOT_CONFIRMED"
    return {
        "decision": decision,
        "alpha_one_sided": alpha,
        "pass_by_component_and_cohort": passes,
        "guardrail": "PAPER03 is a frozen phase-null adjudication on previously analyzed cohorts; it is not a new independent acquisition and does not establish causality, PAC, a universal phase code, or consciousness specificity."
    }


def make_figures(results_root: Path, summaries: dict[str, dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir = results_root / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    cohorts = list(summaries)
    x = np.arange(len(COMPONENTS) * len(cohorts))
    labels, obs, nul = [], [], []
    qlo, qhi = [], []
    for comp in COMPONENTS:
        for cohort in cohorts:
            r = summaries[cohort]["components"][comp]
            labels.append(f"{cohort}\n{comp}")
            obs.append(r["observed_mean"]); nul.append(r["phase_null_population_mean"])
            qlo.append(r["phase_null_population_q025_q975"][0]); qhi.append(r["phase_null_population_q025_q975"][1])
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - w/2, obs, w, label="Observed")
    ax.bar(x + w/2, nul, w, label="Phase-null mean")
    for xi, lo, hi in zip(x, qlo, qhi):
        ax.vlines(xi + w/2, lo, hi, linewidth=2)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean cross-run Spearman reliability")
    ax.set_title("PAPER03: observed phase geometry versus amplitude-preserving phase null")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir / "FIG_PAPER03_OBSERVED_VS_PHASE_NULL.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pos = 1
    xt, xl = [], []
    for comp in COMPONENTS:
        for cohort in cohorts:
            csv = pd.read_csv(results_root / cohort / "SUBJECT_PHASE_NULL.csv")
            vals = csv[csv["component"] == comp]["observed_minus_null"].to_numpy(float)
            ax.boxplot(vals, positions=[pos], widths=0.6)
            xt.append(pos); xl.append(f"{cohort}\n{comp}")
            pos += 1
        pos += 0.4
    ax.axhline(0, linewidth=1)
    ax.set_xticks(xt, xl, rotation=20, ha="right")
    ax.set_ylabel("Participant observed − phase-null mean")
    ax.set_title("PAPER03: paired participant effects")
    fig.tight_layout(); fig.savefig(figdir / "FIG_PAPER03_PAIRED_EFFECTS.png", dpi=180); plt.close(fig)


def write_article_draft(results_root: Path, summaries: dict[str, dict], decision: dict):
    lines = [
        "# PAPER03 article-ready result draft",
        "",
        "PAPER03 was frozen after the PAPER02 component analysis and before any PAPER03 phase-null outcome was inspected. It directly tests whether the reproducible INTERFERENCE and ANGLE geometries exceed an amplitude-preserving category-phase permutation null.",
        "",
        f"Frozen decision: **{decision['decision']}**.",
        "",
    ]
    for cohort, s in summaries.items():
        lines.append(f"## {cohort}")
        for comp in COMPONENTS:
            r = s["components"][comp]
            lines.append(
                f"{comp}: observed R={r['observed_mean']:.4f}; phase-null mean={r['phase_null_population_mean']:.4f} "
                f"(95% null interval {r['phase_null_population_q025_q975'][0]:.4f} to {r['phase_null_population_q025_q975'][1]:.4f}); "
                f"observed-null={r['observed_minus_null_mean']:.4f}; empirical p={r['empirical_p_observed_gt_phase_null']:.4g}; "
                f"paired subject effect={r['paired_subject_effect_mean']:.4f}, 95% bootstrap CI [{r['paired_subject_effect_ci95'][0]:.4f}, {r['paired_subject_effect_ci95'][1]:.4f}], "
                f"one-sided sign-flip p={r['paired_subject_signflip_p_one_sided']:.4g}."
            )
        lines += [
            f"PAPER02 observed-score reconstruction audit: max absolute difference={s['reference_consistency']['max_abs_diff']:.3e} (PASS).",
            f"Amplitude-preservation audit: max error={s['audits']['max_amplitude_error']:.3e}; phase-multiset permutation audit: max error={s['audits']['max_phase_multiset_error']:.3e}.",
            "",
        ]
    (results_root / "ARTICLE_RESULTS_DRAFT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--drives", nargs="*", default=["D:", "E:"])
    args = ap.parse_args()
    root = Path(args.package_root).resolve()
    cfg_path = root / "protocol" / "PAPER03_CONFIG.json"
    cfg = load_json(cfg_path)
    log = log_factory(root / "logs" / "RUN.log")
    protocol_hash = sha256_file(cfg_path)
    template = root / "protocol" / "SGIT_Level5A_template_graph.npz"
    ref_path = root / "protocol" / "PAPER02_SUBJECT_REFERENCE.csv"
    save_json(root / "state" / "FROZEN_PLAN_HASH.json", {
        "paper03_config_sha256": protocol_hash,
        "template_graph_sha256": sha256_file(template),
        "paper02_subject_reference_sha256": sha256_file(ref_path),
        "status": "HASHED_BEFORE_PAPER03_PHASE_NULL_OUTCOMES"
    })
    log(f"PAPER03 protocol hash: {protocol_hash}")
    log(f"Phase-null B={cfg['phase_null']['B']}; alpha={cfg['alpha_one_sided']}")

    ca_root, cb_root = resolve_data_roots(root, args.drives, log)
    if ca_root is None or cb_root is None:
        log("Could not resolve both frozen FineTF roots. Set SGIT_CA_FINETF_ROOT and SGIT_CB_FINETF_ROOT and rerun.")
        raise SystemExit(3)
    save_json(root / "state" / "DATA_PATHS.json", {"CA": str(ca_root), "CB": str(cb_root)})

    pre = []; bad = []
    for cohort, croot in [("CA_Level5A", ca_root), ("CB_Level6A_strict", cb_root)]:
        for s in cfg["cohorts"][cohort]:
            reasons = preflight_subject(croot, s, template, k_modes=int(cfg["frozen_signal_pipeline"]["K"]))
            pre.append({"cohort": cohort, "subject": s, "eligible": not reasons, "reasons": " ; ".join(reasons)})
            if reasons: bad.append((cohort, s, reasons))
    pd.DataFrame(pre).to_csv(root / "state" / "PREFLIGHT.csv", index=False)
    save_json(root / "state" / "PREFLIGHT.json", pre)
    if bad:
        log(f"Preflight FAILED for {len(bad)} frozen subjects; no PAPER03 null outcomes computed.")
        for c, s, r in bad[:20]: log(f"  {c} {s}: {'; '.join(r)}")
        raise SystemExit(4)
    log(f"Preflight PASS: {len(pre)} frozen subjects available and graph-compatible.")

    ref = load_reference(ref_path)
    summaries = {}; cache_root = root / "state" / "subject_cache"
    for cohort, croot in [("CA_Level5A", ca_root), ("CB_Level6A_strict", cb_root)]:
        subjects = cfg["cohorts"][cohort]; caches = []
        log(f"=== {cohort}: {len(subjects)} subjects ===")
        for i, s in enumerate(subjects, 1):
            log(f"{cohort} {i}/{len(subjects)}: {s}")
            caches.append(run_subject(croot, s, cohort, cfg, protocol_hash, cache_path(cache_root, cohort, s), log))
        summaries[cohort] = aggregate(cohort, subjects, caches, cfg, root / "results" / cohort, ref)
        log(f"{cohort}: aggregation complete; PAPER02 reference audit PASS")

    alpha = float(cfg["alpha_one_sided"])
    decision = classify(summaries, alpha)
    save_json(root / "results" / "PAPER03_SUMMARY.json", summaries)
    save_json(root / "results" / "PAPER03_DECISION.json", decision)
    rows = []
    for cohort, s in summaries.items():
        for comp in COMPONENTS:
            r = s["components"][comp]
            rows.append({
                "cohort": cohort, "component": comp,
                "observed_mean": r["observed_mean"],
                "phase_null_mean": r["phase_null_population_mean"],
                "phase_null_q025": r["phase_null_population_q025_q975"][0],
                "phase_null_q975": r["phase_null_population_q025_q975"][1],
                "observed_minus_null": r["observed_minus_null_mean"],
                "empirical_p": r["empirical_p_observed_gt_phase_null"],
                "paired_effect": r["paired_subject_effect_mean"],
                "paired_ci_low": r["paired_subject_effect_ci95"][0],
                "paired_ci_high": r["paired_subject_effect_ci95"][1],
                "paired_signflip_p": r["paired_subject_signflip_p_one_sided"],
            })
    pd.DataFrame(rows).to_csv(root / "results" / "PAPER03_ARTICLE_TABLE.csv", index=False)
    make_figures(root / "results", summaries)
    write_article_draft(root / "results", summaries, decision)
    (root / "state" / "RUN_COMPLETE.flag").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    log(f"PAPER03 COMPLETE. Frozen decision: {decision['decision']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(10)
