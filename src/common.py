from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau

CATS = ["face", "object", "letter", "false"]
RUNS = [1, 2, 3, 4, 5]
TRIU = np.triu_indices(4, 1)


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable_subject_seed(subject: str) -> int:
    d = hashlib.sha256(subject.encode("utf-8")).digest()
    return int.from_bytes(d[:4], "little")


def rho_spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(np.dot(ra, rb) / den) if den else np.nan


def rho_pearson(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else np.nan


def tau_kendall(a, b) -> float:
    r = kendalltau(np.asarray(a, float), np.asarray(b, float), variant="b", nan_policy="omit")
    return float(r.statistic) if np.isfinite(r.statistic) else np.nan


def assoc(a, b, metric: str) -> float:
    if metric == "spearman":
        return rho_spearman(a, b)
    if metric == "pearson":
        return rho_pearson(a, b)
    if metric == "kendall":
        return tau_kendall(a, b)
    raise ValueError(metric)


def partitions_2v2():
    out = []
    seen = set()
    for a in itertools.combinations(RUNS, 2):
        rem = [x for x in RUNS if x not in a]
        for b in itertools.combinations(rem, 2):
            key = tuple(sorted([tuple(a), tuple(b)]))
            if key not in seen:
                seen.add(key)
                unused = [x for x in RUNS if x not in set(a) | set(b)]
                assert len(unused) == 1
                out.append((tuple(a), tuple(b), unused[0]))
    assert len(out) == 15
    return out


def decomposed_rdm(y: np.ndarray) -> dict[str, np.ndarray]:
    """y shape: (4, coordinates..., time...) but flattened by summation."""
    A = np.abs(y)
    phi = np.angle(y)
    amp, ap, full = [], [], []
    for i, j in zip(*TRIU):
        da = float(np.sum((A[i] - A[j]) ** 2))
        dp = float(np.sum(2 * A[i] * A[j] * (1 - np.cos(phi[i] - phi[j]))))
        df = float(np.sum(np.abs(y[i] - y[j]) ** 2))
        amp.append(da); ap.append(dp); full.append(df)
    return {"AMP": np.asarray(amp), "AP": np.asarray(ap), "FULL": np.asarray(full)}


def bootstrap_mean_ci(vals, B: int, seed: int, alpha: float = 0.05):
    x = np.asarray(vals, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    n = len(x)
    # memory-safe batches
    samples = np.empty(B, dtype=float)
    batch = 2000
    pos = 0
    while pos < B:
        m = min(batch, B - pos)
        idx = rng.integers(0, n, size=(m, n))
        samples[pos:pos+m] = x[idx].mean(axis=1)
        pos += m
    lo, hi = np.quantile(samples, [alpha/2, 1-alpha/2])
    return float(x.mean()), float(lo), float(hi)


def signflip_one_sided(vals, B: int, seed: int):
    x = np.asarray(vals, float)
    x = x[np.isfinite(x)]
    obs = float(x.mean())
    n = len(x)
    if n <= 20:
        ge = 0; tot = 0
        for sg in itertools.product([-1.0, 1.0], repeat=n):
            tot += 1
            if np.mean(x * np.asarray(sg)) >= obs - 1e-15:
                ge += 1
        return obs, ge / tot, "exact", tot
    rng = np.random.default_rng(seed)
    ge = 0
    batch = 5000
    for start in range(0, B, batch):
        m = min(batch, B - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
        null = (signs * x).mean(axis=1)
        ge += int(np.sum(null >= obs - 1e-15))
    # plus-one correction for robustness analyses
    return obs, (ge + 1) / (B + 1), "monte_carlo_plus_one", B


def random_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(k, k))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R)); s[s == 0] = 1.0
    Q = Q * s[None, :]
    # deterministic column orientation
    for j in range(k):
        i = int(np.argmax(np.abs(Q[:, j])))
        if Q[i, j] < 0:
            Q[:, j] *= -1
    return Q.astype(np.float64)


def fit_pca_rotation(raw_window: np.ndarray) -> np.ndarray:
    """raw_window: trials x K x T. Return Q with columns = PCA axes in old coordinates."""
    X = np.asarray(raw_window, float).transpose(0, 2, 1).reshape(-1, raw_window.shape[1])
    X = X - X.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    w, V = np.linalg.eigh(C)
    Q = V[:, np.argsort(w)[::-1]]
    for j in range(Q.shape[1]):
        i = int(np.argmax(np.abs(Q[:, j])))
        if Q[i, j] < 0:
            Q[:, j] *= -1
    return Q.astype(np.float64)


def rotate_modes(arr: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """arr: trials x K x T, Q columns are new axes. y = Q^T x."""
    return np.einsum("kj,nkt->njt", Q, arr, optimize=True)


def is_finetf_root(path: Path, prefix: str | None = None) -> bool:
    try:
        if not ((path / "meeg_fine").is_dir() and (path / "metadata").is_dir() and (path / "graphs").is_dir()):
            return False
        pat = "sub-*_*run-01_sgit_level4e_finetf.npz"
        files = list((path / "meeg_fine").glob(pat))
        if not files:
            return False
        if prefix:
            return any(f.name.startswith(f"sub-{prefix}") for f in files)
        return True
    except OSError:
        return False


def score_root(path: Path, prefix: str) -> int:
    score = 0
    s = str(path).lower()
    if prefix == "CA" and path.name.lower() == "cohort_finetf": score += 20
    if prefix == "CB" and path.name.lower() == "cohort_finetf_cb": score += 20
    if "sgit" in s: score += 10
    if "cogitate" in s: score += 8
    if "population_runner" in s: score += 6
    if "level6" in s: score += 6
    try:
        score += min(30, len(list((path / "meeg_fine").glob(f"sub-{prefix}*_run-01_sgit_level4e_finetf.npz"))))
    except OSError:
        pass
    return score


def discover_finetf_root(drives: list[str], prefix: str, env_var: str, log) -> Path | None:
    override = os.environ.get(env_var, "").strip().strip('"')
    if override:
        p = Path(override)
        if is_finetf_root(p, prefix):
            log(f"{prefix}: using {env_var}={p}")
            return p.resolve()
        log(f"{prefix}: environment override invalid: {p}")

    candidates: list[Path] = []
    known_rel = [
        Path("SGIT_POP_V2/SGIT_COGITATE_Population_Runner_v2_1/cohort_finetf"),
        Path("SGIT_POP_V2/SGIT_COGITATE_Population_Runner_v2_2/cohort_finetf"),
        Path("SGIT_COGITATE_Population_Runner_v2_1/cohort_finetf"),
        Path("SGIT_COGITATE_Population_Runner_v2_2/cohort_finetf"),
        Path("SGIT_POP_V2/SGIT_LEVEL6_AllInOne_v3_0/SGIT_LEVEL6_AllInOne_v3_1/level6_data/cohort_finetf_cb"),
        Path("SGIT_POP_V2/SGIT_LEVEL6_AllInOne_v3_1/level6_data/cohort_finetf_cb"),
        Path("SGIT_LEVEL6_AllInOne_v3_1/level6_data/cohort_finetf_cb"),
    ]
    for d in drives:
        base = Path(d + "\\") if os.name == "nt" else Path(d)
        for rel in known_rel:
            p = base / rel
            if is_finetf_root(p, prefix):
                candidates.append(p)

    # bounded directory BFS. We only inspect directory names, never dataset contents.
    skip_names = {"system volume information", "$recycle.bin", "windows", "program files", "program files (x86)", "recovery", "node_modules", ".git"}
    likely_tokens = ("sgit", "cogitate", "meeg", "level", "experiment", "data", "dataset", "research", "pop")
    for d in drives:
        base = Path(d + "\\") if os.name == "nt" else Path(d)
        if not base.exists():
            log(f"drive not available: {base}")
            continue
        queue = [(base, 0)]
        seen = set()
        while queue:
            p, depth = queue.pop(0)
            try:
                key = str(p.resolve()).lower()
            except Exception:
                key = str(p).lower()
            if key in seen:
                continue
            seen.add(key)
            if p.name.lower() in skip_names:
                continue
            if is_finetf_root(p, prefix):
                candidates.append(p)
                continue
            if depth >= 6:
                continue
            try:
                children = [x for x in p.iterdir() if x.is_dir()]
            except (OSError, PermissionError):
                continue
            for ch in children:
                nm = ch.name.lower()
                if nm in skip_names:
                    continue
                # At deeper levels only descend through plausible research folders.
                if depth >= 2 and not any(tok in nm for tok in likely_tokens) and nm not in {"cohort_finetf", "cohort_finetf_cb", "level6_data"}:
                    # allow one level under a plausible parent
                    parent_nm = p.name.lower()
                    if not any(tok in parent_nm for tok in likely_tokens):
                        continue
                queue.append((ch, depth + 1))

    uniq = []
    seen = set()
    for p in candidates:
        try: rp = p.resolve()
        except Exception: rp = p
        k = str(rp).lower()
        if k not in seen:
            seen.add(k); uniq.append(rp)
    if not uniq:
        return None
    uniq.sort(key=lambda p: score_root(p, prefix), reverse=True)
    for p in uniq[:10]:
        log(f"{prefix} candidate score={score_root(p,prefix)}: {p}")
    return uniq[0]


def paths_for(root: Path, subject: str, run: int):
    fine = list((root / "meeg_fine").glob(f"sub-{subject}_*run-{run:02d}_sgit_level4e_finetf.npz"))
    meta = list((root / "metadata").glob(f"sub-{subject}_*run-{run:02d}_trials.csv"))
    graph = list((root / "graphs").glob(f"sub-{subject}_*run-{run:02d}_meeg_graph.npz"))
    return (
        fine[0] if len(fine) == 1 else None,
        meta[0] if len(meta) == 1 else None,
        graph[0] if len(graph) == 1 else None,
    )


def graph_compatible(path: Path, template: Path):
    z = np.load(path, allow_pickle=True)
    t = np.load(template, allow_pickle=True)
    keys = ["channel_names", "coords", "eigenvalues", "eigenvectors"]
    for k in keys:
        if k not in z.files or k not in t.files:
            return False, f"missing graph key {k}"
        a, b = z[k], t[k]
        if a.shape != b.shape:
            return False, f"{k} shape {a.shape} != {b.shape}"
        if a.dtype.kind in "USO":
            if not np.array_equal(a, b):
                return False, f"{k} differs"
        else:
            if not np.allclose(a, b, rtol=0, atol=1e-7):
                return False, f"{k} differs"
    return True, "ok"


def preflight_subject(root: Path, subject: str, template: Path, k_modes: int = 16, min_trials: int = 32):
    reasons = []
    for r in RUNS:
        ff, mf, gf = paths_for(root, subject, r)
        if not ff: reasons.append(f"missing_fine_run{r}")
        if not mf: reasons.append(f"missing_metadata_run{r}")
        if not gf: reasons.append(f"missing_graph_run{r}")
        if ff:
            try:
                # Fast preflight: inspect NPZ member names only. Full shape/finite validation
                # is performed exactly once when the subject is actually loaded.
                with np.load(ff, allow_pickle=False) as z:
                    if "gft_trajectories" not in z.files or "times_s" not in z.files:
                        reasons.append(f"invalid_fine_keys_run{r}")
            except Exception as e:
                reasons.append(f"fine_read_error_run{r}:{type(e).__name__}")
        if mf:
            try:
                meta = pd.read_csv(mf)
                if "sgit_category" not in meta.columns:
                    reasons.append(f"missing_category_run{r}")
                else:
                    vc = meta["sgit_category"].astype(str).value_counts()
                    for c in CATS:
                        if int(vc.get(c, 0)) < min_trials:
                            reasons.append(f"run{r}_{c}_trials={int(vc.get(c,0))}<{min_trials}")
            except Exception as e:
                reasons.append(f"meta_read_error_run{r}:{type(e).__name__}")
        if gf:
            ok, msg = graph_compatible(gf, template)
            if not ok:
                reasons.append(f"graph_run{r}:{msg}")
    return sorted(set(reasons))
