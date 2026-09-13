#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGIT COGITATE Converter
=======================
Vytvori kompaktní trial-level derivative dataset pro SGIT Level-2 z BIDS dat
COGITATE. Program je určen k lokálnímu spuštění na počítači uživatele.

Výstup je záměrně odvozený a malý; raw neurodata se do ZIPu nekopírují.

Hlavní principy:
- zachovat jednotlivé trialy a experimentální metadata,
- fMRI: coarse parcel responses -> graph -> GFT,
- MEEG/iEEG: complex Fourier coefficients in time/frequency windows -> graph -> GFT,
- uložit U, lambda, adjacency a trial-level modal coefficients pro pozdější
  výpočet stabilních bivectorů/trivectorů a cross-modal RDM.

Tento nástroj je populační Level-2 converter, nikoli náhrada plného
oficiálního preprocessing pipeline COGITATE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import traceback
import zipfile
import tempfile
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import linalg, signal

APP_VERSION = "1.2.0"
DEFAULT_K = 16
DEFAULT_MAX_SUBJECTS = 30
DEFAULT_FMRI_GRID = (4, 4, 2)  # <= 32 coarse parcels
DEFAULT_KNN = 6
DEFAULT_FMRI_HRF_WINDOW = (4.0, 8.0)
DEFAULT_EPHYS_EPOCH = (-0.20, 1.30)
DEFAULT_TF_WINDOWS = [
    (0.05, 0.25),
    (0.25, 0.50),
    (0.50, 0.80),
    (0.80, 1.20),
]
DEFAULT_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 70.0),
}
DEFAULT_IEEG_EXTRA_BANDS = {
    "high_gamma": (70.0, 120.0),
}

SUPPORTED_MODALITIES = ("fmri", "meeg", "ieeg")
LEVEL2B_COMPACT_MEEG = True


class UserFacingError(RuntimeError):
    pass


class Logger:
    def __init__(self, path: Path, callback: Optional[Callable[[str], None]] = None):
        self.path = path
        self.callback = callback
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        line = str(msg)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.callback:
            try:
                self.callback(line)
            except Exception:
                pass


@dataclass
class RunRecord:
    modality: str
    subject: str
    session: str
    task: str
    run: str
    events_file: str
    source_file: str
    quality: str
    n_trials_input: int = 0
    n_trials_output: int = 0
    n_nodes: int = 0
    n_modes: int = 0
    graph_source: str = ""
    notes: str = ""


@dataclass
class Settings:
    bids_root: str
    output_dir: str
    max_subjects: int = DEFAULT_MAX_SUBJECTS
    k_modes: int = DEFAULT_K
    do_fmri: bool = True
    do_meeg: bool = True
    do_ieeg: bool = True
    fmri_grid_x: int = DEFAULT_FMRI_GRID[0]
    fmri_grid_y: int = DEFAULT_FMRI_GRID[1]
    fmri_grid_z: int = DEFAULT_FMRI_GRID[2]
    knn: int = DEFAULT_KNN
    zip_files: tuple[str, ...] = ()
    source_archives: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Utility / BIDS discovery
# ---------------------------------------------------------------------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_bids_roots(selected_root: Path) -> list[Path]:
    """Find BIDS roots below selected directory.

    A BIDS root is identified by dataset_description.json. If none is found,
    selected_root itself is returned so that a best-effort scan can continue.
    """
    selected_root = selected_root.resolve()
    roots: list[Path] = []
    if (selected_root / "dataset_description.json").exists():
        roots.append(selected_root)

    # Limit search depth to avoid walking massive derivative trees pointlessly.
    for p in selected_root.glob("**/dataset_description.json"):
        try:
            root = p.parent.resolve()
            rel = root.relative_to(selected_root)
            if len(rel.parts) <= 4 and root not in roots:
                # Derivative roots are useful for detection but primary BIDS roots
                # are preferred. Keep all; later event scan will naturally choose raw.
                roots.append(root)
        except Exception:
            continue

    if not roots:
        roots = [selected_root]

    # Prefer roots containing sub-* and events files.
    def score(r: Path) -> tuple[int, int]:
        has_sub = int(any(r.glob("sub-*")))
        has_events = int(any(r.glob("sub-*/**/*_events.tsv")))
        return (has_events, has_sub)

    roots = sorted(set(roots), key=lambda r: score(r), reverse=True)
    return roots


def parse_bids_entities(path: Path) -> dict[str, str]:
    text = path.name
    out = {"subject": "", "session": "", "task": "", "run": "", "acq": ""}
    patterns = {
        "subject": r"(?:^|_)sub-([^_]+)",
        "session": r"(?:^|_)ses-([^_]+)",
        "task": r"(?:^|_)task-([^_]+)",
        "run": r"(?:^|_)run-([^_]+)",
        "acq": r"(?:^|_)acq-([^_]+)",
    }
    # Parent folders may carry sub/ses when not repeated in filename.
    combined = "_".join([p.name for p in path.parents if p.name][:4][::-1] + [text])
    for key, pat in patterns.items():
        m = re.search(pat, combined)
        if m:
            out[key] = m.group(1)
    return out


def infer_modality(events_path: Path) -> str:
    parts = [p.lower() for p in events_path.parts]
    parent = events_path.parent.name.lower()
    if parent == "ieeg" or "ieeg" in parts:
        return "ieeg"
    if parent in ("meg", "eeg") or "meg" in parts or "eeg" in parts:
        return "meeg"
    if parent == "func" or "func" in parts:
        return "fmri"
    return "unknown"


def scan_events(selected_root: Path, logger: Logger) -> list[tuple[Path, str]]:
    logger.log(f"Hledám BIDS *_events.tsv pod: {selected_root}")
    events = []
    seen = set()
    for p in selected_root.glob("**/*_events.tsv"):
        # Avoid derivatives; we want original trial annotations.
        low_parts = [x.lower() for x in p.parts]
        if "derivatives" in low_parts:
            continue
        mod = infer_modality(p)
        if mod == "unknown":
            continue
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        events.append((p, mod))
    events.sort(key=lambda x: natural_key(str(x[0])))
    logger.log(f"Nalezeno event souborů: {len(events)}")
    for mod in SUPPORTED_MODALITIES:
        logger.log(f"  {mod}: {sum(1 for _, m in events if m == mod)}")
    return events


def choose_subjects(events: list[tuple[Path, str]], max_subjects: int, modalities: set[str]):
    subjects_by_mod: dict[str, list[str]] = {m: [] for m in modalities}
    for p, mod in events:
        if mod not in modalities:
            continue
        sub = parse_bids_entities(p)["subject"] or "unknown"
        if sub not in subjects_by_mod[mod]:
            subjects_by_mod[mod].append(sub)
    selected: dict[str, set[str]] = {}
    for mod, subs in subjects_by_mod.items():
        subs = sorted(subs, key=natural_key)
        if max_subjects > 0:
            subs = subs[:max_subjects]
        selected[mod] = set(subs)
    return selected




# ---------------------------------------------------------------------------
# ZIP streaming / selective extraction
# ---------------------------------------------------------------------------

def _zip_member_parts(name: str) -> tuple[str, ...]:
    return tuple(p for p in name.replace("\\", "/").split("/") if p and p not in (".",))


def _short_zip_relative_parts(name: str) -> tuple[str, ...]:
    """Map an archive member to a short, BIDS-relevant relative path.

    Public/research ZIPs are often created on Linux clusters and may contain
    prefixes such as ``mnt/beegfs/workspace/...``. Reproducing that prefix on
    Windows can exceed the legacy MAX_PATH limit even when only one subject is
    extracted. We therefore keep only the semantically relevant BIDS suffix:
    ``derivatives/...`` for derivative files, otherwise from the first
    ``sub-*`` directory. Shared metadata are flattened to their basename.
    """
    parts = _zip_member_parts(name)
    if not parts:
        return ()
    low = [p.lower() for p in parts]
    if "derivatives" in low:
        i = low.index("derivatives")
        return parts[i:]
    for i, part in enumerate(low):
        if re.fullmatch(r"sub-[^/]+", part, flags=re.IGNORECASE):
            return parts[i:]
    # Root/codebook metadata selected by _zip_member_needed: keep only a short name.
    return (parts[-1],)


def _zip_member_is_safe(name: str) -> bool:
    parts = _zip_member_parts(name)
    return bool(parts) and ".." not in parts and not name.startswith(("/", "\\"))


def _zip_event_records(zip_files: Iterable[str], logger: Logger) -> list[tuple[str, str, str, str]]:
    """Return (zip_path, member_name, modality, subject) for BIDS events.tsv entries."""
    records: list[tuple[str, str, str, str]] = []
    for zstr in zip_files:
        zp = Path(zstr).expanduser().resolve()
        if not zp.exists():
            raise UserFacingError(f"ZIP soubor neexistuje: {zp}")
        logger.log(f"Čtu seznam souborů v ZIP: {zp.name} ({zp.stat().st_size/1024**3:.1f} GB)")
        try:
            with zipfile.ZipFile(zp, "r", allowZip64=True) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename.replace("\\", "/")
                    low = name.lower()
                    if not low.endswith("_events.tsv") or "/derivatives/" in f"/{low}":
                        continue
                    pp = Path(name)
                    mod = infer_modality(pp)
                    if mod == "unknown":
                        continue
                    sub = parse_bids_entities(pp).get("subject", "") or "unknown"
                    records.append((str(zp), name, mod, sub))
        except zipfile.BadZipFile as e:
            raise UserFacingError(f"Soubor není platný ZIP nebo je poškozen: {zp}") from e
    logger.log(f"V archivech nalezeno BIDS event souborů: {len(records)}")
    return records


def _select_zip_subjects(records: list[tuple[str, str, str, str]], modalities: set[str], max_subjects: int) -> dict[str, set[str]]:
    by_mod: dict[str, list[str]] = {m: [] for m in modalities}
    for _, _, mod, sub in records:
        if mod not in modalities:
            continue
        if sub not in by_mod[mod]:
            by_mod[mod].append(sub)
    out: dict[str, set[str]] = {}
    for mod, subs in by_mod.items():
        subs = sorted(subs, key=natural_key)
        if max_subjects > 0:
            subs = subs[:max_subjects]
        out[mod] = set(subs)
    return out


def _zip_member_needed(name: str, selected_subjects_union: set[str]) -> bool:
    """Select complete selected-subject trees plus tiny root/codebook metadata."""
    if not _zip_member_is_safe(name):
        return False
    parts = _zip_member_parts(name)
    lowparts = [p.lower() for p in parts]
    subject_segments = {f"sub-{s}".lower() for s in selected_subjects_union}
    if any(p.lower() in subject_segments for p in parts):
        return True
    base = parts[-1].lower()
    # Small shared BIDS descriptors can sit outside subject trees.
    if base == "dataset_description.json":
        return True
    if base.startswith("task-") and base.endswith(".json"):
        return True
    if base.endswith("_events.json"):
        return True
    if base in ("readme", "readme.txt", "changes", "changes.txt", "participants.json"):
        return True
    return False


def extract_selected_zip_subset(zip_files: Iterable[str], dest_root: Path, modalities: set[str],
                                max_subjects: int, logger: Logger) -> tuple[Path, dict[str, set[str]], list[str]]:
    """Extract only selected subjects from huge ZIP archives.

    ZIPs remain unchanged. Only selected subject trees are temporarily extracted;
    the temporary directory is deleted after SGIT derivatives are produced.
    """
    zip_files = tuple(str(Path(z).expanduser().resolve()) for z in zip_files)
    if not zip_files:
        raise UserFacingError("Nebyly vybrány žádné ZIP soubory.")
    records = _zip_event_records(zip_files, logger)
    if not records:
        raise UserFacingError("V ZIP archivech jsem nenašel žádné BIDS *_events.tsv soubory.")
    selected = _select_zip_subjects(records, modalities, max_subjects)
    for mod in sorted(modalities):
        logger.log(f"ZIP – vybrané subjekty {mod}: {', '.join(sorted(selected.get(mod, []), key=natural_key)) or 'žádné'}")
    if "meeg" in modalities and max_subjects >= 20 and len(selected.get("meeg", set())) < 20:
        logger.log("VAROVÁNÍ LEVEL-2b: tento ZIP obsahuje méně než 20 MEEG subjektů. "
                   "Pravděpodobně jde o SAMPLE balík. Převod proběhne, ale pro Level-2b-full "
                   "bude potřeba FULL MEEG BIDS archiv s alespoň 20–30 subjekty.")
    selected_union = set().union(*selected.values()) if selected else set()
    if not selected_union:
        raise UserFacingError("V ZIP archivech nejsou subjekty pro zvolené modality.")

    # Estimate uncompressed temporary disk requirement before extracting.
    total_uncompressed = 0
    per_zip_members: list[tuple[Path, list[zipfile.ZipInfo]]] = []
    for zstr in zip_files:
        zp = Path(zstr)
        with zipfile.ZipFile(zp, "r", allowZip64=True) as zf:
            members = [i for i in zf.infolist() if (not i.is_dir()) and _zip_member_needed(i.filename, selected_union)]
            total_uncompressed += sum(max(0, int(i.file_size)) for i in members)
            # ZipInfo remains usable for extraction by filename; store names robustly.
            per_zip_members.append((zp, members))
    free = shutil.disk_usage(dest_root.parent).free
    need_gb = total_uncompressed / 1024**3
    free_gb = free / 1024**3
    logger.log(f"Dočasně bude potřeba přibližně {need_gb:.1f} GB; volné místo {free_gb:.1f} GB.")
    if total_uncompressed > free * 0.85:
        raise UserFacingError(
            f"Pro dočasné rozbalení vybraných subjektů je odhad {need_gb:.1f} GB, "
            f"ale volných je jen {free_gb:.1f} GB. Nastavte pro první běh 1 subjekt/modalitu "
            "nebo zvolte výstupní disk s větší kapacitou."
        )

    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    extracted_names: list[str] = []
    for zi, (zp, members) in enumerate(per_zip_members, start=1):
        archive_root = dest_root / f"archive_{zi:02d}"
        archive_root.mkdir(parents=True, exist_ok=True)
        logger.log(f"Selektivně rozbaluji {len(members)} souborů z {zp.name} ...")
        with zipfile.ZipFile(zp, "r", allowZip64=True) as zf:
            for n, info in enumerate(members, start=1):
                name = info.filename.replace("\\", "/")
                parts = _short_zip_relative_parts(name)
                if not parts:
                    continue
                target = archive_root.joinpath(*parts)
                # Defensive Windows path-length check. Internal work is already
                # kept under LOCALAPPDATA, but this catches pathological filenames.
                if os.name == "nt" and len(str(target)) >= 245:
                    raise UserFacingError(
                        "I po zkrácení je cesta k jednomu souboru příliš dlouhá pro Windows: "
                        f"{target.name}. Pošlete SGIT_ZIP_LOG.txt; archiv vyžaduje další mapování názvů."
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                # Stream copy; no archive-wide extraction and no full file held in RAM.
                with zf.open(info, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                extracted_names.append(name)
                if n % 200 == 0 or n == len(members):
                    logger.log(f"    {zp.name}: {n}/{len(members)} souborů")
    logger.log("Selektivní rozbalení hotovo. Původní ZIP archivy zůstaly beze změny.")
    return dest_root, selected, extracted_names


def read_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "onset" not in df.columns:
        raise UserFacingError(f"Chybí povinný sloupec onset v {path}")
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    else:
        df["duration"] = 0.0
    return df


def sanitize_metadata(df: pd.DataFrame, entities: dict[str, str], modality: str, source: Path) -> pd.DataFrame:
    """Keep event metadata but avoid participant-level demographics or free filesystem paths."""
    out = df.copy()
    # Remove columns that look like absolute paths / operator notes if present.
    drop = [c for c in out.columns if c.lower() in {
        "participant_name", "patient_name", "name", "email", "address", "operator"
    }]
    out = out.drop(columns=drop, errors="ignore")
    out.insert(0, "modality", modality)
    out.insert(1, "subject", entities.get("subject", ""))
    out.insert(2, "session", entities.get("session", ""))
    out.insert(3, "task", entities.get("task", ""))
    out.insert(4, "run", entities.get("run", ""))
    out.insert(5, "events_basename", source.name)
    return out


def make_trial_uids(meta: pd.DataFrame) -> list[str]:
    vals = []
    for i, row in meta.reset_index(drop=True).iterrows():
        vals.append(
            f"{row.get('modality','')}:sub-{row.get('subject','')}:ses-{row.get('session','')}"
            f":task-{row.get('task','')}:run-{row.get('run','')}:trial-{i+1:04d}"
        )
    return vals


# ---------------------------------------------------------------------------
# Graph / GFT
# ---------------------------------------------------------------------------

def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    x = np.asarray(coords, float)
    return np.sqrt(np.maximum(0.0, ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)))


def graph_from_coords(coords: np.ndarray, knn: int = DEFAULT_KNN) -> tuple[np.ndarray, str]:
    coords = np.asarray(coords, float)
    n = len(coords)
    if n < 2:
        raise UserFacingError("Pro graf jsou potřeba alespoň 2 uzly.")
    valid = np.all(np.isfinite(coords), axis=1) & (np.linalg.norm(coords, axis=1) > 1e-12)
    if valid.sum() < max(3, int(0.7 * n)):
        return chain_graph(n), "chain_fallback_missing_coordinates"

    # Fill missing coords by centroid so they get weak / local fallback connections.
    centroid = np.nanmedian(coords[valid], axis=0)
    coords2 = coords.copy()
    coords2[~valid] = centroid
    D = pairwise_distances(coords2)
    np.fill_diagonal(D, np.inf)
    k = min(max(1, knn), n - 1)
    neigh = np.sort(D[np.isfinite(D)])
    sigma = np.median(neigh[: max(n * k, 1)]) if neigh.size else 1.0
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = np.median(D[np.isfinite(D) & (D > 0)]) if np.any(np.isfinite(D) & (D > 0)) else 1.0
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        js = np.argsort(D[i])[:k]
        for j in js:
            if not np.isfinite(D[i, j]):
                continue
            w = math.exp(-(D[i, j] ** 2) / (2.0 * sigma ** 2))
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)
    return A, "coordinate_knn"


def chain_graph(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A


def functional_graph(sample_data: np.ndarray, knn: int = DEFAULT_KNN) -> np.ndarray:
    """Functional fallback graph from channel x sample data, labels ignored."""
    X = np.asarray(sample_data, float)
    if X.ndim != 2:
        raise ValueError("sample_data must be channels x samples")
    X = X - np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    X /= sd
    C = np.nan_to_num(np.corrcoef(X), nan=0.0, posinf=0.0, neginf=0.0)
    C = np.abs(C)
    np.fill_diagonal(C, 0.0)
    n = len(C)
    k = min(max(1, knn), n - 1)
    A = np.zeros_like(C)
    for i in range(n):
        js = np.argsort(C[i])[-k:]
        A[i, js] = C[i, js]
    A = np.maximum(A, A.T)
    return A


def gft_basis(A: np.ndarray, k_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(A, float)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)
    vals, vecs = vals[idx], vecs[:, idx]
    k = min(max(1, int(k_modes)), A.shape[0])
    U = vecs[:, :k].copy()
    # Canonical sign where possible; degenerate-subspace ambiguity remains and is
    # handled later with projectors / RDMs, so raw U is also saved.
    for j in range(U.shape[1]):
        imax = int(np.argmax(np.abs(U[:, j])))
        if U[imax, j] < 0:
            U[:, j] *= -1
    return L, vals[:k], U


def graph_from_positions_or_functional(raw, picks: np.ndarray, knn: int, logger: Logger):
    coords = np.array([raw.info["chs"][int(i)]["loc"][:3] for i in picks], dtype=float)
    valid = np.all(np.isfinite(coords), axis=1) & (np.linalg.norm(coords, axis=1) > 1e-9)
    if valid.sum() >= max(3, int(0.70 * len(coords))):
        return (*graph_from_coords(coords, knn), coords)

    logger.log("    Pozice kanálů nejsou dostatečné; tvořím label-free funkční fallback graf.")
    n_times = raw.n_times
    target = min(5000, n_times)
    idx = np.linspace(0, n_times - 1, target, dtype=int)
    # get_data cannot directly pick arbitrary discontinuous samples, so read a
    # compact set of evenly spread short blocks.
    blocks = []
    n_blocks = min(20, max(1, n_times // max(1, int(raw.info["sfreq"]))))
    block_len = max(50, target // n_blocks)
    starts = np.linspace(0, max(0, n_times - block_len - 1), n_blocks, dtype=int)
    for st in starts:
        blocks.append(raw.get_data(picks=picks, start=int(st), stop=int(min(n_times, st + block_len))))
    X = np.concatenate(blocks, axis=1) if blocks else raw.get_data(picks=picks, start=0, stop=min(n_times, 1000))
    A = functional_graph(X, knn=knn)
    # Visual coordinate placeholder only; NOT used to define graph.
    coords = np.column_stack([np.arange(len(picks), dtype=float), np.zeros(len(picks)), np.zeros(len(picks))])
    return A, "functional_correlation_fallback", coords


# ---------------------------------------------------------------------------
# fMRI
# ---------------------------------------------------------------------------

def find_fmri_bold(events_path: Path, selected_root: Path, entities: dict[str, str]) -> tuple[Optional[Path], str]:
    base_raw = Path(str(events_path).replace("_events.tsv", "_bold.nii.gz"))
    if base_raw.exists():
        raw_candidate = base_raw
    else:
        raw_candidate = None

    sub = entities.get("subject", "")
    ses = entities.get("session", "")
    task = entities.get("task", "")
    run = entities.get("run", "")
    patterns = []
    if sub:
        core = f"sub-{sub}*"
        if task:
            core += f"task-{task}*"
        if run:
            core += f"run-{run}*"
        patterns.extend([
            f"**/{core}space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz",
            f"**/{core}space-*_desc-preproc_bold.nii.gz",
            f"**/{core}desc-preproc_bold.nii.gz",
        ])
    for pattern in patterns:
        for p in selected_root.glob(pattern):
            if "fmriprep" in [x.lower() for x in p.parts]:
                return p, "FMRIPREP_PREPROCESSED"
    if raw_candidate is not None:
        return raw_candidate, "UNPREPROCESSED_APPROX"
    return None, "MISSING"


def matching_fmri_mask(bold_path: Path) -> Optional[Path]:
    name = bold_path.name
    if "desc-preproc_bold.nii.gz" in name:
        guesses = [
            bold_path.with_name(name.replace("desc-preproc_bold.nii.gz", "desc-brain_mask.nii.gz")),
            bold_path.with_name(name.replace("_bold.nii.gz", "_mask.nii.gz")),
        ]
        for g in guesses:
            if g.exists():
                return g
    return None


def matching_confounds(bold_path: Path) -> Optional[Path]:
    if "desc-preproc_bold.nii.gz" not in bold_path.name:
        return None
    prefix = bold_path.name.split("_space-")[0]
    cands = list(bold_path.parent.glob(prefix + "*desc-confounds_timeseries.tsv"))
    return cands[0] if cands else None


def build_grid_labels(mask: np.ndarray, affine: np.ndarray, grid=(4, 4, 2)) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask, bool)
    vox = np.argwhere(mask)
    if len(vox) < 100:
        raise UserFacingError("fMRI maska obsahuje příliš málo voxelů.")
    mins = vox.min(axis=0)
    maxs = vox.max(axis=0) + 1
    edges = [np.linspace(mins[d], maxs[d], grid[d] + 1) for d in range(3)]
    labels = np.zeros(mask.shape, dtype=np.int16)
    coords = []
    label = 1
    for ix in range(grid[0]):
        for iy in range(grid[1]):
            for iz in range(grid[2]):
                x0, x1 = int(math.floor(edges[0][ix])), int(math.ceil(edges[0][ix + 1]))
                y0, y1 = int(math.floor(edges[1][iy])), int(math.ceil(edges[1][iy + 1]))
                z0, z1 = int(math.floor(edges[2][iz])), int(math.ceil(edges[2][iz + 1]))
                submask = mask[x0:x1, y0:y1, z0:z1]
                if submask.sum() < 20:
                    continue
                block = labels[x0:x1, y0:y1, z0:z1]
                block[submask] = label
                labels[x0:x1, y0:y1, z0:z1] = block
                local_vox = np.argwhere(submask) + np.array([x0, y0, z0])
                center_vox = local_vox.mean(axis=0)
                center_h = np.r_[center_vox, 1.0]
                center_mm = (affine @ center_h)[:3]
                coords.append(center_mm)
                label += 1
    if label <= 3:
        raise UserFacingError("Nepodařilo se vytvořit dostatek hrubých fMRI parcel.")
    return labels, np.asarray(coords, dtype=float)


def extract_parcel_timeseries(img, mask_img, grid, logger: Logger) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import nibabel as nib  # optional runtime dependency

    shape = img.shape
    if len(shape) != 4:
        raise UserFacingError(f"BOLD NIfTI není 4D: {shape}")
    if mask_img is not None:
        mask = np.asanyarray(mask_img.dataobj) > 0
    else:
        logger.log("    Brain mask nenalezena; odhaduji masku z několika BOLD objemů.")
        tids = np.unique(np.linspace(0, shape[3] - 1, min(7, shape[3]), dtype=int))
        acc = np.zeros(shape[:3], dtype=np.float64)
        for t in tids:
            vol = np.asanyarray(img.dataobj[..., int(t)]).astype(np.float32, copy=False)
            acc += np.nan_to_num(np.abs(vol), nan=0.0)
        mask = acc > 0
        if mask.sum() < 100:
            thr = np.nanpercentile(acc[np.isfinite(acc)], 25) if np.any(np.isfinite(acc)) else 0
            mask = acc > thr

    labels, coords = build_grid_labels(mask, img.affine, grid=grid)
    lab = labels.ravel()
    n_parcels = int(lab.max())
    counts = np.bincount(lab, minlength=n_parcels + 1).astype(float)
    Y = np.zeros((shape[3], n_parcels), dtype=np.float32)
    logger.log(f"    Extrahuji {n_parcels} hrubých prostorových parcel z {shape[3]} fMRI objemů...")
    for t in range(shape[3]):
        vol = np.asanyarray(img.dataobj[..., t]).astype(np.float32, copy=False).ravel()
        sums = np.bincount(lab, weights=np.nan_to_num(vol, nan=0.0), minlength=n_parcels + 1)
        means = sums[1:] / np.maximum(counts[1:], 1.0)
        Y[t] = means.astype(np.float32)
    return Y, coords, labels


def residualize_fmri(Y: np.ndarray, confound_path: Optional[Path], logger: Logger) -> tuple[np.ndarray, str]:
    Y = np.asarray(Y, float)
    n = len(Y)
    nuisance_cols = [
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
        "csf", "white_matter", "global_signal",
    ]
    Xparts = [np.ones((n, 1)), np.linspace(-1, 1, n)[:, None]]
    quality = "detrend_zscore"
    if confound_path and confound_path.exists():
        try:
            c = pd.read_csv(confound_path, sep="\t")
            cols = [x for x in nuisance_cols if x in c.columns]
            if cols:
                C = c[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
                if len(C) >= n:
                    C = C[:n]
                    C -= C.mean(axis=0, keepdims=True)
                    sd = C.std(axis=0, keepdims=True)
                    sd[sd < 1e-12] = 1.0
                    C /= sd
                    Xparts.append(C)
                    quality = "fmriprep_confound_residualized"
                    logger.log(f"    Regresuji fMRIPrep confounds: {', '.join(cols)}")
        except Exception as e:
            logger.log(f"    VAROVÁNÍ: confounds nelze použít ({e}); zůstává detrend.")
    X = np.concatenate(Xparts, axis=1)
    beta = np.linalg.pinv(X) @ Y
    R = Y - X @ beta
    R -= R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    R /= sd
    return R.astype(np.float32), quality


def process_fmri_run(events_path: Path, selected_root: Path, out_root: Path, settings: Settings,
                     logger: Logger) -> tuple[Optional[RunRecord], Optional[pd.DataFrame]]:
    import nibabel as nib

    ent = parse_bids_entities(events_path)
    df = read_events(events_path)
    meta = sanitize_metadata(df, ent, "fmri", events_path)
    meta["trial_uid"] = make_trial_uids(meta)
    bold, quality = find_fmri_bold(events_path, selected_root, ent)
    if bold is None:
        logger.log(f"  fMRI {events_path.name}: BOLD nenalezen, přeskakuji.")
        return RunRecord("fmri", ent["subject"], ent["session"], ent["task"], ent["run"],
                         str(events_path), "", "MISSING", n_trials_input=len(df), notes="BOLD not found"), None

    logger.log(f"  fMRI sub-{ent['subject']} task-{ent['task']} run-{ent['run']}: {quality}")
    img = nib.load(str(bold))
    zooms = img.header.get_zooms()
    if len(zooms) < 4 or zooms[3] <= 0:
        raise UserFacingError(f"Nelze určit TR z {bold}")
    tr = float(zooms[3])
    mask_path = matching_fmri_mask(bold)
    mask_img = nib.load(str(mask_path)) if mask_path else None
    Y, coords, labels = extract_parcel_timeseries(
        img, mask_img,
        grid=(settings.fmri_grid_x, settings.fmri_grid_y, settings.fmri_grid_z),
        logger=logger,
    )
    Y, resid_quality = residualize_fmri(Y, matching_confounds(bold), logger)

    A, graph_source = graph_from_coords(coords, settings.knn)
    L, evals, U = gft_basis(A, settings.k_modes)

    t0, t1 = DEFAULT_FMRI_HRF_WINDOW
    trial_vecs = []
    keep_rows = []
    for ridx, row in df.iterrows():
        onset = float(row["onset"]) if pd.notna(row["onset"]) else np.nan
        if not np.isfinite(onset):
            continue
        v0 = max(0, int(math.floor((onset + t0) / tr)))
        v1 = min(len(Y), int(math.ceil((onset + t1) / tr)))
        if v1 <= v0:
            continue
        vec = np.nanmean(Y[v0:v1], axis=0)
        if not np.all(np.isfinite(vec)):
            vec = np.nan_to_num(vec)
        trial_vecs.append(vec.astype(np.float32))
        keep_rows.append(ridx)

    if not trial_vecs:
        logger.log("    Žádné použitelné fMRI trialy po časovém mapování.")
        return RunRecord("fmri", ent["subject"], ent["session"], ent["task"], ent["run"],
                         str(events_path), str(bold), quality, n_trials_input=len(df), notes="No valid trial windows"), None

    Xtrial = np.vstack(trial_vecs)
    modes = (Xtrial @ U).astype(np.float32)
    meta_out = meta.loc[keep_rows].reset_index(drop=True)
    meta_out["trial_uid"] = make_trial_uids(meta_out)
    meta_out["derivative_quality"] = quality + "+" + resid_quality
    meta_out["fmri_response_window_s"] = f"{t0}-{t1}"
    meta_out["TR_s"] = tr

    stem = f"sub-{ent['subject']}_ses-{ent['session'] or 'NA'}_task-{ent['task'] or 'NA'}_run-{ent['run'] or 'NA'}"
    fmri_dir = out_root / "fmri"
    graph_dir = out_root / "graphs"
    fmri_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fmri_dir / f"{stem}_sgit_fmri.npz",
        trial_uid=np.asarray(meta_out["trial_uid"].astype(str), dtype="U"),
        parcel_response=Xtrial.astype(np.float32),
        gft_modes=modes,
        coords_mm=coords.astype(np.float32),
        k_modes=np.int32(U.shape[1]),
        tr_s=np.float32(tr),
        hrf_window_s=np.asarray([t0, t1], dtype=np.float32),
    )
    np.savez_compressed(
        graph_dir / f"{stem}_fmri_graph.npz",
        adjacency=A.astype(np.float32),
        laplacian=L.astype(np.float32),
        eigenvalues=evals.astype(np.float32),
        eigenvectors=U.astype(np.float32),
        coords=coords.astype(np.float32),
        graph_source=np.asarray(graph_source),
    )

    rec = RunRecord(
        modality="fmri", subject=ent["subject"], session=ent["session"], task=ent["task"], run=ent["run"],
        events_file=str(events_path), source_file=str(bold), quality=quality + "+" + resid_quality,
        n_trials_input=len(df), n_trials_output=len(meta_out), n_nodes=Xtrial.shape[1], n_modes=modes.shape[1],
        graph_source=graph_source,
        notes="Coarse grid parcellation; trial response = mean standardized BOLD in 4-8 s post-onset window.",
    )
    return rec, meta_out


# ---------------------------------------------------------------------------
# MEEG / iEEG
# ---------------------------------------------------------------------------

def get_bids_root_for_event(events_path: Path, selected_root: Path) -> Path:
    # Walk upward until dataset_description.json or selected root.
    p = events_path.parent
    selected_root = selected_root.resolve()
    while True:
        if (p / "dataset_description.json").exists():
            return p
        if p == selected_root or p.parent == p:
            break
        p = p.parent
    # The event may sit in a nested BIDS root below user selection.
    for parent in events_path.parents:
        if (parent / "dataset_description.json").exists():
            return parent
    return selected_root


def load_raw_for_events(events_path: Path, selected_root: Path, modality: str, logger: Logger):
    try:
        from mne_bids import get_bids_path_from_fname, read_raw_bids, find_matching_paths
    except Exception as e:
        raise UserFacingError("Chybí mne-bids. Spusťte program přes 1_SPUSTIT_SGIT.bat.") from e

    bids_root = get_bids_root_for_event(events_path, selected_root)
    ent = parse_bids_entities(events_path)
    datatype = "ieeg" if modality == "ieeg" else events_path.parent.name.lower()
    if datatype not in ("meg", "eeg", "ieeg"):
        datatype = "meg" if modality == "meeg" else "ieeg"

    # Preferred path: find matching raw recording, because an events.tsv itself
    # is not a raw-data BIDSPath accepted by read_raw_bids.
    kwargs = {
        "root": bids_root,
        "subjects": ent["subject"] or None,
        "sessions": ent["session"] or None,
        "tasks": ent["task"] or None,
        "runs": ent["run"] or None,
        "datatypes": datatype,
        "suffixes": datatype,
        "check": False,
        "ignore_json": True,
    }

    # IMPORTANT for BrainVision BIDS: one recording consists of .vhdr + .vmrk + .eeg.
    # The binary .eeg payload must NOT be passed directly to mne_bids.read_raw_bids;
    # MNE-BIDS expects the .vhdr header.  v0.8 could accidentally choose .eeg first
    # because it sorted all non-TSV/JSON matches lexicographically.
    allowed_exts = {
        "meg": {".con", ".sqd", ".fif", ".pdf", ".ds"},
        "eeg": {".vhdr", ".edf", ".bdf", ".set"},
        "ieeg": {".vhdr", ".edf", ".set", ".mefd", ".nwb"},
    }
    preference = {
        "meg": [".fif", ".ds", ".con", ".sqd", ".pdf"],
        "eeg": [".vhdr", ".edf", ".bdf", ".set"],
        "ieeg": [".vhdr", ".edf", ".set", ".nwb", ".mefd"],
    }

    def supported_raw_candidates(paths):
        out = []
        for cand in paths:
            ext = (cand.extension or "").lower()
            if ext in allowed_exts.get(datatype, set()):
                out.append(cand)
        return out

    matches = find_matching_paths(**kwargs)
    raw_matches = supported_raw_candidates(matches)
    if not raw_matches:
        # Try less restrictive search; run entity is sometimes absent.
        kwargs["runs"] = None
        raw_matches = supported_raw_candidates(find_matching_paths(**kwargs))
    if not raw_matches:
        # Diagnostic: report extensions that existed, useful for unusual acquisition formats.
        seen = sorted({(bp.extension or "<none>").lower() for bp in matches})
        raise UserFacingError(
            f"MNE-BIDS nenašlo podporovaný raw záznam pro {events_path}. "
            f"Nalezené přípony: {', '.join(seen) or 'žádné'}; očekávám pro {datatype}: "
            f"{', '.join(sorted(allowed_exts.get(datatype, set())))}"
        )
    if len(raw_matches) > 1:
        # Prefer exact run match.
        exact = [bp for bp in raw_matches if (ent["run"] and str(bp.run or "") == ent["run"])]
        if exact:
            raw_matches = exact

    pref = {ext: i for i, ext in enumerate(preference.get(datatype, []))}
    bp = sorted(
        raw_matches,
        key=lambda x: (pref.get((x.extension or "").lower(), 999), natural_key(str(x.fpath)))
    )[0]
    if (bp.extension or "").lower() == ".vhdr":
        logger.log("    BrainVision záznam: používám správně .vhdr hlavičku; .eeg/.vmrk zůstávají jako doprovodné soubory.")
    logger.log(f"    Čtu BIDS raw: {bp.fpath}")
    raw = read_raw_bids(bp, extra_params={"preload": False}, on_ch_mismatch="reorder", verbose="ERROR")
    return raw, bp


def choose_ephys_picks(raw, modality: str) -> tuple[np.ndarray, str]:
    import mne
    if modality == "ieeg":
        picks = mne.pick_types(raw.info, ecog=True, seeg=True, dbs=True, exclude="bads")
        if len(picks) < 3:
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        return np.asarray(picks, int), "ieeg_ecog_seeg"

    # For multisite MEG, one homogeneous sensor class is preferable to mixing
    # sensor units. Prefer magnetometers, then gradiometers, then EEG.
    mag = mne.pick_types(raw.info, meg="mag", eeg=False, exclude="bads")
    grad = mne.pick_types(raw.info, meg="grad", eeg=False, exclude="bads")
    eeg = mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")
    if len(mag) >= 20:
        return np.asarray(mag, int), "meg_magnetometers"
    if len(grad) >= 20:
        return np.asarray(grad, int), "meg_gradiometers"
    both = mne.pick_types(raw.info, meg=True, eeg=False, exclude="bads")
    if len(both) >= 20:
        return np.asarray(both, int), "meg_all"
    if len(eeg) >= 8:
        return np.asarray(eeg, int), "eeg"
    if len(both) >= 3:
        return np.asarray(both, int), "meg_all_small"
    return np.asarray([], int), "none"


def estimate_run_channel_scale(raw, picks: np.ndarray, target_samples: int = 8000) -> np.ndarray:
    """Robust per-channel scale estimated once per run (label-free).

    The scale is constant across trials, so trial-to-trial energy differences are
    preserved. This avoids mixing heterogeneous physical sensor units while not
    normalizing every trial to the same amplitude.
    """
    n_times = int(raw.n_times)
    if n_times <= 0:
        return np.ones(len(picks), dtype=float)
    n_blocks = min(20, max(1, n_times // max(1, int(raw.info["sfreq"]))))
    block_len = max(50, min(max(100, target_samples // n_blocks), n_times))
    starts = np.linspace(0, max(0, n_times - block_len - 1), n_blocks, dtype=int)
    blocks = []
    for st in starts:
        blocks.append(raw.get_data(picks=picks, start=int(st), stop=int(min(n_times, st + block_len))))
    X = np.concatenate(blocks, axis=1) if blocks else raw.get_data(picks=picks, start=0, stop=min(n_times, 1000))
    med = np.nanmedian(X, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(X - med), axis=1) / 0.67448975
    mad[~np.isfinite(mad) | (mad < 1e-15)] = 1.0
    return mad.astype(float)


def nearest_frequency_coeff(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return complex center-frequency coeff and band power per channel.

    data: channels x samples in one trial/time window.
    """
    n = data.shape[1]
    if n < 8:
        z = np.full(data.shape[0], np.nan + 1j * np.nan, dtype=np.complex64)
        p = np.full(data.shape[0], np.nan, dtype=np.float32)
        return z, p
    taper = signal.windows.hann(n, sym=False).astype(np.float64)
    X = np.fft.rfft(data * taper[None, :], axis=1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    f0 = 0.5 * (band[0] + band[1])
    center_idx = int(np.argmin(np.abs(freqs - f0)))
    mask = (freqs >= band[0]) & (freqs < band[1])
    if not np.any(mask):
        mask[center_idx] = True
    # Normalize amplitude approximately by taper energy / length for stable scale.
    norm = max(np.sqrt(np.sum(taper ** 2)), 1e-12)
    z = (X[:, center_idx] / norm).astype(np.complex64)
    power = (np.mean(np.abs(X[:, mask]) ** 2, axis=1) / (norm ** 2)).astype(np.float32)
    return z, power


def process_ephys_run(events_path: Path, selected_root: Path, out_root: Path, settings: Settings,
                      logger: Logger, modality: str) -> tuple[RunRecord, Optional[pd.DataFrame]]:
    import mne

    ent = parse_bids_entities(events_path)
    df = read_events(events_path)
    meta = sanitize_metadata(df, ent, modality, events_path)
    meta["trial_uid"] = make_trial_uids(meta)

    # Level-2b compact MEEG: COGITATE MEEG events.tsv contains a marker stream,
    # not only stimulus trials. Keep the original row indices for epoch extraction,
    # but enrich each true visual stimulus with category, identity, orientation and
    # task relevance reconstructed from the following marker rows.
    stimulus_mask = np.ones(len(df), dtype=bool)
    if modality == "meeg" and LEVEL2B_COMPACT_MEEG:
        stimulus_mask[:] = False
        meta["sgit_category"] = ""
        meta["sgit_identity"] = ""
        meta["sgit_orientation"] = ""
        meta["sgit_relevance"] = ""
        ttcol = df["trial_type"].astype(str) if "trial_type" in df.columns else pd.Series([""] * len(df))
        for q, tt in enumerate(ttcol):
            mt = re.match(r"^(face|object|letter|false)(\d+)$", str(tt))
            if not mt:
                continue
            orientation = ""
            relevance = ""
            for off in range(1, 6):
                if q + off >= len(df):
                    break
                nxt = str(ttcol.iloc[q + off]).strip().lower()
                if nxt in ("left", "right", "center") and not orientation:
                    orientation = nxt
                if nxt.startswith("task ") and not relevance:
                    relevance = nxt
            stimulus_mask[q] = True
            meta.loc[q, "sgit_category"] = mt.group(1)
            meta.loc[q, "sgit_identity"] = mt.group(0)
            meta.loc[q, "sgit_orientation"] = orientation
            meta.loc[q, "sgit_relevance"] = relevance
        logger.log(f"    Level-2b compact: z {len(df)} markerů zpracovávám jen {int(stimulus_mask.sum())} skutečných vizuálních stimulů.")

    raw, bp = load_raw_for_events(events_path, selected_root, modality, logger)
    picks, pick_kind = choose_ephys_picks(raw, modality)
    if len(picks) < 3:
        raise UserFacingError(f"Příliš málo použitelných kanálů pro {events_path}")
    logger.log(f"    Používám {len(picks)} kanálů ({pick_kind}), sfreq={raw.info['sfreq']:.3f} Hz")

    A, graph_source, coords = graph_from_positions_or_functional(raw, picks, settings.knn, logger)
    L, evals, U = gft_basis(A, settings.k_modes)
    ch_names = [raw.ch_names[int(i)] for i in picks]
    sfreq = float(raw.info["sfreq"])
    run_scale = estimate_run_channel_scale(raw, picks)
    logger.log("    Odhaduji konstantní robustní škálu kanálů z celého runu (bez znalosti labelů).")
    bands = dict(DEFAULT_BANDS)
    if modality == "ieeg" and sfreq >= 260:
        bands.update(DEFAULT_IEEG_EXTRA_BANDS)
    band_names = list(bands.keys())
    windows = DEFAULT_TF_WINDOWS

    ntr = len(df)
    Zmodes = np.full((ntr, len(band_names), len(windows), U.shape[1]),
                     np.nan + 1j * np.nan, dtype=np.complex64)
    Pmodes = np.full((ntr, len(band_names), len(windows), U.shape[1]), np.nan, dtype=np.float32)
    # Optional node power is useful for validating energy-only baselines but small.
    Pnodes = np.full((ntr, len(band_names), len(windows), len(picks)), np.nan, dtype=np.float32)
    valid_trial = np.zeros(ntr, dtype=bool)

    ep0, ep1 = DEFAULT_EPHYS_EPOCH
    first_samp = int(raw.first_samp)
    duration_s = raw.n_times / sfreq
    logger.log(f"    Počítám komplexní trialové koeficienty: {ntr} trialů x {len(band_names)} pásem x {len(windows)} oken")
    processed_counter = 0
    target_counter = int(stimulus_mask.sum()) if (modality == "meeg" and LEVEL2B_COMPACT_MEEG) else ntr
    for ridx, row in df.iterrows():
        if modality == "meeg" and LEVEL2B_COMPACT_MEEG and not stimulus_mask[ridx]:
            continue
        processed_counter += 1
        if processed_counter % 25 == 1 or processed_counter == target_counter:
            logger.log(f"      trial {processed_counter}/{target_counter}")
        onset = float(row["onset"]) if pd.notna(row["onset"]) else np.nan
        if not np.isfinite(onset):
            continue
        start_s = onset + ep0
        stop_s = onset + ep1
        if stop_s <= 0 or start_s >= duration_s:
            continue
        s0 = max(0, int(round(start_s * sfreq)))
        s1 = min(raw.n_times, int(round(stop_s * sfreq)))
        if s1 - s0 < 16:
            continue
        data = raw.get_data(picks=picks, start=s0, stop=s1).astype(np.float64, copy=False)
        # Robust per-channel baseline correction from [-0.2,0) if present.
        b0 = 0
        b1 = min(data.shape[1], max(1, int(round((-ep0) * sfreq))))
        baseline = np.nanmean(data[:, b0:b1], axis=1, keepdims=True) if b1 > b0 else np.nanmean(data, axis=1, keepdims=True)
        data = np.nan_to_num(data - baseline, nan=0.0, posinf=0.0, neginf=0.0)
        # Scale by one run-level robust factor per channel. This preserves
        # trial-to-trial energy differences needed for energy-only baselines.
        data = data / run_scale[:, None]

        # Relative time of each sample to event onset.
        ts = ep0 + np.arange(data.shape[1]) / sfreq
        for wi, (w0, w1) in enumerate(windows):
            m = (ts >= w0) & (ts < w1)
            if m.sum() < 8:
                continue
            seg = data[:, m]
            # linear detrend in the local window
            seg = signal.detrend(seg, axis=1, type="linear")
            for bi, bname in enumerate(band_names):
                z_nodes, p_nodes = nearest_frequency_coeff(seg, sfreq, bands[bname])
                z_modes = U.T @ z_nodes.astype(np.complex128)
                # Since U is orthonormal, |z_mode|^2 is the modal center-frequency energy.
                Zmodes[ridx, bi, wi] = z_modes.astype(np.complex64)
                Pmodes[ridx, bi, wi] = (np.abs(z_modes) ** 2).astype(np.float32)
                Pnodes[ridx, bi, wi] = p_nodes
        valid_trial[ridx] = np.any(np.isfinite(Pmodes[ridx]))

    meta_out = meta.loc[valid_trial].reset_index(drop=True)
    meta_out["trial_uid"] = make_trial_uids(meta_out)
    meta_out["derivative_quality"] = "lightweight_sensor_or_electrode_level"
    meta_out["channel_selection"] = pick_kind
    Zmodes = Zmodes[valid_trial]
    Pmodes = Pmodes[valid_trial]
    Pnodes = Pnodes[valid_trial]

    stem = f"sub-{ent['subject']}_ses-{ent['session'] or 'NA'}_task-{ent['task'] or 'NA'}_run-{ent['run'] or 'NA'}"
    mod_dir = out_root / modality
    graph_dir = out_root / "graphs"
    mod_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    if modality == "meeg" and LEVEL2B_COMPACT_MEEG:
        # For the 20–30 subject Level-2b-full run, keep only the information
        # needed to recompute energy, Bphase, Bcov and gauge-invariant descriptors.
        # modal_power is exactly derivable from complex_gft_modes and node_band_power
        # dominates file size, therefore neither is duplicated in the upload ZIP.
        np.savez_compressed(
            mod_dir / f"{stem}_sgit_{modality}.npz",
            trial_uid=np.asarray(meta_out["trial_uid"].astype(str), dtype="U"),
            complex_gft_modes=Zmodes,
            band_names=np.asarray(band_names, dtype="U"),
            band_edges_hz=np.asarray([bands[b] for b in band_names], dtype=np.float32),
            time_windows_s=np.asarray(windows, dtype=np.float32),
            channel_names=np.asarray(ch_names, dtype="U"),
            channel_coords=coords.astype(np.float32),
            sfreq_hz=np.float32(sfreq),
            epoch_s=np.asarray([ep0, ep1], dtype=np.float32),
            level2b_compact=np.asarray(True),
        )
    else:
        np.savez_compressed(
            mod_dir / f"{stem}_sgit_{modality}.npz",
            trial_uid=np.asarray(meta_out["trial_uid"].astype(str), dtype="U"),
            complex_gft_modes=Zmodes,
            modal_power=Pmodes,
            node_band_power=Pnodes,
            band_names=np.asarray(band_names, dtype="U"),
            band_edges_hz=np.asarray([bands[b] for b in band_names], dtype=np.float32),
            time_windows_s=np.asarray(windows, dtype=np.float32),
            channel_names=np.asarray(ch_names, dtype="U"),
            channel_coords=coords.astype(np.float32),
            sfreq_hz=np.float32(sfreq),
            epoch_s=np.asarray([ep0, ep1], dtype=np.float32),
        )
    np.savez_compressed(
        graph_dir / f"{stem}_{modality}_graph.npz",
        adjacency=A.astype(np.float32),
        laplacian=L.astype(np.float32),
        eigenvalues=evals.astype(np.float32),
        eigenvectors=U.astype(np.float32),
        coords=coords.astype(np.float32),
        channel_names=np.asarray(ch_names, dtype="U"),
        graph_source=np.asarray(graph_source),
        channel_selection=np.asarray(pick_kind),
    )
    rec = RunRecord(
        modality=modality, subject=ent["subject"], session=ent["session"], task=ent["task"], run=ent["run"],
        events_file=str(events_path), source_file=str(bp.fpath), quality="lightweight_sensor_or_electrode_level",
        n_trials_input=len(df), n_trials_output=len(meta_out), n_nodes=len(picks), n_modes=U.shape[1],
        graph_source=graph_source,
        notes="Complex FFT center-frequency coefficients per trial/band/window; constant run-level robust channel scaling; no trial averaging.",
    )
    try:
        raw.close()
    except Exception:
        pass
    return rec, meta_out


def privacy_safe_record_dict(r: RunRecord, selected_root: Path) -> dict:
    d = asdict(r)
    for key in ("events_file", "source_file"):
        val = d.get(key, "")
        if not val:
            continue
        try:
            d[key] = str(Path(val).resolve().relative_to(selected_root.resolve()))
        except Exception:
            d[key] = Path(val).name
    return d


# ---------------------------------------------------------------------------
# Archive / manifest
# ---------------------------------------------------------------------------

def copy_codebook_files(selected_root: Path, out_root: Path, logger: Logger):
    meta_dir = out_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    # Small experiment descriptors only. No participants.tsv demographics.
    patterns = ["**/task-*.json", "**/*_events.json", "**/dataset_description.json"]
    copied = 0
    seen_names = set()
    for pat in patterns:
        for p in selected_root.glob(pat):
            if "derivatives" in [x.lower() for x in p.parts] and p.name == "dataset_description.json":
                continue
            if p.stat().st_size > 2_000_000:
                continue
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.name)
            if safe in seen_names:
                continue
            try:
                shutil.copy2(p, meta_dir / safe)
                seen_names.add(safe)
                copied += 1
            except Exception:
                pass
    logger.log(f"Zkopírováno {copied} malých BIDS codebook/JSON souborů (bez raw dat).")


def write_format_readme(out_root: Path):
    text = f"""SGIT compact derivative format v{APP_VERSION}
========================================

Tento archiv obsahuje ODVOZENE trial-level charakteristiky. Neobsahuje raw fMRI,
MEG/EEG ani iEEG signály.

fmri/*_sgit_fmri.npz
--------------------
trial_uid          Unicode ID trialů, odpovídá řádkům metadata/fmri_trials.csv
parcel_response    [trial, parcel] hrubý parcelovaný standardizovaný BOLD response
gft_modes          [trial, K] reálné GFT koeficienty
coords_mm          [parcel, 3] střed parcel

meeg/*_sgit_meeg.npz (Level-2b compact)
--------------------------------------
trial_uid          pouze skutečné vizuální stimulus trialy
complex_gft_modes  [trial, band, window, K] komplexní GFT koeficienty
band_names         theta/alpha/beta/gamma
time_windows_s     časová okna relativně ke stimulus onset
channel_names      použité MEEG kanály
channel_coords     jejich souřadnice
level2b_compact    True
Pozn.: modal_power se přesně dopočítá jako |complex_gft_modes|^2 a proto se neduplikuje.
Objemný node_band_power se v Level-2b MEEG výstupu neukládá.

ieeg/*_sgit_ieeg.npz (pokud uživatel iEEG ručně zapne)
------------------------------------------------------
trial_uid, complex_gft_modes, modal_power, node_band_power, band_names,
time_windows_s, channel_names, channel_coords

graphs/*_graph.npz
------------------
adjacency, laplacian, eigenvalues, eigenvectors, coords, graph_source

metadata/*_trials.csv
---------------------
Původní event sloupce + modality/subject/session/task/run + trial_uid.
Pro Level-2b MEEG jsou stimulus řádky navíc obohaceny o sgit_category,
sgit_identity, sgit_orientation a sgit_relevance. Demografické participants.tsv
se záměrně nepřenáší.

DŮLEŽITÉ PRO ANALÝZU
--------------------
- Bivektor B_ij = beta_ij e_i wedge e_j se nemá odhadovat z jednoho trialu;
  beta se určuje stabilně z rozdělení trialů / času / subjektů.
- Trivektor T_ijk = tau_ijk e_i wedge e_j wedge e_k se má potvrdit PID/PhiID
  nebo jinou vyšší synergickou mírou. O-information je vhodná jako screening.
- GFT báze různých subjektů/modalit se nesmí bez dalšího porovnávat po indexech;
  pro cross-modal validaci se používají RDM/projektory/alignment.
- fMRI UNPREPROCESSED_APPROX není finální publikační preprocessing.
"""
    (out_root / "FORMAT_FOR_CHATGPT.txt").write_text(text, encoding="utf-8")


def make_zip(src_dir: Path, zip_path: Path, logger: Logger):
    logger.log(f"Balím výstup do {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))
    logger.log(f"Hotovo. Velikost ZIP: {zip_path.stat().st_size / 1024**2:.1f} MB")


def _run_conversion_folder(settings: Settings, callback: Optional[Callable[[str], None]] = None) -> Path:
    selected_root = Path(settings.bids_root).expanduser().resolve()
    output_dir = Path(settings.output_dir).expanduser().resolve()
    if not selected_root.exists():
        raise UserFacingError(f"Vstupní složka neexistuje: {selected_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / "SGIT_compact_dataset"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    logger = Logger(output_dir / "SGIT_LOG.txt", callback)
    logger.log("=" * 70)
    logger.log(f"SGIT COGITATE Converter v{APP_VERSION}")
    logger.log(f"Vstup:  {selected_root}")
    logger.log(f"Výstup: {output_dir}")
    logger.log("Program pouze čte původní data; raw neurodata se do výstupu nekopírují.")
    logger.log("=" * 70)

    modalities = set()
    if settings.do_fmri:
        modalities.add("fmri")
    if settings.do_meeg:
        modalities.add("meeg")
    if settings.do_ieeg:
        modalities.add("ieeg")
    if not modalities:
        raise UserFacingError("Není vybrána žádná modalita.")

    events = scan_events(selected_root, logger)
    if not events:
        zips = list(selected_root.glob("**/*.zip"))
        if zips:
            raise UserFacingError(
                "Nenašel jsem *_events.tsv, ale vidím ZIP soubory. Data je nutné nejprve rozbalit; "
                "pak zvolte složku obsahující BIDS sub-* adresáře."
            )
        raise UserFacingError("Nenašel jsem žádná BIDS *_events.tsv data.")

    selected = choose_subjects(events, settings.max_subjects, modalities)
    for mod in sorted(modalities):
        logger.log(f"Vybrané subjekty {mod}: {', '.join(sorted(selected.get(mod, []), key=natural_key)) or 'žádné'}")

    records: list[RunRecord] = []
    meta_by_mod: dict[str, list[pd.DataFrame]] = {m: [] for m in modalities}

    for ev, mod in events:
        if mod not in modalities:
            continue
        ent = parse_bids_entities(ev)
        if ent["subject"] not in selected.get(mod, set()):
            continue
        try:
            if mod == "fmri":
                rec, meta = process_fmri_run(ev, selected_root, work, settings, logger)
            else:
                rec, meta = process_ephys_run(ev, selected_root, work, settings, logger, mod)
            if rec:
                records.append(rec)
            if meta is not None and len(meta):
                meta_by_mod[mod].append(meta)
        except Exception as e:
            logger.log(f"  CHYBA při {mod} {ev}: {e}")
            logger.log(traceback.format_exc())
            records.append(RunRecord(
                modality=mod, subject=ent["subject"], session=ent["session"], task=ent["task"], run=ent["run"],
                events_file=str(ev), source_file="", quality="ERROR", n_trials_input=0, n_trials_output=0,
                notes=f"{type(e).__name__}: {e}",
            ))

    meta_dir = work / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for mod, frames in meta_by_mod.items():
        if frames:
            combined = pd.concat(frames, ignore_index=True, sort=False)
            combined.to_csv(meta_dir / f"{mod}_trials.csv", index=False, encoding="utf-8")
            logger.log(f"Metadata {mod}: {len(combined)} trialů")

    safe_records = [privacy_safe_record_dict(r, selected_root) for r in records]
    pd.DataFrame(safe_records).to_csv(work / "run_inventory.csv", index=False, encoding="utf-8")
    copy_codebook_files(selected_root, work, logger)
    write_format_readme(work)

    manifest = {
        "format": "SGIT_COGITATE_COMPACT",
        "format_version": APP_VERSION,
        "settings": {
            "input_root_name": selected_root.name,
            "max_subjects": settings.max_subjects,
            "k_modes": settings.k_modes,
            "do_fmri": settings.do_fmri,
            "do_meeg": settings.do_meeg,
            "do_ieeg": settings.do_ieeg,
            "fmri_grid": [settings.fmri_grid_x, settings.fmri_grid_y, settings.fmri_grid_z],
            "knn": settings.knn,
            "source_archives": list(settings.source_archives),
        },
        "notes": [
            "Raw neurodata are intentionally not included.",
            "Trial-level derivatives are retained; do not interpret individual coefficients as stable GA blades without resampling/statistical estimation.",
            "fMRI prefers fMRIPrep derivatives; raw BOLD fallback is explicitly flagged as UNPREPROCESSED_APPROX.",
            "MEEG/iEEG processing is lightweight sensor/electrode-level preprocessing for SGIT exploratory Level-1 analysis.",
            "Cross-subject/cross-modal GFT mode indices require alignment or RDM/projector comparison.",
        ],
        "run_records": safe_records,
        "summary": {
            "runs_total": len(records),
            "runs_ok": sum(r.quality != "ERROR" and r.quality != "MISSING" for r in records),
            "runs_error": sum(r.quality == "ERROR" for r in records),
            "trials_output": int(sum(r.n_trials_output for r in records)),
            "modalities_present": sorted({r.modality for r in records if r.n_trials_output > 0}),
        },
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "SGIT COGITATE COMPACT DATASET - PROTOKOL",
        "=" * 55,
        f"Verze převodníku: {APP_VERSION}",
        f"Vstupní dataset: {selected_root.name}",
        f"Počet záznamů/runů: {len(records)}",
        f"Počet odvozených trialů: {sum(r.n_trials_output for r in records)}",
        "",
        "RUNY:",
    ]
    for r in records:
        report_lines.append(
            f"- {r.modality} sub-{r.subject} task-{r.task} run-{r.run}: "
            f"{r.quality}; trials {r.n_trials_output}/{r.n_trials_input}; "
            f"nodes={r.n_nodes}; K={r.n_modes}; graph={r.graph_source}; {r.notes}"
        )
    report_lines += [
        "",
        "Co mi máte nahrát do ChatGPT:",
        "  SGIT_upload.zip",
        "Pokud konverze některé modality selhala, přiložte také SGIT_LOG.txt.",
    ]
    (work / "processing_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    zip_path = output_dir / "SGIT_upload.zip"
    if zip_path.exists():
        zip_path.unlink()
    make_zip(work, zip_path, logger)
    logger.log("=" * 70)
    logger.log("KONVERZE DOKONČENA")
    logger.log(f"NAHRAJTE MI TENTO SOUBOR: {zip_path}")
    logger.log("=" * 70)
    return zip_path


def run_conversion(settings: Settings, callback: Optional[Callable[[str], None]] = None) -> Path:
    """Run conversion from either an unpacked BIDS folder or one/more huge ZIP archives.

    ZIP mode never unpacks the full archive. It reads the ZIP central directory,
    selects the requested subjects, streams only their files to a temporary
    directory, runs the ordinary converter, and removes the temporary files.
    """
    if not settings.zip_files:
        return _run_conversion_folder(settings, callback)

    output_dir = Path(settings.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ziplog = Logger(output_dir / "SGIT_ZIP_LOG.txt", callback)
    ziplog.log("=" * 70)
    ziplog.log(f"SGIT COGITATE Converter ZIP mode v{APP_VERSION}")
    ziplog.log("Původní ZIP soubory se NEMĚNÍ a NEJSOU celé rozbalovány.")
    for z in settings.zip_files:
        zp = Path(z).expanduser().resolve()
        ziplog.log(f"ZIP: {zp} ({zp.stat().st_size/1024**3:.1f} GB)" if zp.exists() else f"ZIP: {zp}")

    modalities = set()
    if settings.do_fmri:
        modalities.add("fmri")
    if settings.do_meeg:
        modalities.add("meeg")
    if settings.do_ieeg:
        modalities.add("ieeg")
    if not modalities:
        raise UserFacingError("Není vybrána žádná modalita.")

    # IMPORTANT: never use the user-selected output directory for temporary
    # extraction. It may be a very long OneDrive path. Keep all heavy/internal
    # work under a short LOCALAPPDATA path and copy only the final small files.
    local_base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "SGIT_Cogitate_v09"
    temp_root = local_base / "zip"
    stage_output = local_base / "stage"
    try:
        for p in (temp_root, stage_output):
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)
        ziplog.log(f"Krátká lokální pracovní cesta: {local_base}")
        extracted_root, selected, names = extract_selected_zip_subset(
            settings.zip_files, temp_root, modalities, settings.max_subjects, ziplog
        )
        ziplog.log(f"Dočasně rozbaleno vybraných souborů: {len(names)}")
        folder_settings = replace(
            settings,
            bids_root=str(extracted_root),
            output_dir=str(stage_output),
            zip_files=(),
            source_archives=tuple(Path(z).name for z in settings.zip_files),
        )
        staged_result = _run_conversion_folder(folder_settings, callback)

        # Copy only compact outputs back to the user's chosen directory.
        final_result = output_dir / "SGIT_upload.zip"
        shutil.copy2(staged_result, final_result)
        for logname in ("SGIT_LOG.txt",):
            src = stage_output / logname
            if src.exists():
                shutil.copy2(src, output_dir / logname)
        ziplog.log(f"Výsledný soubor: {final_result}")
        return final_result
    finally:
        if local_base.exists():
            ziplog.log("Mažu krátká lokální dočasná data...")
            try:
                shutil.rmtree(local_base)
                ziplog.log("Dočasná data smazána. Na disku zůstávají jen původní ZIPy a SGIT výstup.")
            except Exception as e:
                ziplog.log(f"VAROVÁNÍ: dočasnou složku se nepodařilo úplně smazat: {e}")



# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> int:
    print(f"SGIT converter self-test v{APP_VERSION}")
    # Graph/GFT exact reconstruction on retained full basis.
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    A, src = graph_from_coords(coords, knn=2)
    L, lam, U = gft_basis(A, 4)
    x = np.array([1.0, 2.0, -1.0, 0.5])
    xr = U @ (U.T @ x)
    assert np.allclose(x, xr, atol=1e-8), "GFT reconstruction failed"
    assert np.allclose(L, L.T), "Laplacian not symmetric"

    # Complex phase relation changes sign under order reversal when encoded via imag cross-product.
    z1 = np.array([1 + 0j, 1j, -1 + 0j])
    z2 = np.array([1j, 1 + 0j, -1j])
    b12 = np.imag(z1 * np.conj(z2)).mean()
    b21 = np.imag(z2 * np.conj(z1)).mean()
    assert np.allclose(b12, -b21), "Phase bivector antisymmetry failed"

    # FFT helper shape.
    rng = np.random.default_rng(1)
    data = rng.normal(size=(5, 250))
    z, p = nearest_frequency_coeff(data, 250.0, (8, 13))
    assert z.shape == (5,) and p.shape == (5,)
    assert np.all(np.isfinite(p))

    print("OK: 3/3 základních testů prošlo.")
    return 0


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import threading

    root = tk.Tk()
    root.title("SGIT COGITATE Converter v1.2 – Level-2b-full MEEG")
    root.geometry("930x720")

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="SGIT – COGITATE Level‑2b-full: gauge-kovariantní populační MEEG běh",
              font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 10))

    data_var = tk.StringVar()
    mode_var = tk.StringVar(value="Nebyl vybrán vstup")
    out_var = tk.StringVar(value=str(Path.home() / "SGIT_OUTPUT"))
    max_var = tk.StringVar(value=str(DEFAULT_MAX_SUBJECTS))
    k_var = tk.StringVar(value=str(DEFAULT_K))
    fmri_var = tk.BooleanVar(value=False)
    meeg_var = tk.BooleanVar(value=True)
    ieeg_var = tk.BooleanVar(value=False)
    selected_zips: list[str] = []

    def choose_zips():
        paths = filedialog.askopenfilenames(
            title="Vyberte jeden nebo více COGITATE ZIP souborů",
            filetypes=[("ZIP archivy", "*.zip"), ("Všechny soubory", "*.*")],
        )
        if paths:
            selected_zips.clear()
            selected_zips.extend(paths)
            data_var.set(" ; ".join(paths))
            mode_var.set(f"ZIP režim: {len(paths)} archiv(y) – úplné rozbalení není potřeba")

    def choose_data_folder():
        pth = filedialog.askdirectory(title="Vyberte složku s již rozbalenými COGITATE BIDS daty")
        if pth:
            selected_zips.clear()
            data_var.set(pth)
            mode_var.set("Složkový režim: data jsou již rozbalena")

    def choose_out():
        pth = filedialog.askdirectory(title="Vyberte výstupní složku (na tomto disku vznikne i dočasný výběr subjektů)")
        if pth:
            out_var.set(pth)

    ttk.Label(frm, text="1. Vstupní data COGITATE:").grid(row=1, column=0, sticky="w")
    ttk.Entry(frm, textvariable=data_var, width=76).grid(row=2, column=0, columnspan=3, sticky="ew", padx=(0, 6))
    ttk.Button(frm, text="Vybrat ZIP soubory", command=choose_zips).grid(row=2, column=3, sticky="ew", padx=3)
    ttk.Button(frm, text="Nebo vybrat složku", command=choose_data_folder).grid(row=2, column=4, sticky="ew", padx=3)
    ttk.Label(frm, textvariable=mode_var).grid(row=3, column=0, columnspan=5, sticky="w", pady=(3, 0))

    ttk.Label(frm, text="2. Kam uložit malý výstup:").grid(row=4, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(frm, textvariable=out_var, width=76).grid(row=5, column=0, columnspan=4, sticky="ew", padx=(0, 6))
    ttk.Button(frm, text="Vybrat výstup", command=choose_out).grid(row=5, column=4, sticky="ew")

    opts = ttk.LabelFrame(frm, text="3. Nastavení", padding=10)
    opts.grid(row=6, column=0, columnspan=5, sticky="ew", pady=12)
    ttk.Checkbutton(opts, text="fMRI", variable=fmri_var).grid(row=0, column=0, padx=8, sticky="w")
    ttk.Checkbutton(opts, text="MEEG", variable=meeg_var).grid(row=0, column=1, padx=8, sticky="w")
    ttk.Checkbutton(opts, text="iEEG", variable=ieeg_var).grid(row=0, column=2, padx=8, sticky="w")
    ttk.Label(opts, text="Max. subjektů na modalitu:").grid(row=1, column=0, sticky="e", pady=6)
    ttk.Entry(opts, textvariable=max_var, width=8).grid(row=1, column=1, sticky="w")
    ttk.Label(opts, text="GFT módy K:").grid(row=1, column=2, sticky="e")
    ttk.Entry(opts, textvariable=k_var, width=8).grid(row=1, column=3, sticky="w")

    note = (
        "LEVEL-2b-full: primární preregistrovatelný běh je MEEG, 30 subjektů a K=16; fMRI a iEEG proto nejsou "
        "ve výchozím nastavení zaškrtnuty. Použijte FULL MEEG BIDS ZIP, ne SAMPLE archiv se 4 subjekty. Program ZIP celý "
        "nerozbaluje a nově ukládá jen skutečné vizuální stimulus trialy a komplexní GFT koeficienty potřebné pro E, Bphase, "
        "Bcov a kanonické gauge-invariantní deskriptory. Tím se výsledný upload zmenší přibližně o řád. K=16 neměňte."
    )
    ttk.Label(frm, text=note, wraplength=890).grid(row=7, column=0, columnspan=5, sticky="w", pady=(0, 8))

    text = tk.Text(frm, height=21, wrap="word")
    text.grid(row=9, column=0, columnspan=5, sticky="nsew")
    scroll = ttk.Scrollbar(frm, orient="vertical", command=text.yview)
    scroll.grid(row=9, column=5, sticky="ns")
    text.configure(yscrollcommand=scroll.set)

    frm.columnconfigure(0, weight=1)
    frm.rowconfigure(9, weight=1)

    def append_log(msg):
        def _do():
            text.insert("end", msg + "\n")
            text.see("end")
        root.after(0, _do)

    start_btn = ttk.Button(frm, text="SPUSTIT PŘEVOD", style="Accent.TButton")
    start_btn.grid(row=8, column=0, columnspan=5, sticky="ew", pady=(3, 10), ipady=7)

    def do_run():
        try:
            if not data_var.get():
                raise UserFacingError("Nejprve vyberte 1–3 ZIP soubory COGITATE nebo rozbalenou BIDS složku.")
            max_subjects = int(max_var.get())
            k_modes = int(k_var.get())
            if max_subjects < 0 or k_modes < 2 or k_modes > 64:
                raise UserFacingError("Nastavení musí být: max subjektů >= 0; K mezi 2 a 64.")
            s = Settings(
                bids_root="" if selected_zips else data_var.get(),
                output_dir=out_var.get(),
                max_subjects=max_subjects,
                k_modes=k_modes,
                do_fmri=fmri_var.get(),
                do_meeg=meeg_var.get(),
                do_ieeg=ieeg_var.get(),
                zip_files=tuple(selected_zips),
            )
            root.after(0, lambda: start_btn.configure(state="disabled"))
            zip_path = run_conversion(s, append_log)
            root.after(0, lambda: messagebox.showinfo(
                "Hotovo",
                f"Level-2b-full převod je dokončen.\n\nNahrajte mi do ChatGPT soubor:\n{zip_path}\n\n"
                f"Pokud byla některá chyba, přiložte také SGIT_LOG.txt a SGIT_ZIP_LOG.txt."
            ))
        except Exception as e:
            append_log("CHYBA: " + str(e))
            append_log(traceback.format_exc())
            root.after(0, lambda: messagebox.showerror(
                "Chyba",
                f"Převod se nedokončil:\n\n{e}\n\nPošlete mi SGIT_LOG.txt / SGIT_ZIP_LOG.txt nebo screenshot chyby."
            ))
        finally:
            root.after(0, lambda: start_btn.configure(state="normal"))

    def start():
        text.delete("1.0", "end")
        threading.Thread(target=do_run, daemon=True).start()

    start_btn.configure(command=start)
    root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="COGITATE BIDS/ZIP -> SGIT compact derivative converter")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", type=str, help="BIDS root or parent directory")
    parser.add_argument("--zip", dest="zip_files", nargs="*", help="One or more COGITATE ZIP archives")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--max-subjects", type=int, default=DEFAULT_MAX_SUBJECTS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.output and (args.input or args.zip_files):
        s = Settings(
            args.input or "", args.output, args.max_subjects, args.k,
            zip_files=tuple(args.zip_files or ()),
        )
        run_conversion(s)
        return 0
    launch_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
