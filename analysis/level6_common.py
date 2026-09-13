from __future__ import annotations
import hashlib, json, os, re, shutil, sys, time
from pathlib import Path
from typing import Any

SUB_RE = re.compile(r"^(?:sub-)?(.+)$")

def load_json(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_atomic(path: str | Path, obj: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')
    os.replace(tmp, p)

def sha256_file(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def canonical_subject(label: str) -> str:
    label = str(label).strip()
    if label.lower().startswith('sub-'):
        label = label[4:]
    return label

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

def resolve_relative(base_file: str | Path, value: str) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path(base_file).resolve().parent / p
    return p.resolve()

def ensure_free_space(path: Path, required_bytes: int, safety: float = 1.25) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    need = int(required_bytes * safety)
    if free < need:
        raise RuntimeError(
            f'Nedostatek místa: odhad potřeby {need/1024**3:.1f} GB, volno {free/1024**3:.1f} GB v {path}'
        )

def utc_now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')

def log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f'[{utc_now()}] {text}'
    print(line, flush=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(line + '\n')

def config_hash(config_path: Path, locked_path: Path) -> str:
    h = hashlib.sha256()
    for p in (config_path, locked_path):
        h.update(p.read_bytes())
        h.update(b'\0')
    return h.hexdigest()
