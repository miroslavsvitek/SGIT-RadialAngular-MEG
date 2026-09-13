from pathlib import Path
import csv, hashlib

ROOT = Path(__file__).resolve().parent
MAN = ROOT / 'MANIFEST_SHA256.csv'
BINARY_EXT = {'.pdf', '.png', '.npz'}

def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in BINARY_EXT:
        return data
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return data
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.encode('utf-8')

if not MAN.exists():
    raise SystemExit('MANIFEST_SHA256.csv missing')

bad = []
n = 0
with MAN.open(newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        n += 1
        p = ROOT / row['path']
        if not p.is_file():
            bad.append((row['path'], 'MISSING'))
            continue
        h = hashlib.sha256(canonical_bytes(p)).hexdigest()
        if h.lower() != row['sha256'].lower():
            bad.append((row['path'], h))

if bad:
    print('Manifest verification FAILED')
    for x in bad[:20]:
        print(x)
    raise SystemExit(1)

print(f'Manifest verification PASS: {n} files')