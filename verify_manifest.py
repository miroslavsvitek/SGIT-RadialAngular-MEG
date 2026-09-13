from pathlib import Path
import csv, hashlib, sys
ROOT=Path(__file__).resolve().parent
MAN=ROOT/'MANIFEST_SHA256.csv'
if not MAN.exists():
    raise SystemExit('MANIFEST_SHA256.csv missing')
bad=[]; n=0
with MAN.open(newline='',encoding='utf-8') as f:
    for row in csv.DictReader(f):
        n+=1
        p=ROOT/row['path']
        if not p.is_file():
            bad.append((row['path'],'MISSING'))
            continue
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h.lower()!=row['sha256'].lower():
            bad.append((row['path'],h))
if bad:
    print('Manifest verification FAILED')
    for x in bad[:20]: print(x)
    raise SystemExit(1)
print(f'Manifest verification PASS: {n} files')
