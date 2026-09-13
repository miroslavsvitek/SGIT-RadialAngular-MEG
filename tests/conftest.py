from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'analysis'
for p in (str(ANALYSIS), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
