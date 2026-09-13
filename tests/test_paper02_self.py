from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_paper02 import COMPONENTS, component_rdm, subject_scores

rng=np.random.default_rng(2026091208)
# exact algebra on arbitrary complex fields
y=(0.2+rng.random((4,16,50)))*np.exp(1j*rng.uniform(-np.pi,np.pi,(4,16,50)))
d,a,r=component_rdm(y)
assert a < 1e-9, (a,r)
assert r < 1e-12, (a,r)
# stable synthetic run fields should produce finite scores for every endpoint
base=(0.4+rng.random((4,16,50)))*np.exp(1j*rng.normal(0,0.5,(4,16,50)))
fields={k:base + 0.02*(rng.normal(size=base.shape)+1j*rng.normal(size=base.shape)) for k in [1,2,3,4,5]}
s,x,a2,r2=subject_scores(fields,'spearman')
assert len(s)==len(COMPONENTS) and np.isfinite(s).all()
assert np.isfinite(x).all()
assert r2 < 1e-12
print('PAPER02 SELF-TEST PASS')
print('max exact absolute audit error =',a)
print('max exact relative audit error =',r)
