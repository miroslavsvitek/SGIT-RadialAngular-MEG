from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_paper03 import component_rdm, shuffled_rdms, row_spearman, subject_scores_and_null

rng = np.random.default_rng(2026091211)

y = (0.3 + rng.random((4,16,50))) * np.exp(1j*rng.uniform(-np.pi,np.pi,(4,16,50)))
i, a = component_rdm(y)
assert i.shape == (6,) and a.shape == (6,)
assert np.isfinite(i).all() and np.isfinite(a).all()

ni, na, amp_err, phase_err = shuffled_rdms(y, rng, B=31, batch_size=8)
assert ni.shape == (31,6) and na.shape == (31,6)
assert amp_err < 1e-12, amp_err
assert phase_err < 1e-12, phase_err
assert np.isfinite(row_spearman(ni, ni)).all()

base_amp = 0.8 + 0.15*rng.random((4,16,50))
base_phase = np.empty((4,16,50), float)
for c in range(4):
    base_phase[c] = 0.55*c + 0.12*rng.normal(size=(16,50))
base = base_amp * np.exp(1j*base_phase)
fields = {}
for r in [1,2,3,4,5]:
    amp = np.maximum(0.05, base_amp + 0.01*rng.normal(size=base_amp.shape))
    ph = base_phase + 0.03*rng.normal(size=base_phase.shape)
    fields[r] = amp * np.exp(1j*ph)
obs, null, ae, pe = subject_scores_and_null(fields, B=79, seed=2026091212, batch_size=16)
assert obs.shape == (2,) and null.shape == (2,79)
assert np.isfinite(obs).all() and np.isfinite(null).all()
assert ae < 1e-12 and pe < 1e-12
print('PAPER03 SELF-TEST PASS')
print('synthetic observed INTERFERENCE, ANGLE =', obs.tolist())
print('synthetic null means =', null.mean(axis=1).tolist())
print('amplitude audit =', ae, 'phase multiset audit =', pe)
