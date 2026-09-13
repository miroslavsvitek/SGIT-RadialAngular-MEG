from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import decomposed_rdm, partitions_2v2, random_orthogonal, fit_pca_rotation, rotate_modes
from run_controls import shuffle_category_phases, build_field, RunData, reliability


def main():
    assert len(partitions_2v2())==15
    rng=np.random.default_rng(123)
    y=rng.normal(size=(4,16,50))+1j*rng.normal(size=(4,16,50))
    d=decomposed_rdm(y)
    err=np.max(np.abs(d['FULL']-(d['AMP']+d['AP'])))
    assert err < 1e-9, err
    ys,e=shuffle_category_phases(y,rng)
    assert e < 1e-12
    assert np.max(np.abs(np.abs(ys)-np.abs(y))) < 1e-12
    Q=random_orthogonal(16,rng)
    assert np.max(np.abs(Q.T@Q-np.eye(16))) < 1e-10
    raw=rng.normal(size=(20,16,50))
    Qp=fit_pca_rotation(raw)
    assert np.max(np.abs(Qp.T@Qp-np.eye(16))) < 1e-10
    rot=rotate_modes(raw,Q)
    assert rot.shape==raw.shape
    cats=np.array(['face']*5+['object']*5+['letter']*5+['false']*5)
    runs={}
    for r in range(1,6):
        zt=rng.normal(size=(20,16,50))+1j*rng.normal(size=(20,16,50))
        za=rng.normal(size=(20,16,50))+1j*rng.normal(size=(20,16,50))
        rd=RunData(cats,raw.copy(),zt,za,zt.copy(),za.copy())
        runs[r]=build_field(rd,2,'post')
    rel,_=reliability(runs,'spearman')
    assert rel.shape==(3,)
    print('SELF_TEST_PASS')
    print('max decomposition error',err)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
