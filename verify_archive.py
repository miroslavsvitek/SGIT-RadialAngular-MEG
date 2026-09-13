#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def load(p):
    return json.loads((ROOT/p).read_text(encoding='utf-8'))

def close(a,b,tol=1e-12):
    return math.isclose(float(a),float(b),rel_tol=0,abs_tol=tol)

def main():
    l5=load('results/level5a/SGIT_Level5A_results.json')
    p1=load('results/paper01/ARTICLE_CONTROL_SUMMARY.json')
    p2=load('results/paper02/PAPER02_SUMMARY.json')
    p3=load('results/paper03/PAPER03_SUMMARY.json')
    d2=load('results/paper02/PAPER02_DECISION.json')
    d3=load('results/paper03/PAPER03_DECISION.json')
    checks=[]
    ca='CA_Level5A'; cb='CB_Level6A_strict'
    # Level5A -> PAPER01
    checks += [
      ('L5 FULL matches P1 CA', close(l5['primary']['mean_FULL_all2v2_rho'],p1[ca]['observed_spearman']['FULL']['mean'])),
      ('L5 AP matches P1 CA', close(l5['hierarchical_AP_led']['mean_AP_rho'],p1[ca]['observed_spearman']['AP']['mean'])),
      ('L5 AMP matches P1 CA', close(l5['hierarchical_AP_led']['mean_AMP_rho'],p1[ca]['observed_spearman']['AMP']['mean'])),
    ]
    # PAPER01 -> PAPER02
    for coh in (ca,cb):
        for key in ('FULL','AP','AMP'):
            checks.append((f'P1 {coh} {key} matches P2', close(p1[coh]['observed_spearman'][key]['mean'],p2[coh]['spearman'][key]['mean'])))
    # PAPER02 -> PAPER03 observed scores
    for coh in (ca,cb):
        for key in ('INTERFERENCE','ANGLE'):
            checks.append((f'P2 {coh} {key} matches P3 observed', close(p2[coh]['spearman'][key]['mean'],p3[coh]['components'][key]['observed_mean'])))
        checks.append((f'P3 {coh} reference audit pass', bool(p3[coh]['reference_consistency']['PASS'])))
        checks.append((f'P3 {coh} amplitude audit', p3[coh]['audits']['max_amplitude_error'] < 1e-12))
        checks.append((f'P3 {coh} phase multiset audit', p3[coh]['audits']['max_phase_multiset_error'] < 1e-12))
    checks.append(('PAPER02 decision', d2['decision']=='MIXED_PHASE_AND_PRODUCT'))
    checks.append(('PAPER03 decision', d3['decision']=='PHASE_ORGANIZATION_CONFIRMED'))
    failed=[name for name,ok in checks if not ok]
    print('SGIT archive verification')
    for name,ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if failed:
        raise SystemExit('Archive verification failed: '+', '.join(failed))
    print(f'All {len(checks)} checks passed.')

if __name__=='__main__': main()
