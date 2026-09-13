from __future__ import annotations
import json
from pathlib import Path
from common import load_json,save_json_atomic,resolve_relative,utc_now
from level3a_frozen import run_final as run_l3
from level5a_frozen_dir import run_final as run_l5_strict
from level5a_siteadapted import run_final_siteadapted

HERE=Path(__file__).resolve().parent

def main():
    cfg=load_json(HERE/'config.json');lock=load_json(HERE/'state'/'LEVEL6A_FROZEN_COHORTS.json')
    compact=resolve_relative(HERE/'config.json',cfg['processing']['compact_root']);fine=resolve_relative(HERE/'config.json',cfg['processing']['finetf_root']);out=HERE/'results'/'Level6A';out.mkdir(parents=True,exist_ok=True)
    print('\n=== LEVEL-6A / Level-3A exact CB-prefix replication ===',flush=True)
    l3=run_l3(compact,lock['level6a_level3_exact_subjects'],out/'L3_exact',cfg['level3a']['semantic_permutations'],cfg['level3a']['seed'],cfg['level3a']['batch_size'])
    print('\n=== LEVEL-6A / Level-5A site-adapted CB-prefix replication ===',flush=True)
    l5a=run_final_siteadapted(fine,lock['level6a_level5_siteadapted_subjects'],out/'L5_siteadapted')
    strict_subs=lock['level6a_level5_strict_template_subjects'];strict=None
    if len(strict_subs)>=20:
        print('\n=== LEVEL-6A / Level-5A strict original graph-template replication ===',flush=True)
        strict=run_l5_strict(fine,strict_subs,HERE/'protocol'/'SGIT_Level5A_template_graph.npz',out/'L5_strict_template')
    else:
        strict={'level':'6A-L5-strict-template','status':'INSUFFICIENT_STRICT_GRAPH_COMPATIBLE_CB','N':len(strict_subs),'minimum':20,'scientific_outcome_computed':False}
        save_json_atomic(out/'L5_strict_template_STATUS.json',strict)
    summary={'level':'6A','completed_utc':utc_now(),'cohort_definition':'CB-prefix external cohort; do not call this cross-site unless independent metadata confirm that prefix maps to acquisition site',
             'level3_exact':{'N':l3['n_subjects'],'mean_BA':l3['mean_subject_BA'],'p':l3['p_full_pipeline_semantic_mc'],'decision':l3['decision']},
             'level5_siteadapted':{'N':l5a.get('N_new_evaluable'),'mean_R_FULL':l5a.get('primary',{}).get('mean_FULL_all2v2_rho'),'p':l5a.get('primary',{}).get('signflip_p_one_sided'),'PASS':l5a.get('primary',{}).get('PASS'),'AP_led_PASS':l5a.get('hierarchical_AP_led',{}).get('PASS')},
             'level5_strict_template':{'N':len(strict_subs),'status':strict.get('status'),'PASS':strict.get('primary',{}).get('PASS') if isinstance(strict,dict) else None},
             'guardrail':'6A establishes external-prefix replication only. Site generalization requires explicit site metadata.'}
    save_json_atomic(out/'LEVEL6A_SUMMARY.json',summary);print(json.dumps(summary,indent=2,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
