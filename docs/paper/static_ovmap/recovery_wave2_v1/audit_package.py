"""Recheck bundle consistency; does not execute native/model experiments.
Run unittest separately to refresh REFERENCE_TEST_LOG.txt before rerunning.
"""
from pathlib import Path
import json,hashlib,re,zipfile
from datetime import datetime, timezone
p=Path(__file__).resolve().parent
s=json.loads((p/'PROTOCOL_SPEC.json').read_text())
main=(p/'CODEX_FINAL_EXECUTION_EN.md').read_text();c=(p/'IMPLEMENTATION_CONTRACTS.md').read_text();zh=(p/'CODE_REVIEW_AND_PLAN_ZH.md').read_text()
log=(p/'REFERENCE_TEST_LOG.txt').read_text()
checks=[]
def check(name,condition,detail):
    checks.append({'check':name,'passed':bool(condition),'detail':detail})
check('immutable_start',s['base_commit']=='c21297413954ecb7d706d1050a8e07822938ea6f' and s['base_commit'] in main,'Full inspected user-repository commit bound.')
check('method_registry',len(s['light_methods'])==8 and len(s['map_variants'])==5,'Two controls, two weight arms, four recovery arms; five map arms.')
check('method_docs',all(r['id'] in main for r in s['light_methods'] if r['family']=='U') and all(r['id'] in main for r in s['map_variants']),'All new U/map IDs named in main instructions.')
check('cohort',len(s['datasets']['development'])==4 and len(s['datasets']['replica'])==8 and s['datasets']['all_exposed'],'Same explicit exposed cohorts; no independent-confirmation claim.')
check('map_budget',len(s['map_variants'])*4+s['selection']['max_map_finalists']*8==28==s['limits']['new_full_map_scene_successes'],'20 development maps + at most8 transfer maps.')
check('no_forced_bad_map_transfer',s['selection']['max_map_finalists']==1 and '.20pp' in main and 'If no map qualifies' in main,'No required losing family transfer.')
check('no_legacy_rerun',s['limits']['new_full_baseline_maps']==0 and 'Do not regenerate the 12' in main,'Historical complete baseline maps reused.')
check('weight_missing_sources', 'partial_sources' in s['weight_rule'] and '6*gamma-2' in main and 'missing' in c,'Endpoints use actual source-availability behavior.')
check('recovery_old_support',s['recovery']['base_geometry_and_existing_owner_masks_immutable'] and 'O[v]=0' in c and not s['recovery']['whole_raw_registry_swap'],'U additions cannot steal existing painted support.')
check('recovery_no_gt_selection',not s['recovery']['candidate_selection_uses_target_coordinates_or_GT'] and 'BEFORE any new inference or GT' in main,'No GT-based candidate/view selection.')
check('matched_FC_control', 'same candidate/request as U1' in s['recovery']['U2'] and 'common-success' in main,'FC vs native tested on same actual request, with coverage disclosed.')
check('paid_query_control', 'RW_UQ_PAID_QUERY' in main and 'no extra query acquisition' in s['recovery']['UQ'],'Existing paid Q recovery is not misattributed to FC.')
check('no_unsupported_projection',not s['recovery']['new_views_from_final_mesh_projection'] and 'NO_LEGAL_CAPTURED_VIEW' in c,'Absent captured views remain explicitly unrecovered.')
check('dense_reuse_not_pooled_reuse','NOW stores' in main and 'pooled vector is not a dense feature map' in main,'Distinguishes current dense cache from older pooled-only caches.')
check('FC_recovery_budget',s['recovery']['FC_additional_image_encoder_input_cap_per_scene']==32 and s['recovery']['candidate_cap_per_scene']==128 and 'initial cache inventory' in c,'Small recovery-only budget; snapshot prevents order-dependent budget expansion.')
check('SAM_saved_representation','not full logits' in main and not s['fixed']['new_SAM_forward'],'No imagined logits or unauthorized SAM rerun.')
check('SAM_crop_priority','all currently CropFormer-labeled pixels' in s['sam']['protect'] and 'Preserve every C>0 pixel' in c,'Protects full candidates rather than creating residual fragments.')
check('causal_conflict',s['sam']['S2_min_owner_stable_previous_snapshots']==2 and 'OWN mapping state' in c and 'Unknown region is never negative' in c,'Two past own-map snapshots; no final-map oracle.')
check('native_fallback_full_path',s['association']['native_fallback_at_all_layers'] and 'Do not preload every historical owner' in main and 'all-USE_NATIVE' in c,'Candidate/count/fresh-label/alias semantics specified.')
check('many_to_one_control','RW_A2_MULTI_FREE' in main and 'RW_A3_MULTI_UNION' in main and s['association']['A3_union_reverse_threshold_strict']==.2,'Separates capacity relaxation from specificity/union constraints.')
check('geometry_screen','SCREENED_OUT_GEOMETRY' in main and s['screening']['geometric_catastrophe']['fragment_factor']==1.75,'Missing semantic measurements not fabricated as zero.')
check('expanded_official_export',s['evaluation']['expanded_registry_supported'] and 'ALL actual output owners' in ' '.join(main.split()) and 'Recalculate official' in ' '.join(main.split()),'New owners enter official masks and ranking context.')
check('frozen_selection',not s['selection']['replica_results_change_selection'] and 'BEFORE any new Replica prediction' in main,'Freeze before transfer; all data still exposed.')
check('truthful_publication',all(x in main for x in s['publication']['reports']) and 'git ls-remote' in main and 'PUSH_VERIFIED' in main,'Actual code/results/report push with external final receipt.')
check('reference_tests',bool(re.search(r'Ran 12 tests',log)) and log.rstrip().endswith('OK'),'12 executed synthetic tests; not a production native or dataset validation.')
check('no_test_count_target','Do not run all repository tests' in main and 'audit-count target' in main,'Scoped checks only, not inherited giant audits.')
check('all_local_markdown_links_exist', all((p/t).exists() for f in p.glob('*.md') for t in re.findall(r'\]\(([^)]+)\)',f.read_text()) if not (t.startswith('https://') or t.startswith('#'))),'Local links resolve; public source links pinned.')
if not all(x['passed'] for x in checks):
    print(json.dumps([x for x in checks if not x['passed']],ensure_ascii=False,indent=2));raise SystemExit(1)
manual=[
 {'issue':'Previous simple gamma formula would change missing-source behavior','resolution':'Interpolate exact FC_EQ/D2 endpoints; missing sources fixed; synthetic test executed.'},
 {'issue':'Old SAM cache does not contain logits','resolution':'Use per-track bit masks and recorded winning pixels; no second-best score reconstruction.'},
 {'issue':'Native-ineligible owner can already have Q evidence','resolution':'Add UQ paid-query control before attributing gains to new FC.'},
 {'issue':'Recovery eligibility was frozen intentionally in wave1','resolution':'Authorize a new append-only U identity; preserve old painted masks and official scoring.'},
 {'issue':'Association fallback can still be broken by global alias tokens or fresh-group restrictions','resolution':'Per-group actions across C++ and hard tokens only on actually constrained labels; real native test assigned to Codex.'},
 {'issue':'A strict one-to-one control cannot isolate group-coverage constraint','resolution':'A1 fallback, A2 free many-to-one, A3 specificity+union; three separate arms.'},
 {'issue':'Geometry-first screen conflicts with old all-readouts-locked diagnose_map','resolution':'Task-local map-lock diagnostic stage and truthful unmeasured semantic status.'},
 {'issue':'Adding candidates can change old ranks despite fixed old masks','resolution':'Recompute all official area ranks and report lost matches.'},
 {'issue':'Past exhaustive schedule spent on an equivalent ratio gate','resolution':'No ratio arm, no forced map transfer, predeclared catastrophic fragmentation stop.'},
 {'issue':'External cache/real data availability is unverified by this assistant','resolution':'Explicit bind stage; incomplete leaf status rather than manufactured coverage; no new benchmark claims.'},
]
audit={'status':'PASS_SPECIFICATION_AUDIT_NOT_BENCHMARK_VALIDATION','task_id':s['task_id'],
       'date':'2026-10-01','automated_checks':checks,'manual_design_review':manual,
       'executed_here':{'synthetic_reference_tests':12,'real_mapper_runs':0,'neural_forward_passes':0,'production_native_compilations':0,'remote_writes':0},
       'unverified':['Target-server assets and GPU availability','Production integration of new algorithms','New development/Replica metrics','Method novelty and independent generalization'],
       'reference_test_log_sha256':hashlib.sha256((p/'REFERENCE_TEST_LOG.txt').read_bytes()).hexdigest()}
(p/'PACKAGE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
files=sorted(f for f in p.iterdir() if f.is_file() and f.name!='MANIFEST.sha256')
(p/'MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n' for f in files))
zpath=p.with_suffix('.zip')
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(p.iterdir()):
        if f.is_file():z.write(f,arcname=p.name+'/'+f.name)
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None
    for f in p.iterdir():
        if f.is_file():assert z.read(p.name+'/'+f.name)==f.read_bytes()
print(json.dumps({'audit_passed':len(checks),'synthetic_tests':12,'zip':str(zpath),'bytes':zpath.stat().st_size,'files':len(files)+1},ensure_ascii=False,indent=2))
